import sys
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from phase2_imbalance import load_and_preprocess

# CHANGED: DEMO MODE. Run `python phase4_explainability.py --demo` for a quick
# (~30 sec) version on a 15% stratified sample - good for showing live to a
# reviewer. It writes to 'stable_features_demo.csv' so it can NEVER overwrite
# the real 'stable_features.csv' that phase5 reads. Quote results from the
# full run (no flag), not from demo mode.
DEMO_FRACTION = 0.15

def audit_feature_stability(file_path, n_splits=3, demo=False):
    X, y, target_encoder = load_and_preprocess(file_path)

    # CHANGED: load_and_preprocess() no longer fills blanks (that now happens
    # after the train/test split to avoid leakage), but LIME crashes on blanks.
    # Many columns are blank because they simply don't apply (e.g. UDP traffic
    # has no TCP window size), so instead of inventing a median we use a
    # constant -1 marker. It needs no statistics from the data (so no leakage)
    # and keeps the "this field didn't apply" signal visible to the model.
    X = X.fillna(-1)

    if demo:
        X, _, y, _ = train_test_split(
            X, y, train_size=DEMO_FRACTION, stratify=y, random_state=42
        )
        X = X.reset_index(drop=True)
        print(f"\n*** DEMO MODE: using a {DEMO_FRACTION:.0%} stratified sample ({len(X)} rows). "
              "Numbers are illustrative - use the full run for reported results. ***")
    
    # XGBoost is used for fast tree-based SHAP implementations
    model = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    feature_names = X.columns.tolist()
    fold_importances = pd.DataFrame(index=feature_names)
    
    print("\n==========================================")
    print("   SHAP Cross-Fold Stability Audit")
    print("==========================================")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training and computing SHAP for Fold {fold + 1}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        # Safely handle y indexing depending on whether it's a Series or numpy array
        if isinstance(y, pd.Series):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        
        # Calculate SHAP values (using a sample to speed up the audit)
        X_val_sample = X_val.sample(n=min(1000, len(X_val)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_sample)
        
        # Handle Multiclass vs Binary SHAP array outputs
        if isinstance(shap_values, list):
            # Multiclass: Average feature impact across all classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                mean_shap = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
        # Rank features for this fold (1 is most important)
        fold_importances[f'Fold_{fold+1}_Impact'] = mean_shap
        fold_importances[f'Fold_{fold+1}_Rank'] = fold_importances[f'Fold_{fold+1}_Impact'].rank(ascending=False)

    # Calculate stability (Standard Deviation of Ranks)
    rank_cols = [c for c in fold_importances.columns if 'Rank' in c]
    fold_importances['Average_Rank'] = fold_importances[rank_cols].mean(axis=1)
    fold_importances['Rank_Std_Dev'] = fold_importances[rank_cols].std(axis=1)
    
    # Sort by overall most important
    stable_features = fold_importances.sort_values(by='Average_Rank')
    
    print("\n--- Top 10 Most Stable Features ---")
    # Low Std Dev means the feature consistently stays at that rank across all folds
    print(stable_features[['Average_Rank', 'Rank_Std_Dev']].head(10).to_string())

    # CHANGED: added a Jaccard top-10 overlap metric. Rank_Std_Dev is a fine
    # stability signal, but this is what you've actually been telling the
    # review panel ("Jaccard similarity of top-10 features across folds") —
    # the code should produce the number you're citing out loud.
    top10_per_fold = []
    for fold in range(1, n_splits + 1):
        impact_col = f'Fold_{fold}_Impact'
        top10 = set(fold_importances[impact_col].sort_values(ascending=False).head(10).index)
        top10_per_fold.append(top10)

    jaccard_scores = []
    for i in range(len(top10_per_fold)):
        for j in range(i + 1, len(top10_per_fold)):
            inter = len(top10_per_fold[i] & top10_per_fold[j])
            union = len(top10_per_fold[i] | top10_per_fold[j])
            jaccard_scores.append(inter / union if union else 0)
    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else float('nan')
    print(f"\nAverage pairwise Jaccard overlap of top-10 features across folds: {avg_jaccard:.3f}")
    print("(Closer to 1.0 = more stable; this is the number to quote to the panel.)")

    # CHANGED: mark features as "Stable" using a concrete, explainable rule
    # (rank std-dev under 3 positions) and export the ranked list to CSV.
    # phase5_mitigation.py now reads this file instead of a hardcoded guess.
    STABILITY_THRESHOLD = 3.0
    stable_features['Stable'] = stable_features['Rank_Std_Dev'] <= STABILITY_THRESHOLD
    out_csv = 'stable_features_demo.csv' if demo else 'stable_features.csv'
    stable_features.reset_index().rename(columns={'index': 'Feature'}).to_csv(
        out_csv, index=False
    )
    print(f"\nSaved full ranked, stability-flagged feature list to '{out_csv}'")
    print(f"(features with Rank_Std_Dev <= {STABILITY_THRESHOLD} are flagged Stable=True)")

    print("\n==========================================")
    print("   LIME Local Explanation Sanity Check")
    print("==========================================")
    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=[str(c) for c in target_encoder.classes_],
        mode='classification',
        # CHANGED: LIME's default "bin continuous features into quartiles" step
        # crashes (scipy 'scale must be positive') when a feature has no
        # variation inside a bin - common in network data (e.g. a flag that is
        # always 0). Turning discretization off avoids it and still gives the
        # same kind of top-feature explanation.
        discretize_continuous=False
    )
    
    # Pick a sample instance (e.g., the first instance in the validation set)
    instance_idx = 0
    test_instance = X_val.iloc[instance_idx]
    
    if isinstance(y_val, pd.Series):
        true_label_idx = y_val.iloc[instance_idx]
    else:
        true_label_idx = y_val[instance_idx]
        
    true_label = target_encoder.classes_[true_label_idx]
    predicted_label = target_encoder.classes_[model.predict(X_val.iloc[[instance_idx]])[0]]
    
    print(f"Explaining instance {instance_idx}:")
    print(f"True Label: {true_label} | Predicted: {predicted_label}")
    
    # Generate explanation
    # CHANGED: LIME explains class index 1 by default, which is NOT necessarily
    # the class the model predicted. We now explain the model's actual predicted
    # class so the weights answer "why did it say Benign (or whatever it said)?"
    pred_idx = int(model.predict(X_val.iloc[[instance_idx]])[0])
    exp = explainer_lime.explain_instance(
        data_row=test_instance.values, 
        predict_fn=model.predict_proba,
        num_features=5,
        labels=(pred_idx,)
    )
    
    print(f"\nLIME top feature contributions toward '{target_encoder.classes_[pred_idx]}' for this specific flow:")
    for feature, weight in exp.as_list(label=pred_idx):
        print(f"{feature}: {weight:.4f}")

    return stable_features

if __name__ == "__main__":
    audit_feature_stability('Combined.csv', demo='--demo' in sys.argv)