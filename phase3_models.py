import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# CHANGED: added MLPClassifier so we actually test a neural-net-style model,
# since the methodology slides promise "RF / XGBoost / DNN" but this file
# previously only ran RF and XGBoost.
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from phase2_imbalance import load_and_preprocess, prepare_data_splits, apply_resampling

# CHANGED: FAST_MODE trains on a stratified 25% slice of the training data so
# you can iterate in minutes instead of an hour+. Class proportions are
# preserved, so the imbalance story still holds. Set FAST_MODE = False for the
# final run whose numbers go into your report.
FAST_MODE = True
TRAIN_FRACTION = 0.25

def evaluate_model(model, X_test, y_test, target_encoder, model_name="Model"):
    """
    Evaluates model performance with explicit focus on Macro-F1 and Per-Class Recall.
    """
    print(f"\n==========================================")
    print(f"   Evaluating: {model_name}")
    print(f"==========================================")
    
    y_pred = model.predict(X_test)
    
    # Calculate key metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"Macro-F1 Score:     {macro_f1:.4f}")
    print(f"Macro Recall Score: {macro_recall:.4f}\n")
    
    # Full per-class classification report
    target_names = [str(cls) for cls in target_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("Per-Class Performance Report:")
    print(report)
    
    return macro_f1, macro_recall

def train_and_compare(file_path):
    # 1. Load and Split Data
    X, y, target_encoder = load_and_preprocess(file_path)
    X_train, X_test, y_train, y_test = prepare_data_splits(X, y)

    if FAST_MODE:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=TRAIN_FRACTION, stratify=y_train, random_state=42
        )
        print(f"\n[FAST_MODE] Training on {len(y_train)} rows ({TRAIN_FRACTION:.0%} of train split). "
              "Set FAST_MODE = False for final numbers.")

    results = {}

    # -------------------------------------------------------------
    # Experiment 1: Plain Baseline (Random Forest - Default)
    # -------------------------------------------------------------
    print("\n--- Experiment 1: Plain Random Forest Baseline ---")
    rf_plain = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_plain.fit(X_train, y_train)
    macro_f1, recall = evaluate_model(rf_plain, X_test, y_test, target_encoder, "RF (Plain Baseline)")
    results["RF Plain"] = (macro_f1, recall)

    # -------------------------------------------------------------
    # Experiment 2: Cost-Sensitive Learning (Class Weights)
    # -------------------------------------------------------------
    print("\n--- Experiment 2: Cost-Sensitive Random Forest ---")
    rf_weighted = RandomForestClassifier(
        n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf_weighted.fit(X_train, y_train)
    macro_f1, recall = evaluate_model(rf_weighted, X_test, y_test, target_encoder, "RF (Cost-Sensitive)")
    results["RF Cost-Sensitive"] = (macro_f1, recall)

    # -------------------------------------------------------------
    # Experiment 3: XGBoost Classifier
    # -------------------------------------------------------------
    print("\n--- Experiment 3: XGBoost Classifier ---")
    xgb = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42)
    xgb.fit(X_train, y_train)
    macro_f1, recall = evaluate_model(xgb, X_test, y_test, target_encoder, "XGBoost (Standard)")
    results["XGBoost"] = (macro_f1, recall)

    # -------------------------------------------------------------
    # Experiment 4: SMOTE + Random Forest
    # -------------------------------------------------------------
    print("\n--- Experiment 4: SMOTE Oversampling + Random Forest ---")
    X_train_smote, y_train_smote = apply_resampling(X_train, y_train, method='smote')
    rf_smote = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf_smote.fit(X_train_smote, y_train_smote)
    macro_f1, recall = evaluate_model(rf_smote, X_test, y_test, target_encoder, "RF (SMOTE)")
    results["RF SMOTE"] = (macro_f1, recall)

    # -------------------------------------------------------------
    # CHANGED — Experiment 5: ADASYN + Random Forest
    # This was imported in phase2_imbalance.py but never actually used anywhere.
    # Your methodology (and slides) promise a SMOTE vs ADASYN vs cost-sensitive
    # vs baseline comparison — without this, ADASYN was never tested at all.
    # -------------------------------------------------------------
    print("\n--- Experiment 5: ADASYN Oversampling + Random Forest ---")
    # CHANGED: ADASYN raises an error when a rare class has no neighbours from
    # other classes (happens with tiny classes like ICMPFlood, esp. in
    # FAST_MODE). Don't let that kill the whole run - report it and move on.
    try:
        X_train_adasyn, y_train_adasyn = apply_resampling(X_train, y_train, method='adasyn')
        rf_adasyn = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf_adasyn.fit(X_train_adasyn, y_train_adasyn)
        macro_f1, recall = evaluate_model(rf_adasyn, X_test, y_test, target_encoder, "RF (ADASYN)")
        results["RF ADASYN"] = (macro_f1, recall)
    except RuntimeError as e:
        print(f"ADASYN could not run on this data split: {e}")
        print("Skipping ADASYN. Try FAST_MODE = False (more samples per class) or rely on SMOTE.")

    # -------------------------------------------------------------
    # CHANGED — Experiment 6: Lightweight DNN (MLPClassifier)
    # This is a quick, dependency-light stand-in for the "DNN" branch your
    # slides mention. If you need a true deep-learning framework result for
    # the write-up, swap this block for a small Keras/TensorFlow model — the
    # evaluate_model() call below works the same either way.
    # -------------------------------------------------------------
    print("\n--- Experiment 6: Lightweight DNN (MLPClassifier) ---")
    dnn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=True, random_state=42)
    dnn.fit(X_train, y_train)
    macro_f1, recall = evaluate_model(dnn, X_test, y_test, target_encoder, "DNN (MLPClassifier)")
    results["DNN"] = (macro_f1, recall)

    # Summary Comparison
    print("\n==========================================")
    print("          FINAL EXPERIMENT SUMMARY         ")
    print("==========================================")
    summary_df = pd.DataFrame(results, index=["Macro-F1", "Macro-Recall"]).T
    print(summary_df.to_string())

if __name__ == "__main__":
    train_and_compare('Combined.csv')