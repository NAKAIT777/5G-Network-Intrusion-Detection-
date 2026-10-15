import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
# CHANGED: no longer re-implements its own copy of the cleaning/leakage logic.
# The old version here was a separate copy-pasted block that didn't drop
# 'Offset' — meaning it would have kept producing the inflated 0.999 AUC and
# the suspicious SHAP chart even after phase2 got fixed, since nobody would
# remember to fix it in three places at once. Now it just imports the one
# real version.
from phase2_imbalance import load_and_preprocess

def create_visualizations(file_path):
    print("Loading data for visualization...")
    # load_and_preprocess() returns UNSCALED X (scaling happens later, inside
    # prepare_data_splits) — which is exactly what we want here, so the SHAP
    # plots keep showing real network values like TTL = 57 instead of
    # standardized numbers. This also means Offset/RunTime/Seq etc. are
    # already dropped, matching the real modeling pipeline.
    X, y_encoded, target_encoder = load_and_preprocess(file_path)

    # Create a binary target for the ROC curve (0 = Benign, 1 = Attack)
    benign_idx = list(target_encoder.classes_).index('Benign')
    y_binary = np.where(y_encoded == benign_idx, 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.1, random_state=42, stratify=y_binary)
    
    print("Training XGBoost model for graphics extraction...")
    model = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # Visual 1 & 2: SHAP Explanations
    # ---------------------------------------------------------
    print("Generating SHAP plots (this may take a moment)...")
    # Take a representative sample to render plots quickly
    X_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Plot 1: SHAP Bar Chart (Global Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Top Network Signatures (Global Feature Importance)", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_bar_chart.png", dpi=300)
    plt.close()
    print("Saved: shap_bar_chart.png")

    # Plot 2: SHAP Beeswarm Plot (Impact Direction)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("How Features Drive Predictions (SHAP Summary)", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_summary_beeswarm.png", dpi=300)
    plt.close()
    print("Saved: shap_summary_beeswarm.png")

    # ---------------------------------------------------------
    # Visual 3: ROC Curve (Benign vs Malicious)
    # ---------------------------------------------------------
    print("Generating ROC Curve...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Benign Dropped)', fontsize=12)
    plt.ylabel('True Positive Rate (Attacks Blocked)', fontsize=12)
    plt.title('Detection Performance: Attack vs. Benign Traffic', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()
    print("Saved: roc_curve.png")
    print("\n✅ All presentation graphics successfully generated in your folder!")

if __name__ == "__main__":
    create_visualizations('Combined.csv')