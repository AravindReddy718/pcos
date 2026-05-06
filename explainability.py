"""
Phase 5: Explainable AI using SHAP for Random Forest
Generates:
- SHAP summary plot (global feature importance)
- SHAP bar plot
- SHAP waterfall plots (one positive, one negative case)
- Feature importance JSON for Flask app
"""

import numpy as np
import json
import os
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from data_pipeline import load_dataax, preprocess_data

# Strict reproducibility
np.random.seed(0)

# Feature names used in the pipeline
FEATURE_NAMES = [
    "Follicle No. (R)",
    "Follicle No. (L)",
    "Skin darkening (Y/N)",
    "hair growth(Y/N)",
    "Weight gain(Y/N)",
    "Cycle(R/I)"
]


def main():
    print("=" * 60)
    print("PHASE 5: Explainable AI (SHAP Analysis)")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)

    # Load data using LOCKED pipeline
    df = load_dataax()
    X_train_dl, X_test_dl, y_train, y_test, scaler = preprocess_data(df, save_artifacts=False)

    # Flatten for RF
    X_train_2d = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_test_2d = X_test_dl.reshape(X_test_dl.shape[0], -1)

    # Load the FROZEN RF model
    rf_model = joblib.load('rf_model.joblib')
    print("RF model loaded from rf_model.joblib")

    # --- SHAP Analysis ---
    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_2d)

    # For binary classification with SHAP v0.49+:
    # shap_values can be a 3D array of shape (n_samples, n_features, n_classes)
    # or a list of two arrays. We need class-1 values for PCOS positive.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals_pos = shap_values[:, :, 1]  # class 1 (PCOS positive)
        base_value = explainer.expected_value[1]
    elif isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_pos = shap_values
        base_value = explainer.expected_value

    # --- 1. SHAP Summary Plot ---
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_pos, X_test_2d,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.tight_layout()
    plt.savefig('results/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/shap_summary.png")

    # --- 2. SHAP Bar Plot ---
    print("Generating SHAP Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_pos, X_test_2d,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig('results/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/shap_bar.png")

    # --- 3. Waterfall Plots ---
    # Find one positive and one negative case
    y_test_arr = np.array(y_test)
    pos_indices = np.where(y_test_arr == 1)[0]
    neg_indices = np.where(y_test_arr == 0)[0]

    # Positive case waterfall
    if len(pos_indices) > 0:
        pos_idx = pos_indices[0]
        print(f"Generating Waterfall Plot for PCOS-positive case (index {pos_idx})...")
        
        explanation_pos = shap.Explanation(
            values=shap_vals_pos[pos_idx],
            base_values=float(base_value),
            data=X_test_2d[pos_idx],
            feature_names=FEATURE_NAMES
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation_pos, show=False)
        plt.tight_layout()
        plt.savefig('results/shap_waterfall_pos.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: results/shap_waterfall_pos.png")

    # Negative case waterfall
    if len(neg_indices) > 0:
        neg_idx = neg_indices[0]
        print(f"Generating Waterfall Plot for PCOS-negative case (index {neg_idx})...")
        
        explanation_neg = shap.Explanation(
            values=shap_vals_pos[neg_idx],
            base_values=float(base_value),
            data=X_test_2d[neg_idx],
            feature_names=FEATURE_NAMES
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation_neg, show=False)
        plt.tight_layout()
        plt.savefig('results/shap_waterfall_neg.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: results/shap_waterfall_neg.png")

    # --- 4. Feature Importance Table ---
    print("\nComputing Global Feature Importance...")
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    
    # Sort by importance
    importance_order = np.argsort(mean_abs_shap)[::-1]
    
    feature_importance = []
    for i, idx in enumerate(importance_order):
        entry = {
            "rank": i + 1,
            "feature": FEATURE_NAMES[idx],
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6)
        }
        feature_importance.append(entry)
        print(f"  #{i+1}: {FEATURE_NAMES[idx]} (SHAP: {mean_abs_shap[idx]:.6f})")

    # Save feature importance for Flask app
    importance_data = {
        "top_features": [f["feature"] for f in feature_importance[:3]],
        "full_ranking": feature_importance
    }
    
    with open('results/feature_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=4)
    print("\n  Saved: results/feature_importance.json")

    print(f"\n{'=' * 60}")
    print("SHAP analysis complete. All plots and data saved.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
