"""
Phase 6: Visualization & Reporting
Aggregates all results, generates comparison tables, ROC curves, 
confusion matrices, and exports everything for report inclusion.
"""

import numpy as np
import json
import os
import csv
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
from data_pipeline import load_dataax, preprocess_data

np.random.seed(0)

# ────────────────────────────────────────
# HELPER: Collect all probabilities for ROC
# ────────────────────────────────────────

def get_all_model_probs(X_train_dl, X_test_dl, y_train, y_test):
    """Collect predicted probabilities from all major models."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    X_train_2d = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_test_2d = X_test_dl.reshape(X_test_dl.shape[0], -1)

    probs = {}

    # Classical models
    models_2d = {
        "Logistic Regression": LogisticRegression(random_state=0, max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel='rbf', random_state=0, probability=True),
        "Random Forest": RandomForestClassifier(random_state=0),
        "XGBoost": XGBClassifier(random_state=0, eval_metric='logloss', use_label_encoder=False)
    }

    for name, model in models_2d.items():
        model.fit(X_train_2d, y_train)
        if hasattr(model, 'predict_proba'):
            probs[name] = model.predict_proba(X_test_2d)[:, 1]
        else:
            probs[name] = model.decision_function(X_test_2d)

    # DL models — load saved baseline models
    cnn = load_model('cnn_model.h5')
    cnn_lstm = load_model('cnn_lstm_model.h5')
    probs["CNN (Baseline)"] = cnn.predict(X_test_dl, verbose=0).flatten()
    probs["CNN-LSTM (Baseline)"] = cnn_lstm.predict(X_test_dl, verbose=0).flatten()

    # Ensemble
    rf_probs = probs["Random Forest"]
    cnn_probs = probs["CNN (Baseline)"]
    cnn_lstm_probs = probs["CNN-LSTM (Baseline)"]
    probs["Proposed Ensemble"] = (0.4 * rf_probs) + (0.3 * cnn_probs) + (0.3 * cnn_lstm_probs)

    return probs


def main():
    print("=" * 60)
    print("PHASE 6: Visualization & Reporting")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)

    # Load data
    df = load_dataax()
    X_train_dl, X_test_dl, y_train, y_test, scaler = preprocess_data(df, save_artifacts=False)
    y_test_arr = np.array(y_test)

    # ────────────────────────────────────────
    # 1. Master Comparison Table
    # ────────────────────────────────────────
    print("\n--- Building Master Comparison Table ---")

    all_results = {}

    # Load classical metrics
    with open('results/classical_metrics.json') as f:
        classical = json.load(f)
    for name, metrics in classical.items():
        all_results[name] = metrics

    # Load DL metrics
    with open('results/dl_metrics.json') as f:
        dl = json.load(f)
    for name, metrics in dl.items():
        all_results[name] = metrics

    # Load ensemble (baseline) results  
    with open('final_results.json') as f:
        ensemble = json.load(f)
    all_results["Proposed Hybrid Ensemble (Primary)"] = {
        "accuracy": round(ensemble["accuracy"], 4),
        "precision": round(ensemble["precision"], 4),
        "recall": round(ensemble["recall"], 4),
        "f1_score": round(ensemble["f1"], 4),
        "roc_auc": round(ensemble["auc"], 4)
    }

    # Write CSV
    csv_path = 'results/master_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
        for name, m in all_results.items():
            writer.writerow([
                name,
                f"{m['accuracy']*100:.2f}%",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['roc_auc']:.4f}"
            ])
    print(f"  Saved: {csv_path}")

    # Print table
    print(f"\n{'Model':<40} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 80)
    for name, m in all_results.items():
        print(f"{name:<40} {m['accuracy']*100:>7.2f}% {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f}")

    # ────────────────────────────────────────
    # 2. Ablation Study Table (CSV)
    # ────────────────────────────────────────
    print("\n--- Building Ablation Study Table ---")
    with open('results/ablation_metrics.json') as f:
        ablation = json.load(f)

    # Add Primary Baseline for comparison
    ablation["Ensemble (Primary Baseline - Frozen)"] = {
        "accuracy": ensemble["accuracy"],
        "precision": ensemble["precision"],
        "recall": ensemble["recall"],
        "f1_score": ensemble["f1"],
        "roc_auc": ensemble["auc"]
    }
    
    # Sort keys to put Primary Baseline at the top or bottom
    # We'll just write them all
    
    csv_path_abl = 'results/ablation_comparison.csv'
    with open(csv_path_abl, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
        
        # Write Primary first
        m = ablation["Ensemble (Primary Baseline - Frozen)"]
        writer.writerow([
            "Ensemble (Primary Baseline - Frozen)",
            f"{m['accuracy']*100:.2f}%",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['roc_auc']:.4f}"
        ])
        
        for name, m in ablation.items():
            if name == "Ensemble (Primary Baseline - Frozen)":
                continue
            writer.writerow([
                name,
                f"{m['accuracy']*100:.2f}%",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['roc_auc']:.4f}"
            ])
    print(f"  Saved: {csv_path_abl}")

    # ────────────────────────────────────────
    # 3. ROC Curves
    # ────────────────────────────────────────
    print("\n--- Generating ROC Curves ---")
    all_probs = get_all_model_probs(X_train_dl, X_test_dl, y_train, y_test)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_probs)))

    for (name, y_prob), color in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/roc_curves.png")

    # ────────────────────────────────────────
    # 4. Confusion Matrices
    # ────────────────────────────────────────
    print("\n--- Generating Confusion Matrices ---")

    # Select key models for confusion matrix display
    key_models = {
        "Logistic Regression": all_probs["Logistic Regression"],
        "SVM (RBF)": all_probs["SVM (RBF)"],
        "Random Forest": all_probs["Random Forest"],
        "XGBoost": all_probs["XGBoost"],
        "CNN (Baseline)": all_probs["CNN (Baseline)"],
        "Proposed Ensemble": all_probs["Proposed Ensemble"]
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices — Key Models', fontsize=16, fontweight='bold')

    for ax, (name, y_prob) in zip(axes.flatten(), key_models.items()):
        y_pred = (np.array(y_prob) >= 0.5).astype(int)
        cm = confusion_matrix(y_test_arr, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No PCOS', 'PCOS'],
                    yticklabels=['No PCOS', 'PCOS'])
        ax.set_title(name, fontsize=11)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/confusion_matrices.png")

    # ────────────────────────────────────────
    # 5. Feature Importance Chart
    # ────────────────────────────────────────
    print("\n--- Generating Feature Importance Chart ---")
    with open('results/feature_importance.json') as f:
        fi_data = json.load(f)

    features = [f["feature"] for f in fi_data["full_ranking"]]
    importances = [f["mean_abs_shap"] for f in fi_data["full_ranking"]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(features)), importances, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(features))))
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mean |SHAP value|', fontsize=12)
    plt.title('Feature Importance (SHAP — Random Forest)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/feature_importance_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/feature_importance_chart.png")

    print(f"\n{'=' * 60}")
    print("All reports and visualizations generated successfully.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
