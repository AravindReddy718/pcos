"""
Phase 2: Train and evaluate classical ML models using the SAME 
preprocessing pipeline and LOCKED train/test split.
"""

import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from data_pipeline import load_dataax, preprocess_data

# Strict reproducibility
np.random.seed(0)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model. Returns metrics dict."""
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # SVM with decision_function
        y_prob = model.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Confusion Matrix: {cm}")
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }


def main():
    print("=" * 60)
    print("PHASE 2: Classical ML Model Evaluation")
    print("=" * 60)
    
    # Load data using LOCKED pipeline
    df = load_dataax()
    X_train_dl, X_test_dl, y_train, y_test, scaler = preprocess_data(df, save_artifacts=False)
    
    # Flatten for classical models (remove DL channel dimension)
    X_train_2d = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_test_2d = X_test_dl.reshape(X_test_dl.shape[0], -1)
    
    print(f"Train shape: {X_train_2d.shape}, Test shape: {X_test_2d.shape}")
    
    # Define models with fixed random states
    models = {
        "Logistic Regression": LogisticRegression(random_state=0, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel='rbf', random_state=0, probability=True),
        "Random Forest": RandomForestClassifier(random_state=0),
        "XGBoost": XGBClassifier(
            random_state=0, 
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_train_2d, X_test_2d, y_train, y_test)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/classical_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
