"""
Phase 4: Ablation Study
Compare the impact of SMOTEENN resampling on model performance.
Uses LOCKED split indices for fair comparison.

Experiments:
1. dnn without SMOTE    vs  dnn with SMOTEENN
2. CNN without SMOTE   vs  CNN with SMOTEENN
3. Ensemble w/o SMOTE  vs  Ensemble with SMOTEENN
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from data_pipeline import load_dataax, preprocess_data

# Strict reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def preprocess_no_smote(df):
    """
    Preprocess data WITHOUT SMOTEENN.
    Uses the same feature selection and scaling logic but skips resampling.
    """
    target = "PCOS (Y/N)"
    selected_features = [
        "Follicle No. (R)",
        "Follicle No. (L)",
        "Skin darkening (Y/N)",
        "hair growth(Y/N)",
        "Weight gain(Y/N)",
        "Cycle(R/I)"
    ]

    cols_to_use = selected_features + [target]
    df_subset = df[cols_to_use].copy()

    X = df_subset.drop(columns=[target])
    y = df_subset[target]

    # Split WITHOUT SMOTE — use the same random_state for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for DL
    X_train_dl = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_dl = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    return X_train_dl, X_test_dl, y_train, y_test


def build_cnn(input_shape):
    """Same CNN architecture as the baseline."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm(input_shape):
    """Same CNN-LSTM architecture as the baseline."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(100),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_metrics(y_test, y_pred, y_prob):
    """Calculate standard metrics."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }


def train_and_eval_rf(X_train_dl, X_test_dl, y_train, y_test):
    """Train RF and return metrics."""
    X_tr = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_te = X_test_dl.reshape(X_test_dl.shape[0], -1)
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_tr, y_train)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    return get_metrics(y_test, y_pred, y_prob)


def train_and_eval_cnn(X_train_dl, X_test_dl, y_train, y_test):
    """Train CNN and return metrics."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_dl, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    input_shape = (X_tr.shape[1], 1)
    cnn = build_cnn(input_shape)
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    cnn.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    y_prob = cnn.predict(X_test_dl, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    return get_metrics(y_test, y_pred, y_prob)


def train_and_eval_ensemble(X_train_dl, X_test_dl, y_train, y_test):
    """Train full ensemble (RF + CNN + CNN-LSTM) and return metrics."""
    X_tr = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_te = X_test_dl.reshape(X_test_dl.shape[0], -1)

    # RF
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_tr, y_train)
    rf_probs = rf.predict_proba(X_te)[:, 1]

    # CNN
    X_tr_split, X_val, y_tr_split, y_val = train_test_split(
        X_train_dl, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    input_shape = (X_tr_split.shape[1], 1)

    cnn = build_cnn(input_shape)
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    cnn.fit(X_tr_split, y_tr_split, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    cnn_probs = cnn.predict(X_test_dl, verbose=0).flatten()

    # CNN-LSTM
    cnn_lstm = build_cnn_lstm(input_shape)
    cnn_lstm.fit(X_tr_split, y_tr_split, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[es], verbose=0)
    cnn_lstm_probs = cnn_lstm.predict(X_test_dl, verbose=0).flatten()

    # Weighted ensemble
    ensemble_probs = (0.4 * rf_probs) + (0.3 * cnn_probs) + (0.3 * cnn_lstm_probs)
    y_pred = (ensemble_probs >= 0.5).astype(int)

    return get_metrics(y_test, y_pred, ensemble_probs)


def main():
    print("=" * 60)
    print("PHASE 4: Ablation Study (SMOTEENN Impact)")
    print("=" * 60)

    df = load_dataax()

    # --- WITHOUT SMOTEENN ---
    print("\n🔬 Processing WITHOUT SMOTEENN...")
    X_train_no, X_test_no, y_train_no, y_test_no = preprocess_no_smote(df)
    print(f"  No-SMOTE Train: {X_train_no.shape}, Test: {X_test_no.shape}")

    # --- WITH SMOTEENN (locked pipeline) ---
    print("\n🔬 Processing WITH SMOTEENN (locked split)...")
    X_train_sm, X_test_sm, y_train_sm, y_test_sm, _ = preprocess_data(df, save_artifacts=False)
    print(f"  SMOTEENN Train: {X_train_sm.shape}, Test: {X_test_sm.shape}")

    results = {}

    # RF Ablation
    print("\n--- RF without SMOTE ---")
    results["RF (Retrained No SMOTE)"] = train_and_eval_rf(X_train_no, X_test_no, y_train_no, y_test_no)
    print(f"  Accuracy: {results['RF (Retrained No SMOTE)']['accuracy']*100:.2f}%")

    print("\n--- RF with SMOTEENN ---")
    results["RF (Retrained SMOTE)"] = train_and_eval_rf(X_train_sm, X_test_sm, y_train_sm, y_test_sm)
    print(f"  Accuracy: {results['RF (Retrained SMOTE)']['accuracy']*100:.2f}%")

    # CNN Ablation
    print("\n--- CNN without SMOTE ---")
    results["CNN (Retrained No SMOTE)"] = train_and_eval_cnn(X_train_no, X_test_no, y_train_no, y_test_no)
    print(f"  Accuracy: {results['CNN (Retrained No SMOTE)']['accuracy']*100:.2f}%")

    print("\n--- CNN with SMOTEENN ---")
    results["CNN (Retrained SMOTE)"] = train_and_eval_cnn(X_train_sm, X_test_sm, y_train_sm, y_test_sm)
    print(f"  Accuracy: {results['CNN (Retrained SMOTE)']['accuracy']*100:.2f}%")

    # Ensemble Ablation
    print("\n--- Ensemble without SMOTE ---")
    results["Ensemble (Retrained No SMOTE)"] = train_and_eval_ensemble(X_train_no, X_test_no, y_train_no, y_test_no)
    print(f"  Accuracy: {results['Ensemble (Retrained No SMOTE)']['accuracy']*100:.2f}%")

    print("\n--- Ensemble with SMOTEENN ---")
    results["Ensemble (Retrained SMOTE)"] = train_and_eval_ensemble(X_train_sm, X_test_sm, y_train_sm, y_test_sm)
    print(f"  Accuracy: {results['Ensemble (Retrained SMOTE)']['accuracy']*100:.2f}%")

    # Save
    os.makedirs('results', exist_ok=True)
    output_path = 'results/ablation_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'Accuracy':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*60}")
    for name, m in results.items():
        print(f"{name:<25} {m['accuracy']*100:>9.2f}% {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
    print(f"{'=' * 60}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
