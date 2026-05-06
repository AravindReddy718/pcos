"""
Phase 3: Train and evaluate additional deep learning model variants 
using the SAME preprocessing pipeline and LOCKED train/test split.
"""

import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from data_pipeline import load_dataax, preprocess_data

# Strict reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_mlp(input_dim):
    """Simple MLP: 64 -> 32 -> 16 -> 1"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_deep(input_shape):
    """Simple CNN: 32 filters, max 2 layers"""
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        Flatten(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_pure_lstm(input_shape):
    """Minimal LSTM"""
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_dl_model(name, model, X_train, X_val, y_train, y_val, X_test, y_test, epochs=100):
    """Train and evaluate a single DL model."""
    print(f"\n--- Training {name} ---")
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
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
    print("PHASE 3: Deep Learning Variant Evaluation")
    print("=" * 60)
    
    # Load data using LOCKED pipeline
    df = load_dataax()
    X_train_dl, X_test_dl, y_train, y_test, scaler = preprocess_data(df, save_artifacts=False)
    
    # Create validation split from training data ONLY
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_dl, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )
    
    print(f"Train: {X_train_split.shape}, Val: {X_val_split.shape}, Test: {X_test_dl.shape}")
    
    input_shape_3d = (X_train_split.shape[1], 1)  # (features, 1)
    input_dim_2d = X_train_split.shape[1]          # features (flat)
    
    # Flatten for MLP
    X_train_2d = X_train_split.reshape(X_train_split.shape[0], -1)
    X_val_2d = X_val_split.reshape(X_val_split.shape[0], -1)
    X_test_2d = X_test_dl.reshape(X_test_dl.shape[0], -1)
    
    results = {}
    
    # 1. MLP
    mlp = build_mlp(input_dim_2d)
    results["MLP"] = evaluate_dl_model(
        "MLP", mlp,
        X_train_2d, X_val_2d, y_train_split, y_val_split,
        X_test_2d, y_test
    )
    
    # 2. Deeper 1D CNN 
    cnn_deep = build_cnn_deep(input_shape_3d)
    results["1D CNN (Deep)"] = evaluate_dl_model(
        "1D CNN (Deep)", cnn_deep,
        X_train_split, X_val_split, y_train_split, y_val_split,
        X_test_dl, y_test
    )
    
    # 3. Pure LSTM
    lstm = build_pure_lstm(input_shape_3d)
    results["Pure LSTM"] = evaluate_dl_model(
        "Pure LSTM", lstm,
        X_train_split, X_val_split, y_train_split, y_val_split,
        X_test_dl, y_test
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/dl_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
