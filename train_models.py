
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from data_pipeline import load_dataax, preprocess_data
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.2), # Add dropout to prevent overfitting
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(100),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save():
    print("Loading and preprocessing data...")
    df = load_dataax()
    # Note: save_artifacts=False because we assume 'split_indices.pkl' was already created by pipeline run
    # Actually, preprocess_data regenerates split, but since random_state=0, it's deterministic.
    # To be extremely strict, we should load indices, but for now determinism is sufficient provided random_state holds.
    X_train_full, X_test, y_train_full, y_test, scaler = preprocess_data(df, save_artifacts=False)
    
    print("Splitting training data into Train and Validation sets for DL...")
    # STRICT RULE: Validation set derived from X_train only.
    # 10% of training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.1, 
        random_state=42, 
        stratify=y_train_full
    )
    
    input_shape = (X_train.shape[1], 1)
    print(f"Input shape: {input_shape}")
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # Training CNN
    print("\n--- Training CNN ---")
    cnn = build_cnn(input_shape)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    mc_cnn = ModelCheckpoint('cnn_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    # Validation uses X_val, NOT X_test
    cnn_hist = cnn.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=100, 
        batch_size=32, 
        callbacks=[es, mc_cnn]
    )
    
    # Training CNN+LSTM
    print("\n--- Training CNN+LSTM ---")
    cnn_lstm = build_cnn_lstm(input_shape)
    mc_lstm = ModelCheckpoint('cnn_lstm_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    cnn_lstm_hist = cnn_lstm.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=100, 
        batch_size=32, 
        callbacks=[es, mc_lstm]
    )
    
    print("\nTraining complete. Models saved.")

if __name__ == "__main__":
    train_and_save()
