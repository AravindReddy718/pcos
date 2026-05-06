
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import joblib
import os

def load_dataax(filepath="Dataset/clean_data.csv"):
    if not os.path.exists(filepath):
        # Fallback to local path if running from root
        if os.path.exists(f"Dataset/{os.path.basename(filepath)}"):
             filepath = f"Dataset/{os.path.basename(filepath)}"
        else:
             raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    
    # Basic cleaning
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
        
    return df


def preprocess_data(df, save_artifacts=False):
    # STRICT SEEDING
    np.random.seed(0)
    
    target = "PCOS (Y/N)"
    
    # Notebook specific feature selection (matching verified baseline)
    selected_features = [
        "Follicle No. (R)", 
        "Follicle No. (L)", 
        "Skin darkening (Y/N)", 
        "hair growth(Y/N)", 
        "Weight gain(Y/N)", 
        "Cycle(R/I)"
    ]
    
    if target not in df.columns:
         raise ValueError(f"Target '{target}' not found")
         
    # Ensure we use the exact features found in the high-perf notebook
    cols_to_use = selected_features + [target]
    df_subset = df[cols_to_use].copy()
    
    X = df_subset.drop(columns=[target])
    y = df_subset[target]
    
    # SPLIT FIRST (To prevent data leakage)
    # We no longer use locked indices if they were based on resampled data.
    # We create a fresh, clean split of the raw data.
    print("📢 Splitting data BEFORE resampling to ensure realistic evaluation...")
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # SMOTEENN for balancing (Applied ONLY to the training set)
    # Enforce random_state=0 strictly
    print("📢 Applying SMOTEENN to the training set only...")
    resample = SMOTEENN(sampling_strategy=1/1, random_state=0)
    X_train, y_train = resample.fit_resample(X_train_raw, y_train_raw)
    
    # Scale (StandardScaler, fit ONLY on X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # STRICT DATA LEAKAGE PREVENTION: Transform test using train scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for DL: (samples, features, 1)
    X_train_dl = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_dl = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    
    if save_artifacts:
        joblib.dump(scaler, 'scaler.pkl')
        print("Scaler saved to scaler.pkl")
    
    return X_train_dl, X_test_dl, y_train, y_test, scaler

if __name__ == "__main__":
    try:
        df = load_dataax()
        # Clean run to force lock if needed
        X_train, X_test, y_train, y_test, _ = preprocess_data(df, save_artifacts=True)
        print(f"Preprocessing complete.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
