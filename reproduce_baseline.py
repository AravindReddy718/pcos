
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import warnings
import sys

warnings.filterwarnings('ignore')

# EXPECTED BASELINE (Random Forest as found in exploration)
EXPECTED_RF_ACCURACY = 99.11 
TOLERANCE = 0.1

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
    return df

def run_experiment(df):
    target = "PCOS (Y/N)"
    selected_features = [
        "Follicle No. (R)", "Follicle No. (L)", "Skin darkening (Y/N)", 
        "hair growth(Y/N)", "Weight gain(Y/N)", "Cycle(R/I)"
    ]
    
    cols_to_use = selected_features + [target]
    temp_df = df[cols_to_use].copy()
        
    X = temp_df.drop([target], axis=1)
    y = temp_df[[target]]
    
    resample = SMOTEENN(sampling_strategy=1/1, random_state=0)
    X_res, y_res = resample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0, stratify=y_res)
    
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest (The likely "Optimized" model)
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf) * 100
    
    print(f"Random Forest Accuracy: {acc_rf:.2f}%")
    
    # Strict Check
    diff = abs(acc_rf - EXPECTED_RF_ACCURACY)
    if diff > TOLERANCE:
        print(f"CRITICAL ERROR: Baseline accuracy deviation! Expected {EXPECTED_RF_ACCURACY}%, got {acc_rf:.2f}%")
        print("Deviation exceeds tolerance. STOPPING EXECUTION.")
        sys.exit(1)
    else:
        print("Baseline reproduced successfully within tolerance.")

def main():
    filepath = "Dataset/clean_data.csv"
    try:
        df = load_and_clean_data(filepath)
        run_experiment(df)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
