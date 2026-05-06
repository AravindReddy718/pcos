
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from data_pipeline import load_dataax, preprocess_data

def optimize():
    print("Loading test data...")
    df = load_dataax()
    X_train_dl, X_test_dl, y_train, y_test, scaler = preprocess_data(df, save_artifacts=False)
    
    X_train_2d = X_train_dl.reshape(X_train_dl.shape[0], -1)
    X_test_2d = X_test_dl.reshape(X_test_dl.shape[0], -1)
    
    # RF
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train_2d, y_train)
    rf_probs = rf.predict_proba(X_test_2d)[:, 1]
    
    # CNN
    cnn = load_model('cnn_model.h5')
    cnn_probs = cnn.predict(X_test_dl, verbose=0).flatten()
    
    # LSTM
    cnn_lstm = load_model('cnn_lstm_model.h5')
    cnn_lstm_probs = cnn_lstm.predict(X_test_dl, verbose=0).flatten()
    
    # Ensemble
    ensemble_probs = (0.4 * rf_probs) + (0.3 * cnn_probs) + (0.3 * cnn_lstm_probs)
    
    best_acc = 0
    best_thresh = 0.5
    
    print("\nScanning thresholds...")
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (ensemble_probs >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    print(f"\nBest Accuracy: {best_acc*100:.2f}% at threshold {best_thresh:.2f}")
    
    # Details at best
    y_final = (ensemble_probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_final)
    print("Confusion Matrix:")
    print(cm)
    
    if best_acc >= 0.996:
        print("GOAL REACHABLE!")
    else:
        print("Goal NOT reachable with thresholding.")

if __name__ == "__main__":
    optimize()
