"""
Human Activity Recognition (HAR) - Model Comparison & Visualization
=====================================================================
Loads pre-trained models from Phase 1 and Phase 2, evaluates them,
and generates comprehensive comparison charts.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, log_loss
import warnings

warnings.filterwarnings("ignore")

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR = "UCI HAR Dataset"
MODELS_DIR = "models"
RANDOM_STATE = 42

# Ensure models directory exists
if not os.path.exists(MODELS_DIR):
    raise FileNotFoundError(f"Directory '{MODELS_DIR}' not found. Please run phase1_har.py and phase2_har.py first.")

ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

SIGNAL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

print("\n" + "=" * 60)
print("HAR MODEL COMPARISON & VISUALIZATION (LOAD ONLY)")
print("=" * 60)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_signal_file(filepath):
    return pd.read_csv(filepath, sep=r"\s+", header=None).values

def load_inertial_signals(data_type):
    signals = []
    signal_dir = os.path.join(DATASET_DIR, data_type, "Inertial Signals")
    for signal_name in SIGNAL_FILES:
        filename = f"{signal_name}_{data_type}.txt"
        filepath = os.path.join(signal_dir, filename)
        signal_data = load_signal_file(filepath)
        signals.append(signal_data)
    signals = np.array(signals)
    signals = np.transpose(signals, (1, 2, 0))
    return signals

def determine_fit_status(train_loss, test_loss, train_acc, test_acc):
    gap = test_loss - train_loss
    if gap > 0.2 or (train_acc - test_acc) > 0.1:
        return "Overfitting"
    elif train_acc < 0.75:
        return "Underfitting"
    else:
        return "Good Fit"

# =============================================================================
# LOAD DATASET
# =============================================================================
print("\n[LOADING DATASET]")
if not os.path.exists(DATASET_DIR):
    print("  Downloading UCI HAR Dataset...")
    zip_path = "UCI_HAR_Dataset.zip"
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)

# Load Features (for Classical Models)
print("  Loading pre-extracted features...")
X_train_feat = pd.read_csv(os.path.join(DATASET_DIR, "train", "X_train.txt"), sep=r'\s+', header=None).values
y_train = pd.read_csv(os.path.join(DATASET_DIR, "train", "y_train.txt"), header=None).values.ravel()
X_test_feat = pd.read_csv(os.path.join(DATASET_DIR, "test", "X_test.txt"), sep=r'\s+', header=None).values
y_test = pd.read_csv(os.path.join(DATASET_DIR, "test", "y_test.txt"), header=None).values.ravel()

# Load Raw Signals (for LSTM)
print("  Loading raw inertial signals...")
X_train_raw = load_inertial_signals("train")
X_test_raw = load_inertial_signals("test")

# =============================================================================
# LOAD MODELS & PREPROCESSORS
# =============================================================================
print("\n[LOADING MODELS & PREPROCESSORS]")

try:
    # Classical Preprocessors
    print("  Loading Feature Scaler, PCA, and Encoder...")
    scaler_feat = joblib.load(os.path.join(MODELS_DIR, "scaler_features.pkl"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # Raw Data Preprocessors
    print("  Loading Raw Signal Scaler...")
    scaler_raw = joblib.load(os.path.join(MODELS_DIR, "scaler_raw.pkl"))

    # Models
    print("  Loading Logistic Regression...")
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    
    print("  Loading Random Forest...")
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    
    print("  Loading LSTM...")
    lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    
    print("  All artifacts loaded successfully!")

except FileNotFoundError as e:
    print(f"\n[ERROR] Missing artifact: {e}")
    print("Please ensure you have run 'phase1_har.py' and 'phase2_har.py' to generate these files.")
    exit(1)

# =============================================================================
# PREPARE DATA
# =============================================================================
print("\n[PREPARING EVALUATION DATA]")

# 1. Classical Models (Features -> Scaler -> PCA)
X_train_scaled = scaler_feat.transform(X_train_feat)
X_test_scaled = scaler_feat.transform(X_test_feat)

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

y_train_enc = encoder.transform(y_train)
y_test_enc = encoder.transform(y_test)

# 2. LSTM (Raw -> Reshape -> Scaler -> Reshape)
n_train, n_timesteps, n_channels = X_train_raw.shape
n_test = X_test_raw.shape[0]

X_train_raw_2d = X_train_raw.reshape(-1, n_channels)
X_test_raw_2d = X_test_raw.reshape(-1, n_channels)

X_train_raw_scaled = scaler_raw.transform(X_train_raw_2d).reshape(n_train, n_timesteps, n_channels)
X_test_raw_scaled = scaler_raw.transform(X_test_raw_2d).reshape(n_test, n_timesteps, n_channels)

y_train_onehot = to_categorical(y_train_enc, num_classes=6)
y_test_onehot = to_categorical(y_test_enc, num_classes=6)

# =============================================================================
# EVALUATE MODELS (No Training)
# =============================================================================
print("\n[EVALUATING MODELS]")

def evaluate_classical(model, X_tr, y_tr, X_te, y_te, name):
    print(f"  Evaluating {name}...")
    # Train metrics for fit status
    train_pred = model.predict(X_tr)
    train_proba = model.predict_proba(X_tr)
    train_acc = accuracy_score(y_tr, train_pred)
    train_loss = log_loss(y_tr, train_proba)
    
    # Test metrics
    test_pred = model.predict(X_te)
    test_proba = model.predict_proba(X_te)
    test_acc = accuracy_score(y_te, test_pred)
    test_loss = log_loss(y_te, test_proba)
    test_prec = precision_score(y_te, test_pred, average="macro")
    
    return {
        "Accuracy": test_acc,
        "Precision": test_prec,
        "Loss": test_loss,
        "Train_Loss": train_loss,
        "Train_Acc": train_acc
    }

def evaluate_lstm(model, X_tr, y_tr, y_tr_lbl, X_te, y_te, y_te_lbl, name):
    print(f"  Evaluating {name}...")
    # Train metrics
    train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0)
    
    # Test metrics
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    test_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    test_prec = precision_score(y_te_lbl, test_pred, average="macro")
    
    return {
        "Accuracy": test_acc,
        "Precision": test_prec,
        "Loss": test_loss,
        "Train_Loss": train_loss,
        "Train_Acc": train_acc
    }

# Run Evaluations
lr_metrics = evaluate_classical(lr_model, X_train_pca, y_train_enc, X_test_pca, y_test_enc, "Logistic Regression")
rf_metrics = evaluate_classical(rf_model, X_train_pca, y_train_enc, X_test_pca, y_test_enc, "Random Forest")
lstm_metrics = evaluate_lstm(lstm_model, X_train_raw_scaled, y_train_onehot, y_train_enc, X_test_raw_scaled, y_test_onehot, y_test_enc, "LSTM")

# Calculate Fit Status
lr_status = determine_fit_status(lr_metrics["Train_Loss"], lr_metrics["Loss"], lr_metrics["Train_Acc"], lr_metrics["Accuracy"])
rf_status = determine_fit_status(rf_metrics["Train_Loss"], rf_metrics["Loss"], rf_metrics["Train_Acc"], rf_metrics["Accuracy"])
lstm_status = determine_fit_status(lstm_metrics["Train_Loss"], lstm_metrics["Loss"], lstm_metrics["Train_Acc"], lstm_metrics["Accuracy"])

# Compile Results
results = {
    "Model": ["Logistic Regression", "Random Forest", "LSTM"],
    "Data Type": ["Pre-extracted (PCA)", "Pre-extracted (PCA)", "Raw Inertial"],
    "Features": [pca.n_components_, pca.n_components_, n_timesteps * n_channels],
    "Accuracy": [lr_metrics["Accuracy"], rf_metrics["Accuracy"], lstm_metrics["Accuracy"]],
    "Precision": [lr_metrics["Precision"], rf_metrics["Precision"], lstm_metrics["Precision"]],
    "Loss": [lr_metrics["Loss"], rf_metrics["Loss"], lstm_metrics["Loss"]],
    "Fit Status": [lr_status, rf_status, lstm_status],
    "Parameters": [
        pca.n_components_ * 6,  # Approx LR params
        100 * 20,  # Approx RF nodes (est)
        lstm_model.count_params()
    ]
}

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# =============================================================================
# GENERATE COMPARISON VISUALIZATIONS
# =============================================================================
print("\n[GENERATING COMPARISON CHARTS]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#3498db', '#2ecc71', '#e74c3c']
model_names = ['Logistic\nRegression', 'Random\nForest', 'LSTM']

# 1. Accuracy
ax1 = axes[0, 0]
vals = results["Accuracy"]
bars = ax1.bar(model_names, vals, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0.8, 1.0)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.2%}', ha='center', fontsize=11)

# 2. Precision
ax2 = axes[0, 1]
vals = results["Precision"]
bars = ax2.bar(model_names, vals, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_title('Test Precision Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(0.8, 1.0)
for bar, v in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.2%}', ha='center', fontsize=11)

# 3. Loss
ax3 = axes[1, 0]
vals = results["Loss"]
bars = ax3.bar(model_names, vals, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_title('Test Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
for bar, v in zip(bars, vals):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)

# 4. Parameters
ax4 = axes[1, 1]
vals = results["Parameters"]
bars = ax4.bar(model_names, vals, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
ax4.set_yscale('log')
for bar, v in zip(bars, vals):
    ax4.text(bar.get_x() + bar.get_width()/2, v * 1.2, f'{v:,}', ha='center', fontsize=10)

plt.suptitle('HAR Model Comparison (Standardized)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("  Saved: model_comparison.png")

print("\n" + "=" * 80)
print("KEY FINDINGS (AUTO-GENERATED FROM LOADED MODELS)")
print("=" * 80)
best_idx = np.argmax(results["Accuracy"])
best_model_name = results["Model"][best_idx]
print(f"1. BEST MODEL: {best_model_name} ({results['Accuracy'][best_idx]:.2%} accuracy)")
print(f"2. DEEP LEARNING VS ML: LSTM Accuracy is {lstm_metrics['Accuracy']:.2%}")
print(f"3. FIT STATUS: LSTM is '{lstm_status}'")
print("\n[COMPLETE] Analysis finished without retraining.")
