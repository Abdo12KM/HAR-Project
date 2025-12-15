"""
Human Activity Recognition (HAR) Using Smartphone Sensors - Phase 2
=====================================================================
Step 3: Preprocessing (loading, reshaping, scaling, label encoding for raw data)
Step 4: Model Training (LSTM) with evaluation and loss curve visualization
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import warnings

warnings.filterwarnings("ignore")

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR = "UCI HAR Dataset"
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 64

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Raw inertial signal files
SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

print("\n" + "=" * 60)
print("HUMAN ACTIVITY RECOGNITION - PHASE 2")
print("Raw Inertial Data: Preprocessing & LSTM Model")
print("=" * 60)

# =============================================================================
# DOWNLOAD DATASET
# =============================================================================
if not os.path.exists(DATASET_DIR):
    print("\n[INFO] Downloading UCI HAR Dataset...")
    zip_path = "UCI_HAR_Dataset.zip"
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    print("[INFO] Dataset downloaded and extracted successfully!")
else:
    print(f"\n[INFO] Dataset already exists at '{DATASET_DIR}'")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_signal_file(filepath):
    """Load a single signal file and return as numpy array."""
    return pd.read_csv(filepath, sep=r"\s+", header=None).values


def load_inertial_signals(data_type):
    """
    Load all 9 inertial signal files for train or test.
    Returns: numpy array of shape (samples, 128 time_steps, 9 channels)
    """
    signals = []
    signal_dir = os.path.join(DATASET_DIR, data_type, "Inertial Signals")

    for signal_name in SIGNAL_FILES:
        filename = f"{signal_name}_{data_type}.txt"
        filepath = os.path.join(signal_dir, filename)
        signal_data = load_signal_file(filepath)
        signals.append(signal_data)

    # Stack: (9, samples, 128) -> transpose to (samples, 128, 9)
    signals = np.array(signals)
    signals = np.transpose(signals, (1, 2, 0))
    return signals


def load_labels(data_type):
    """Load activity labels."""
    filepath = os.path.join(DATASET_DIR, data_type, f"y_{data_type}.txt")
    return pd.read_csv(filepath, header=None).values.ravel()


# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: PREPROCESSING")
print("=" * 60)

# --- Data Loading ---
print("\n[DATA LOADING]")
print("  Loading raw inertial signals...")
X_train = load_inertial_signals("train")
X_test = load_inertial_signals("test")
y_train = load_labels("train")
y_test = load_labels("test")

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Time steps: {X_train.shape[1]}")
print(f"  Channels: {X_train.shape[2]}")
print(f"  Input shape: {X_train.shape}")

# --- Reshaping for Scaling ---
print("\n[RESHAPING FOR SCALING]")
n_train_samples, n_timesteps, n_channels = X_train.shape
n_test_samples = X_test.shape[0]

# Reshape to 2D for scaling: (samples * timesteps, channels)
X_train_2d = X_train.reshape(-1, n_channels)
X_test_2d = X_test.reshape(-1, n_channels)
print(f"  Reshaped for scaling: {X_train_2d.shape}")

# --- Feature Scaling ---
print("\n[FEATURE SCALING]")
scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_test_scaled_2d = scaler.transform(X_test_2d)

# Reshape back to 3D: (samples, timesteps, channels)
X_train_scaled = X_train_scaled_2d.reshape(n_train_samples, n_timesteps, n_channels)
X_test_scaled = X_test_scaled_2d.reshape(n_test_samples, n_timesteps, n_channels)

print(f"  Applied StandardScaler per channel")
print(f"  Final shape: {X_train_scaled.shape}")
print(f"  Train mean: {X_train_scaled.mean():.6f}, Train std: {X_train_scaled.std():.6f}")

# --- Label Encoding ---
print("\n[LABEL ENCODING]")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Convert to one-hot encoding for categorical cross-entropy
n_classes = len(encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes=n_classes)
y_test_onehot = to_categorical(y_test_encoded, num_classes=n_classes)

print(f"  Classes: {encoder.classes_}")
print(f"  One-hot shape: {y_train_onehot.shape}")
print(f"  Label distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(
        f"    {ACTIVITY_LABELS[label]}: {count} samples ({count / len(y_train) * 100:.1f}%)"
    )

# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING (LSTM)")
print("=" * 60)

# --- Build LSTM Model ---
print("\n[LSTM MODEL ARCHITECTURE]")
model = Sequential(
    [
        LSTM(100, return_sequences=True, input_shape=(n_timesteps, n_channels)),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(n_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

# --- Train Model ---
print("\n[TRAINING]")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Validation split: {VALIDATION_SPLIT * 100:.0f}%")

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

history = model.fit(
    X_train_scaled,
    y_train_onehot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=[early_stopping],
    verbose=1,
)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# --- Test Evaluation ---
print("\n[TEST METRICS]")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_onehot, verbose=0)

# Get predictions for precision
y_test_pred = model.predict(X_test_scaled, verbose=0)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
test_precision = precision_score(y_test_encoded, y_test_pred_classes, average="macro")

print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Precision (macro): {test_precision:.4f}")

# --- Training Metrics (final epoch) ---
print("\n[TRAINING METRICS (Final Epoch)]")
final_train_acc = history.history["accuracy"][-1]
final_train_loss = history.history["loss"][-1]
final_val_acc = history.history["val_accuracy"][-1]
final_val_loss = history.history["val_loss"][-1]

print(f"  {'Metric':<20} {'Train':<15} {'Validation':<15}")
print(f"  {'-' * 50}")
print(f"  {'Accuracy':<20} {final_train_acc:<15.4f} {final_val_acc:<15.4f}")
print(f"  {'Loss':<20} {final_train_loss:<15.4f} {final_val_loss:<15.4f}")

# =============================================================================
# LOSS CURVE VISUALIZATION
# =============================================================================
print("\n[GENERATING LOSS CURVES]")

# --- Loss Curve ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], "b-", label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], "r-", label="Validation Loss", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("LSTM - Training/Validation Loss Curve", fontsize=14)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

# Analyze fit status based on loss curves
train_loss_final = history.history["loss"][-1]
val_loss_final = history.history["val_loss"][-1]
loss_gap = val_loss_final - train_loss_final

if loss_gap > 0.3:
    fit_status = "Overfitting"
elif train_loss_final > 0.5 and val_loss_final > 0.5:
    fit_status = "Underfitting"
else:
    fit_status = "Good Fit"

plt.text(
    0.02,
    0.98,
    f"Status: {fit_status}\nTrain-Val Gap: {loss_gap:.3f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# --- Accuracy Curve ---
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
plt.plot(
    history.history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2
)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("LSTM - Training/Validation Accuracy Curve", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_complexity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Loss/Accuracy curves saved: lstm_complexity_curves.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)

print(f"\n[PREPROCESSING SUMMARY]")
print(f"  Raw signal channels: {n_channels}")
print(f"  Time steps per sample: {n_timesteps}")
print(f"  Total input features: {n_timesteps * n_channels}")
print(f"  Scaling: StandardScaler (mean=0, std=1)")

print(f"\n[MODEL SUMMARY]")
print(f"  Architecture: LSTM(100) → Dropout(0.3) → LSTM(50) → Dropout(0.3) → Dense(6)")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Epochs trained: {len(history.history['loss'])}")

print(f"\n[PERFORMANCE METRICS]")
print(
    f"\n  {'Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}"
)
print(f"  {'-' * 65}")
print(
    f"  {'Accuracy':<20} {final_train_acc:<15.4f} {final_val_acc:<15.4f} {test_accuracy:<15.4f}"
)
print(
    f"  {'Loss':<20} {final_train_loss:<15.4f} {final_val_loss:<15.4f} {test_loss:<15.4f}"
)
print(f"  {'Precision (macro)':<20} {'-':<15} {'-':<15} {test_precision:<15.4f}")
print(f"\n  Fit Status: {fit_status}")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print("\nOutput files generated:")
print("  - lstm_complexity_curves.png (Training/Validation loss and accuracy curves)")
