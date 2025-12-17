"""
Human Activity Recognition (HAR) Using Smartphone Sensors - Phase 2
=====================================================================
Step 3: Preprocessing (loading, reshaping, scaling, label encoding for raw data)
Step 4: Model Training (LSTM and Bidirectional LSTM) with comparison
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score
import warnings
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR = "UCI HAR Dataset"
MODELS_DIR = "models"
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
print("Raw Inertial Data: LSTM vs Bidirectional LSTM Comparison")
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

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)


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


def build_lstm_model(n_timesteps, n_channels, n_classes, bidirectional=False):
    """Build LSTM or Bidirectional LSTM model."""
    model = Sequential()

    if bidirectional:
        model.add(
            Bidirectional(
                LSTM(100, return_sequences=True), input_shape=(n_timesteps, n_channels)
            )
        )
        model.add(Dropout(0.4))
        model.add(Bidirectional(LSTM(50, return_sequences=False)))
        model.add(Dropout(0.4))
    else:
        model.add(
            LSTM(100, return_sequences=True, input_shape=(n_timesteps, n_channels))
        )
        model.add(Dropout(0.3))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.3))

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_and_evaluate(
    model, X_train, y_train, X_test, y_test, y_test_encoded, model_name
):
    """Train model and return metrics."""
    print(f"\n[TRAINING {model_name.upper()}]")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_precision = precision_score(y_test_encoded, y_pred_classes, average="macro")

    # Find best epoch (minimum validation loss - same as early stopping)
    best_epoch = np.argmin(history.history["val_loss"])
    
    return {
        "history": history,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "test_precision": test_precision,
        "best_epoch": best_epoch + 1,  # 1-indexed for display
        "best_train_acc": history.history["accuracy"][best_epoch],
        "best_val_acc": history.history["val_accuracy"][best_epoch],
        "best_train_loss": history.history["loss"][best_epoch],
        "best_val_loss": history.history["val_loss"][best_epoch],
    }


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

# Save scaler for Streamlit app
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_raw.pkl"))

# Reshape back to 3D: (samples, timesteps, channels)
X_train_scaled = X_train_scaled_2d.reshape(n_train_samples, n_timesteps, n_channels)
X_test_scaled = X_test_scaled_2d.reshape(n_test_samples, n_timesteps, n_channels)

print(f"  Applied StandardScaler per channel")
print(f"  Final shape: {X_train_scaled.shape}")
print(
    f"  Train mean: {X_train_scaled.mean():.6f}, Train std: {X_train_scaled.std():.6f}"
)

# --- Label Encoding ---
print("\n[LABEL ENCODING]")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Save encoder
joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

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
# STEP 4: MODEL TRAINING - LSTM vs Bidirectional LSTM
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING (LSTM vs Bidirectional LSTM)")
print("=" * 60)

# --- Build and Train Standard LSTM ---
print("\n" + "-" * 40)
print("MODEL 1: Standard LSTM")
print("-" * 40)
lstm_model = build_lstm_model(n_timesteps, n_channels, n_classes, bidirectional=False)
lstm_model.summary()
lstm_results = train_and_evaluate(
    lstm_model,
    X_train_scaled,
    y_train_onehot,
    X_test_scaled,
    y_test_onehot,
    y_test_encoded,
    "LSTM",
)

# Save LSTM model
lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
print(f"  Model saved: {os.path.join(MODELS_DIR, 'lstm_model.keras')}")

# --- Build and Train Bidirectional LSTM ---
print("\n" + "-" * 40)
print("MODEL 2: Bidirectional LSTM")
print("-" * 40)
bilstm_model = build_lstm_model(n_timesteps, n_channels, n_classes, bidirectional=True)
bilstm_model.summary()
bilstm_results = train_and_evaluate(
    bilstm_model,
    X_train_scaled,
    y_train_onehot,
    X_test_scaled,
    y_test_onehot,
    y_test_encoded,
    "Bidirectional LSTM",
)

# Save Bidirectional LSTM model
bilstm_model.save(os.path.join(MODELS_DIR, "bilstm_model.keras"))
print(f"  Model saved: {os.path.join(MODELS_DIR, 'bilstm_model.keras')}")

# =============================================================================
# VISUALIZATION - COMPARISON
# =============================================================================
print("\n[GENERATING COMPARISON PLOTS]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LSTM Loss Curve
axes[0, 0].plot(
    lstm_results["history"].history["loss"], "b-", label="Training Loss", linewidth=2
)
axes[0, 0].plot(
    lstm_results["history"].history["val_loss"],
    "r-",
    label="Validation Loss",
    linewidth=2,
)
axes[0, 0].set_xlabel("Epoch", fontsize=12)
axes[0, 0].set_ylabel("Loss", fontsize=12)
axes[0, 0].set_title("Standard LSTM - Loss Curve", fontsize=14)
axes[0, 0].legend(loc="upper right")
axes[0, 0].grid(True, alpha=0.3)

# LSTM Accuracy Curve
axes[0, 1].plot(
    lstm_results["history"].history["accuracy"],
    "b-",
    label="Training Accuracy",
    linewidth=2,
)
axes[0, 1].plot(
    lstm_results["history"].history["val_accuracy"],
    "r-",
    label="Validation Accuracy",
    linewidth=2,
)
axes[0, 1].set_xlabel("Epoch", fontsize=12)
axes[0, 1].set_ylabel("Accuracy", fontsize=12)
axes[0, 1].set_title("Standard LSTM - Accuracy Curve", fontsize=14)
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# Bi-LSTM Loss Curve
axes[1, 0].plot(
    bilstm_results["history"].history["loss"], "b-", label="Training Loss", linewidth=2
)
axes[1, 0].plot(
    bilstm_results["history"].history["val_loss"],
    "r-",
    label="Validation Loss",
    linewidth=2,
)
axes[1, 0].set_xlabel("Epoch", fontsize=12)
axes[1, 0].set_ylabel("Loss", fontsize=12)
axes[1, 0].set_title("Bidirectional LSTM - Loss Curve", fontsize=14)
axes[1, 0].legend(loc="upper right")
axes[1, 0].grid(True, alpha=0.3)

# Bi-LSTM Accuracy Curve
axes[1, 1].plot(
    bilstm_results["history"].history["accuracy"],
    "b-",
    label="Training Accuracy",
    linewidth=2,
)
axes[1, 1].plot(
    bilstm_results["history"].history["val_accuracy"],
    "r-",
    label="Validation Accuracy",
    linewidth=2,
)
axes[1, 1].set_xlabel("Epoch", fontsize=12)
axes[1, 1].set_ylabel("Accuracy", fontsize=12)
axes[1, 1].set_title("Bidirectional LSTM - Accuracy Curve", fontsize=14)
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_comparison_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Comparison curves saved: lstm_comparison_curves.png")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
models = ["Standard LSTM", "Bidirectional LSTM"]
test_accuracies = [
    lstm_results["test_accuracy"] * 100,
    bilstm_results["test_accuracy"] * 100,
]
test_precisions = [
    lstm_results["test_precision"] * 100,
    bilstm_results["test_precision"] * 100,
]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(
    x - width / 2, test_accuracies, width, label="Test Accuracy", color="#3498db"
)
bars2 = ax.bar(
    x + width / 2, test_precisions, width, label="Test Precision", color="#2ecc71"
)

ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("LSTM vs Bidirectional LSTM - Performance Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend()
ax.set_ylim(80, 100)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )
for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig("lstm_bar_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Bar comparison saved: lstm_bar_comparison.png")

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

print(f"\n[MODEL COMPARISON]")
print(
    f"\n  {'Model':<25} {'Test Acc':<12} {'Test Prec':<12} {'Test Loss':<12} {'Epochs'}"
)
print(f"  {'-' * 75}")
print(
    f"  {'Standard LSTM':<25} {lstm_results['test_accuracy'] * 100:<12.2f} {lstm_results['test_precision'] * 100:<12.2f} {lstm_results['test_loss']:<12.4f} {len(lstm_results['history'].history['loss'])}"
)
print(
    f"  {'Bidirectional LSTM':<25} {bilstm_results['test_accuracy'] * 100:<12.2f} {bilstm_results['test_precision'] * 100:<12.2f} {bilstm_results['test_loss']:<12.4f} {len(bilstm_results['history'].history['loss'])}"
)

# Determine best model
best_model = (
    "Bidirectional LSTM"
    if bilstm_results["test_accuracy"] > lstm_results["test_accuracy"]
    else "Standard LSTM"
)
best_acc = max(lstm_results["test_accuracy"], bilstm_results["test_accuracy"])
print(f"\n  Best Model: {best_model} (Accuracy: {best_acc * 100:.2f}%)")

# Fit status analysis (using best epoch metrics - matches restored weights)
lstm_gap = lstm_results["best_val_loss"] - lstm_results["best_train_loss"]
bilstm_gap = bilstm_results["best_val_loss"] - bilstm_results["best_train_loss"]

lstm_fit = (
    "Overfitting"
    if lstm_gap > 0.3
    else ("Underfitting" if lstm_results["best_train_loss"] > 0.5 else "Good Fit")
)
bilstm_fit = (
    "Overfitting"
    if bilstm_gap > 0.3
    else ("Underfitting" if bilstm_results["best_train_loss"] > 0.5 else "Good Fit")
)

print(f"\n[FIT STATUS] (at best epoch)")
print(f"  Standard LSTM: {lstm_fit} (Train-Val Gap: {lstm_gap:.3f}, Best Epoch: {lstm_results['best_epoch']})")
print(f"  Bidirectional LSTM: {bilstm_fit} (Train-Val Gap: {bilstm_gap:.3f}, Best Epoch: {bilstm_results['best_epoch']})")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print("\nOutput files generated:")
print("  - lstm_comparison_curves.png (Training/Validation curves for both models)")
print("  - lstm_bar_comparison.png (Side-by-side performance comparison)")
print(f"\nModels saved to '{MODELS_DIR}/':")
print("  - lstm_model.keras (Standard LSTM)")
print("  - bilstm_model.keras (Bidirectional LSTM)")
print("  - scaler_raw.pkl (StandardScaler for raw data)")
print("  - label_encoder.pkl (LabelEncoder)")
