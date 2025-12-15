"""
Human Activity Recognition (HAR) Using Smartphone Sensors - Phase 1 (IMPROVED)
=====================================================================
Step 1: Preprocessing (loading, scaling, label encoding, PCA)
Step 2: Model Training (Logistic Regression, Random Forest) with HYPERPARAMETER TUNING
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.model_selection import learning_curve

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_DIR = "UCI HAR Dataset"
PCA_VARIANCE_THRESHOLD = 0.95
RANDOM_STATE = 42

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

print("\n" + "=" * 60)
print("HUMAN ACTIVITY RECOGNITION - PHASE 1 (IMPROVED)")
print("Pre-extracted Features: Tuned Classical ML Models")
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
# STEP 1: PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 1: PREPROCESSING")
print("=" * 60)

# --- Data Loading ---
print("\n[DATA LOADING]")
X_train = pd.read_csv(
    os.path.join(DATASET_DIR, "train", "X_train.txt"),
    sep=r'\s+',
    header=None,
).values
y_train = pd.read_csv(
    os.path.join(DATASET_DIR, "train", "y_train.txt"), header=None
).values.ravel()
X_test = pd.read_csv(
    os.path.join(DATASET_DIR, "test", "X_test.txt"), sep=r'\s+', header=None
).values
y_test = pd.read_csv(
    os.path.join(DATASET_DIR, "test", "y_test.txt"), header=None
).values.ravel()

original_features = X_train.shape[1]
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Original features: {original_features}")

# --- Feature Scaling ---
print("\n[FEATURE SCALING]")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Applied StandardScaler (mean=0, std=1)")
print(
    f"  Train mean: {X_train_scaled.mean():.6f}, Train std: {X_train_scaled.std():.6f}"
)

# --- Label Encoding ---
print("\n[LABEL ENCODING]")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
print(f"  Classes: {encoder.classes_}")
print(f"  Label distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(
        f"    {ACTIVITY_LABELS[label]}: {count} samples ({count / len(y_train) * 100:.1f}%)"
    )

# --- PCA Dimensionality Reduction ---
print("\n[PCA DIMENSIONALITY REDUCTION]")
pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"  Variance threshold: {PCA_VARIANCE_THRESHOLD * 100:.0f}%")
print(f"  Original features: {original_features}")
print(f"  Reduced features: {pca.n_components_}")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
print(
    f"  Dimensionality reduction: {(1 - pca.n_components_ / original_features) * 100:.1f}%"
)

# =============================================================================
# STEP 2: MODEL TRAINING (TUNED)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: MODEL TRAINING (IMPROVED)")
print("=" * 60)

# --- Logistic Regression ---
print("\n[LOGISTIC REGRESSION - TUNED]")
print("  Training model with C=0.1 for regularization...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
    n_jobs=-1,
    C=0.1,  # Stronger regularization (Default=1.0)
)
lr_model.fit(X_train_pca, y_train_encoded)

# Evaluate LR
y_train_pred_lr = lr_model.predict(X_train_pca)
y_test_pred_lr = lr_model.predict(X_test_pca)
y_train_proba_lr = lr_model.predict_proba(X_train_pca)
y_test_proba_lr = lr_model.predict_proba(X_test_pca)

lr_train_acc = accuracy_score(y_train_encoded, y_train_pred_lr)
lr_test_acc = accuracy_score(y_test_encoded, y_test_pred_lr)
lr_train_loss = log_loss(y_train_encoded, y_train_proba_lr)
lr_test_loss = log_loss(y_test_encoded, y_test_proba_lr)
lr_train_prec = precision_score(y_train_encoded, y_train_pred_lr, average="macro")
lr_test_prec = precision_score(y_test_encoded, y_test_pred_lr, average="macro")

print(f"\n  {'Metric':<20} {'Train':<15} {'Test':<15}")
print(f"  {'-' * 50}")
print(f"  {'Accuracy':<20} {lr_train_acc:<15.4f} {lr_test_acc:<15.4f}")
print(f"  {'Log Loss':<20} {lr_train_loss:<15.4f} {lr_test_loss:<15.4f}")
print(f"  {'Precision (macro)':<20} {lr_train_prec:<15.4f} {lr_test_prec:<15.4f}")

# Learning curve for LR
print("  Generating learning curve...")
train_sizes_lr, train_scores_lr, val_scores_lr = learning_curve(
    lr_model,
    X_train_pca,
    y_train_encoded,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
train_mean_lr = train_scores_lr.mean(axis=1)
val_mean_lr = val_scores_lr.mean(axis=1)
train_std_lr = train_scores_lr.std(axis=1)
val_std_lr = val_scores_lr.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(
    train_sizes_lr,
    train_mean_lr - train_std_lr,
    train_mean_lr + train_std_lr,
    alpha=0.1,
    color="blue",
)
plt.plot(train_sizes_lr, train_mean_lr, "o-", color="blue", label="Training Score")
plt.fill_between(
    train_sizes_lr,
    val_mean_lr - val_std_lr,
    val_mean_lr + val_std_lr,
    alpha=0.1,
    color="orange",
)
plt.plot(train_sizes_lr, val_mean_lr, "o-", color="orange", label="Validation Score")
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Learning Curve - Logistic Regression (Improved)", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.ylim(0.7, 1.02)
gap_lr = train_mean_lr[-1] - val_mean_lr[-1]
lr_fit_status = (
    "Overfitting"
    if gap_lr > 0.05
    else ("Underfitting" if val_mean_lr[-1] < 0.85 else "Good Fit")
)
plt.text(
    0.02,
    0.02,
    f"Status: {lr_fit_status}\nTrain-Val Gap: {gap_lr:.3f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.tight_layout()
plt.savefig("lr_learning_curve_improved.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Learning curve saved: lr_learning_curve_improved.png")
print(f"  Fit Status: {lr_fit_status}")

# --- Random Forest ---
print("\n[RANDOM FOREST - TUNED]")
print("  Training model with max_depth=15, min_samples_leaf=5...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,  # Reduced from 20
    min_samples_leaf=5,  # Added to prevent overfitting on noise
    min_samples_split=10, # Added for stronger regularization
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_pca, y_train_encoded)

# Evaluate RF
y_train_pred_rf = rf_model.predict(X_train_pca)
y_test_pred_rf = rf_model.predict(X_test_pca)
y_train_proba_rf = rf_model.predict_proba(X_train_pca)
y_test_proba_rf = rf_model.predict_proba(X_test_pca)

rf_train_acc = accuracy_score(y_train_encoded, y_train_pred_rf)
rf_test_acc = accuracy_score(y_test_encoded, y_test_pred_rf)
rf_train_loss = log_loss(y_train_encoded, y_train_proba_rf)
rf_test_loss = log_loss(y_test_encoded, y_test_proba_rf)
rf_train_prec = precision_score(y_train_encoded, y_train_pred_rf, average="macro")
rf_test_prec = precision_score(y_test_encoded, y_test_pred_rf, average="macro")

print(f"\n  {'Metric':<20} {'Train':<15} {'Test':<15}")
print(f"  {'-' * 50}")
print(f"  {'Accuracy':<20} {rf_train_acc:<15.4f} {rf_test_acc:<15.4f}")
print(f"  {'Log Loss':<20} {rf_train_loss:<15.4f} {rf_test_loss:<15.4f}")
print(f"  {'Precision (macro)':<20} {rf_train_prec:<15.4f} {rf_test_prec:<15.4f}")

# Learning curve for RF
print("  Generating learning curve...")
train_sizes_rf, train_scores_rf, val_scores_rf = learning_curve(
    rf_model,
    X_train_pca,
    y_train_encoded,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
train_mean_rf = train_scores_rf.mean(axis=1)
val_mean_rf = val_scores_rf.mean(axis=1)
train_std_rf = train_scores_rf.std(axis=1)
val_std_rf = val_scores_rf.std(axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(
    train_sizes_rf,
    train_mean_rf - train_std_rf,
    train_mean_rf + train_std_rf,
    alpha=0.1,
    color="blue",
)
plt.plot(train_sizes_rf, train_mean_rf, "o-", color="blue", label="Training Score")
plt.fill_between(
    train_sizes_rf,
    val_mean_rf - val_std_rf,
    val_mean_rf + val_std_rf,
    alpha=0.1,
    color="orange",
)
plt.plot(train_sizes_rf, val_mean_rf, "o-", color="orange", label="Validation Score")
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Learning Curve - Random Forest (Improved)", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.ylim(0.7, 1.02)
gap_rf = train_mean_rf[-1] - val_mean_rf[-1]
rf_fit_status = (
    "Overfitting"
    if gap_rf > 0.05
    else ("Underfitting" if val_mean_rf[-1] < 0.85 else "Good Fit")
)
plt.text(
    0.02,
    0.02,
    f"Status: {rf_fit_status}\nTrain-Val Gap: {gap_rf:.3f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.tight_layout()
plt.savefig("rf_learning_curve_improved.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Learning curve saved: rf_learning_curve_improved.png")
print(f"  Fit Status: {rf_fit_status}")


# =============================================================================
# SAVE MODELS
# =============================================================================
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

import joblib

# Save Preprocessors
print(f"[INFO] Saving preprocessors to '{MODELS_DIR}/'...")
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_features.pkl"))
joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))
joblib.dump(encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Save Models
print(f"[INFO] Saving trained models to '{MODELS_DIR}/'...")
joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))

print("  - scaler_features.pkl")
print("  - pca.pkl")
print("  - label_encoder.pkl")
print("  - logistic_regression.pkl")
print("  - random_forest.pkl")
print("Models saved successfully!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1 (IMPROVED) SUMMARY")
print("=" * 60)

print(f"\n[PREPROCESSING SUMMARY]")
print(f"  Original features: {original_features}")
print(f"  PCA components: {pca.n_components_}")
print(f"  Reduction: {(1 - pca.n_components_ / original_features) * 100:.1f}%")

print(f"\n[MODEL COMPARISON - IMPROVED]")
print(
    f"\n  {'Model':<25} {'Test Accuracy':<15} {'Test Loss':<15} {'Test Precision':<15} {'Status'}"
)
print(f"  {'-' * 85}")
print(
    f"  {'Logistic Regression':<25} {lr_test_acc:<15.4f} {lr_test_loss:<15.4f} {lr_test_prec:<15.4f} {lr_fit_status}"
)
print(
    f"  {'Random Forest':<25} {rf_test_acc:<15.4f} {rf_test_loss:<15.4f} {rf_test_prec:<15.4f} {rf_fit_status}"
)

best_model = "Logistic Regression" if lr_test_acc > rf_test_acc else "Random Forest"
best_acc = max(lr_test_acc, rf_test_acc)
print(f"\n  Best Model: {best_model} (Accuracy: {best_acc:.4f})")

print("\n" + "=" * 60)
print("PHASE 1 (IMPROVED) COMPLETE")
print("=" * 60)
print("\nOutput files generated:")
print("  - lr_learning_curve_improved.png")
print("  - rf_learning_curve_improved.png")

