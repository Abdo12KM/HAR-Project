# Complete Machine Learning Code Explanation Guide

## Human Activity Recognition (HAR) Project - Line-by-Line Deep Dive

> **Purpose:** This document explains every line of code and ML concept in maximum detail for complete beginners with zero ML knowledge.

---

## Table of Contents

1. [Foundational ML Concepts](#1-foundational-ml-concepts)
2. [Phase 1: Classical Machine Learning (phase1_har.py)](#2-phase-1-classical-machine-learning)
3. [Phase 2: Deep Learning (phase2_har.py)](#3-phase-2-deep-learning)
4. [Model Comparison (model_comparison.py)](#4-model-comparison)
5. [Glossary of Terms](#5-glossary-of-terms)

---

# 1. Foundational ML Concepts

Before diving into code, let's understand the fundamental concepts.

## 1.1 What is Machine Learning?

**Machine Learning (ML)** is a method of teaching computers to make decisions or predictions based on data, without being explicitly programmed with rules.

### Analogy: Learning to Recognize Fruits
Imagine teaching a child to recognize apples:
- **Traditional Programming:** "If red/green, round, has stem → it's an apple"
- **Machine Learning:** Show the child 1000 photos of apples. They learn patterns themselves.

### The Three Types of ML:
| Type | Description | Example |
|------|-------------|---------|
| **Supervised Learning** | Learn from labeled examples | "This sensor data = WALKING" |
| **Unsupervised Learning** | Find patterns without labels | Group similar activities together |
| **Reinforcement Learning** | Learn by trial and reward | Robot learning to walk |

**This project uses Supervised Learning** - we have sensor data (input) with known activity labels (output).

---

## 1.2 Key ML Terminology

### Features (X)
**What it is:** The input data the model uses to make predictions.
- In our project: 561 measurements from smartphone sensors (acceleration, rotation, etc.)
- Think of features as "clues" the model uses to guess the activity

### Labels (y)
**What it is:** The correct answer we're trying to predict.
- In our project: WALKING, SITTING, STANDING, LAYING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
- These are what we want the model to learn to recognize

### Training Data vs Test Data
| Split | Purpose | Analogy |
|-------|---------|---------|
| **Training Data (70-80%)** | Teach the model | Studying from textbook |
| **Test Data (20-30%)** | Evaluate the model | Taking the final exam |

**Why split?** If we test on data the model has seen, it's like giving a student the exam answers beforehand - we won't know if they truly learned.

---

## 1.3 What is Overfitting and Underfitting?

### Overfitting (Memorization)
**Definition:** The model memorizes training data instead of learning general patterns.
- **Symptom:** Very high training accuracy, low test accuracy
- **Analogy:** A student who memorizes textbook pages word-for-word but can't answer rephrased questions

### Underfitting (Too Simple)
**Definition:** The model is too simple to capture the patterns in data.
- **Symptom:** Low accuracy on both training and test data
- **Analogy:** Trying to fit a straight line through curved data

### Good Fit (Goal)
**Definition:** Model learns general patterns that work on new data.
- **Symptom:** Similar accuracy on training and test data
- **Analogy:** Student who understands concepts and can apply them to new problems

---

## 1.4 Model Evaluation Metrics

### Accuracy
**Formula:** `Correct Predictions / Total Predictions × 100%`

**Example:** 
- 100 samples, model predicts 93 correctly
- Accuracy = 93/100 = 93%

### Log Loss (Cross-Entropy Loss)
**What it measures:** How confident the model is in its predictions.

**Key insight:** A model can be correct but with low confidence (bad) or correct with high confidence (good).

| Prediction | Actual | Correct? | Confidence | Log Loss |
|------------|--------|----------|------------|----------|
| WALKING (99%) | WALKING | ✓ | High | Low (Good) |
| WALKING (51%) | WALKING | ✓ | Low | High (Bad) |

**Lower log loss = Better model**

### Precision
**Formula:** `True Positives / (True Positives + False Positives)`

**In simple terms:** Of all the times the model said "WALKING", how many were actually walking?

---

## 1.5 What is Regularization?

**Problem:** Models can become too complex and memorize data (overfitting).

**Solution:** Regularization adds a penalty for complexity, forcing the model to stay simple.

### Types of Regularization:
| Technique | How it Works | Used In |
|-----------|--------------|---------|
| **L2 (Ridge)** | Penalizes large weights | Logistic Regression |
| **Dropout** | Randomly ignores neurons during training | LSTM |
| **Early Stopping** | Stop training when validation performance drops | Neural Networks |
| **Tree Depth Limits** | Restrict how deep trees can grow | Random Forest |

---

# 2. Phase 1: Classical Machine Learning

## File: `phase1_har.py`

### Section 1: Import Statements (Lines 1-19)

```python
"""
Human Activity Recognition (HAR) Using Smartphone Sensors - Phase 1 (IMPROVED)
=====================================================================
Step 1: Preprocessing (loading, scaling, label encoding, PCA)
Step 2: Model Training (Logistic Regression, Random Forest) with HYPERPARAMETER TUNING
"""
```
**Explanation:** This is a **docstring** - a multi-line comment describing what the file does. It's documentation for humans, not code the computer runs.

---

```python
import os
```
**What it does:** Imports the Operating System module.
**Why needed:** To work with file paths, check if folders exist, delete files.
**Example uses:** `os.path.exists("folder")`, `os.remove("file.txt")`

---

```python
import zipfile
```
**What it does:** Imports module for working with ZIP compressed files.
**Why needed:** The dataset downloads as a ZIP file that needs extraction.

---

```python
import urllib.request
```
**What it does:** Imports module for downloading files from the internet.
**Why needed:** To download the UCI HAR dataset from the web.

---

```python
import numpy as np
```
**What it does:** Imports NumPy (Numerical Python) with alias `np`.
**Why needed:** NumPy is the foundation of scientific computing in Python.

### NumPy Key Concepts:
| Concept | Description | Example |
|---------|-------------|---------|
| **ndarray** | N-dimensional array | `np.array([1, 2, 3])` |
| **Vectorization** | Operations on entire arrays at once | `arr * 2` multiplies all elements |
| **Broadcasting** | Automatic shape matching | Adding scalar to array |

**Why faster than Python lists?**
- Stored in contiguous memory (faster access)
- Written in C/Fortran (optimized)
- SIMD operations (single instruction, multiple data)

---

```python
import pandas as pd
```
**What it does:** Imports Pandas (Panel Data) with alias `pd`.
**Why needed:** For reading CSV/text files and data manipulation.

### Pandas Key Concepts:
| Object | Description | Analogy |
|--------|-------------|---------|
| **DataFrame** | 2D table with rows and columns | Excel spreadsheet |
| **Series** | 1D array with labels | Single column |
| **Index** | Row/column labels | Row numbers |

---

```python
import matplotlib.pyplot as plt
```
**What it does:** Imports Matplotlib's pyplot module.
**Why needed:** For creating visualizations (charts, graphs).

### How matplotlib works:
1. Create a figure (canvas)
2. Add axes (plotting area)
3. Plot data
4. Customize (labels, colors)
5. Save or display

---

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
```
**What it does:** Imports specific classes from scikit-learn's preprocessing module.

### StandardScaler Explained:
**Purpose:** Transform features to have mean=0 and standard deviation=1.

**Formula:** 
```
X_scaled = (X - μ) / σ
```
Where:
- X = original value
- μ (mu) = mean of all values
- σ (sigma) = standard deviation

**Why needed?**
- Features have different scales (e.g., acceleration: 0-20, gyroscope: 0-1)
- Without scaling, features with larger values dominate learning
- Many algorithms assume normalized input

**Example:**
| Original | Mean | Std | Scaled |
|----------|------|-----|--------|
| 100 | 50 | 25 | (100-50)/25 = 2.0 |
| 50 | 50 | 25 | (50-50)/25 = 0.0 |
| 25 | 50 | 25 | (25-50)/25 = -1.0 |

### LabelEncoder Explained:
**Purpose:** Convert text labels to numbers.

**Why needed?** 
- ML algorithms work with numbers, not text
- "WALKING" means nothing to a computer

**Example:**
| Text Label | Encoded |
|------------|---------|
| WALKING | 0 |
| SITTING | 1 |
| STANDING | 2 |

---

```python
from sklearn.decomposition import PCA
```
**What it does:** Imports Principal Component Analysis.

### PCA Deep Dive:

**Problem it solves:** We have 561 features - too many! 
- Harder to train models
- Many features are redundant
- Risk of overfitting
- Slow computation

**What PCA does:**
1. Finds directions of maximum variance in data
2. Projects data onto fewer dimensions
3. Keeps most information while reducing size

**Analogy:** 
Imagine a 3D cloud of points shaped like a pancake. A photo from above (2D) captures almost all the information because the pancake is flat. PCA finds that "best angle" automatically.

**Key Parameters:**
- `n_components=0.95` means "keep 95% of variance"
- Result: 561 features → 102 features (81.8% reduction!)

**Visualization:**
```
561 Original Features
        ↓ PCA finds patterns
102 Principal Components (95% of info kept)
        ↓ 
Faster training, less overfitting
```

---

```python
from sklearn.linear_model import LogisticRegression
```
**What it does:** Imports Logistic Regression classifier.

### Logistic Regression Explained:

**Despite the name, it's for CLASSIFICATION, not regression!**

**How it works:**
1. Calculate weighted sum of features: `z = w1*x1 + w2*x2 + ... + wn*xn + b`
2. Apply sigmoid function: `σ(z) = 1 / (1 + e^(-z))`
3. Output is probability between 0 and 1

**Sigmoid Function:**
```
     1 |         ___________
       |        /
  0.5  |-------/----------- (decision boundary)
       |      /
     0 |_____/
       -∞         0         +∞
              z value
```

**For multi-class (6 activities):**
Uses "One-vs-Rest" strategy:
- Train 6 binary classifiers
- Each asks: "Is this WALKING vs not-WALKING?"
- Pick class with highest probability

---

```python
from sklearn.ensemble import RandomForestClassifier
```
**What it does:** Imports Random Forest classifier.

### Random Forest Explained:

**Core idea:** Combine many weak models (decision trees) into one strong model.

**How Decision Trees work:**
```
                   Is acceleration > 5?
                   /                \
                 Yes                No
                 /                    \
        Is rotation > 2?          SITTING
        /           \
      Yes            No
      /              \
   WALKING      STANDING
```

**Why "Random" Forest?**
1. **Bootstrap sampling:** Each tree gets a random subset of training data
2. **Feature randomization:** Each split considers random subset of features
3. **Voting:** Final prediction = majority vote of all trees

**Why it works:**
- Individual trees may be wrong
- Errors of different trees usually don't overlap
- Combining them cancels out individual mistakes

**Hyperparameters explained:**
| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_estimators=100` | 100 trees | More trees = more stable (diminishing returns after ~100) |
| `max_depth=15` | Tree depth limit | Prevents trees from memorizing data |
| `min_samples_leaf=5` | Minimum 5 samples per leaf | Stops creating leaves for rare patterns |

---

```python
from sklearn.metrics import accuracy_score, precision_score, log_loss
```
**What it does:** Imports evaluation metric functions.

**accuracy_score:** Percentage of correct predictions
**precision_score:** Of predicted positives, how many are correct?
**log_loss:** Measures prediction confidence (lower = better)

---

```python
from sklearn.model_selection import learning_curve
```
**What it does:** Imports function to generate learning curves.

### Learning Curves Explained:

**Purpose:** Visualize how model performance changes with training size.

**What it generates:**
- Training score at different data sizes
- Validation score at different data sizes

**How to interpret:**
```
Accuracy
   1.0|    ___Training___
      |   /
   0.9|  /   ___Validation___
      | /   /
   0.8|/___/
      |_________________ Training Set Size
```

**Diagnosing problems:**
| Pattern | Diagnosis |
|---------|-----------|
| Large gap, training near 100% | Overfitting |
| Both curves low | Underfitting |
| Curves converge | Good fit |

---

### Section 2: Configuration (Lines 21-42)

```python
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
```
**Explanation:** Stores the download URL for the UCI HAR dataset. The `%20` is URL encoding for spaces.

---

```python
DATASET_DIR = "UCI HAR Dataset"
```
**Explanation:** Name of folder where dataset will be extracted.

---

```python
PCA_VARIANCE_THRESHOLD = 0.95
```
**Explanation:** Keep 95% of data variance when reducing dimensions.

**Why 95%?**
- Trade-off between dimensionality reduction and information loss
- 95% is standard industry choice
- Higher (99%) = more features, risk of overfitting
- Lower (90%) = fewer features, might lose important patterns

---

```python
RANDOM_STATE = 42
```
**Explanation:** Seed for random number generator.

**Why needed?**
- Many ML algorithms involve randomness
- Setting seed = reproducible results
- Same code = same results every time

**Why 42?**
- Famous number from "Hitchhiker's Guide to the Galaxy"
- The answer to "life, universe, and everything"
- Any number works, 42 is a tradition

---

```python
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
```
**Explanation:** Dictionary mapping numeric activity codes to human-readable names.

**Dictionary in Python:**
- Key-value pairs
- Fast lookup: O(1) time complexity
- Access: `ACTIVITY_LABELS[1]` returns "WALKING"

---

### Section 3: Dataset Download (Lines 43-56)

```python
if not os.path.exists(DATASET_DIR):
```
**Explanation:** Check if dataset folder already exists.

**os.path.exists():** Returns True if path exists, False otherwise.

**Why check?** 
- Avoid re-downloading 60MB file every time
- Efficient programming practice

---

```python
    zip_path = "UCI_HAR_Dataset.zip"
    urllib.request.urlretrieve(DATASET_URL, zip_path)
```
**Explanation:** Download the ZIP file from URL.

**urlretrieve(url, filename):**
- Downloads file from URL
- Saves to local filename
- Blocking operation (waits until complete)

---

```python
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
```
**Explanation:** Extract the ZIP file.

**`with` statement:**
- Context manager pattern
- Automatically closes file when done
- Prevents resource leaks

**`"r"` mode:** Read mode (not writing/creating)

**`extractall(".")`:** Extract to current directory (`.` means current folder)

---

```python
    os.remove(zip_path)
```
**Explanation:** Delete the ZIP file after extraction (no longer needed).

---

### Section 4: Data Loading (Lines 58-108)

```python
X_train = pd.read_csv(
    os.path.join(DATASET_DIR, "train", "X_train.txt"),
    sep=r'\s+',
    header=None,
).values
```
**Explanation:** Load training features from text file.

**Breaking it down:**

| Component | Explanation |
|-----------|-------------|
| `pd.read_csv()` | Read tabular data file |
| `os.path.join()` | Build file path (handles `/` vs `\`) |
| `sep=r'\s+'` | Separator = one or more whitespace characters |
| `header=None` | File has no header row |
| `.values` | Convert DataFrame to NumPy array |

**Why `r'\s+'`?**
- `r` = raw string (backslashes aren't escape characters)
- `\s+` = regex for "one or more whitespace characters"
- Handles both spaces and tabs

**Result:** `X_train` shape is (7352, 561) - 7352 samples, 561 features

---

```python
y_train = pd.read_csv(
    os.path.join(DATASET_DIR, "train", "y_train.txt"), header=None
).values.ravel()
```
**Explanation:** Load training labels.

**`.ravel()`:** Flatten to 1D array.
- Original shape: (7352, 1)
- After ravel: (7352,)

**Why flatten?** Most ML algorithms expect labels as 1D array, not column vector.

---

```python
original_features = X_train.shape[1]
```
**Explanation:** Store number of features (561).

**NumPy shape:**
- `X_train.shape` returns tuple: (rows, columns)
- `X_train.shape[0]` = 7352 (samples)
- `X_train.shape[1]` = 561 (features)

---

### Section 5: Feature Scaling (Lines 87-95)

```python
scaler = StandardScaler()
```
**Explanation:** Create a StandardScaler object.

**Object-Oriented Programming:**
- `StandardScaler` is a class (blueprint)
- `scaler` is an instance (actual object)
- Object stores mean and std after fitting

---

```python
X_train_scaled = scaler.fit_transform(X_train)
```
**Explanation:** Fit scaler to data AND transform in one step.

**Two-step process combined:**
1. **fit:** Calculate mean and std from training data
2. **transform:** Apply formula `(X - mean) / std`

**Why fit on training data only?**
- Simulates real-world scenario
- At deployment, you won't have test data
- Prevents "data leakage"

---

```python
X_test_scaled = scaler.transform(X_test)
```
**Explanation:** Transform test data using TRAINING statistics.

**Critical:** Use `transform()` NOT `fit_transform()`!

**Why?**
- Must use same mean/std as training
- If we fit to test data, we'd "peek" at test distribution
- This is called data leakage - ruins fair evaluation

---

### Section 6: Label Encoding (Lines 97-108)

```python
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
```
**Explanation:** Convert labels 1-6 to 0-5.

**Why 0-indexed?**
- Python/NumPy are 0-indexed
- Many ML algorithms expect labels starting from 0
- Required for one-hot encoding later

---

```python
unique, counts = np.unique(y_train, return_counts=True)
```
**Explanation:** Get unique labels and their frequencies.

**np.unique():**
- `return_counts=True` also returns how many of each
- Returns two arrays: unique values and counts

**Result:**
```
unique = [1, 2, 3, 4, 5, 6]
counts = [1226, 1073, 986, 1286, 1374, 1407]
```

---

### Section 7: PCA (Lines 110-121)

```python
pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)
```
**Explanation:** Create PCA object that keeps 95% of variance.

**n_components options:**
| Value | Meaning |
|-------|---------|
| 0.95 (float < 1) | Keep 95% of variance |
| 10 (int) | Keep exactly 10 components |
| None | Keep all components |

---

```python
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```
**Explanation:** Reduce dimensions.

**Same pattern as StandardScaler:**
- fit on training (learn projection)
- transform both (apply projection)

**Result:**
- Before: (7352, 561)
- After: (7352, 102)

---

```python
print(f"  Reduced features: {pca.n_components_}")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
```
**Explanation:** Print PCA results.

**pca.n_components_:** Actual number of components chosen (102)
**pca.explained_variance_ratio_:** Array of variance explained by each component
**`.sum()`:** Total variance explained (0.9508 = 95.08%)

---

### Section 8: Logistic Regression Training (Lines 130-220)

```python
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    solver="lbfgs",
    n_jobs=-1,
    C=0.1,
)
```
**Explanation:** Create Logistic Regression model with hyperparameters.

**Parameter explanations:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `max_iter=1000` | Maximum iterations for optimizer | Default 100 may not converge |
| `random_state=42` | Random seed | Reproducibility |
| `solver="lbfgs"` | Optimization algorithm | Good for small-medium datasets |
| `n_jobs=-1` | Use all CPU cores | Parallel processing |
| `C=0.1` | Inverse regularization strength | Lower = stronger regularization |

### C Parameter Deep Dive:

**C = 1 / λ (lambda)**

| C Value | Effect |
|---------|--------|
| Large (100) | Weak regularization, complex model |
| Default (1) | Balanced |
| Small (0.1) | Strong regularization, simpler model |

**Why C=0.1?**
- Prevents overfitting on 102 features
- Forces model to find general patterns
- Reduces weight magnitudes

---

```python
lr_model.fit(X_train_pca, y_train_encoded)
```
**Explanation:** Train the model.

**What happens during fit():**
1. Initialize weights randomly
2. For each iteration:
   a. Calculate predictions
   b. Compute loss (how wrong)
   c. Calculate gradients (direction to improve)
   d. Update weights
3. Stop when converged or max_iter reached

---

```python
y_train_pred_lr = lr_model.predict(X_train_pca)
y_test_pred_lr = lr_model.predict(X_test_pca)
```
**Explanation:** Get class predictions (0, 1, 2, 3, 4, or 5).

---

```python
y_train_proba_lr = lr_model.predict_proba(X_train_pca)
```
**Explanation:** Get probability for each class.

**Output shape:** (7352, 6) - probability for each of 6 classes

**Example row:** `[0.85, 0.05, 0.03, 0.02, 0.03, 0.02]`
- 85% chance of class 0 (WALKING)
- 5% chance of class 1, etc.
- Probabilities sum to 1.0

---

```python
lr_train_acc = accuracy_score(y_train_encoded, y_train_pred_lr)
lr_test_acc = accuracy_score(y_test_encoded, y_test_pred_lr)
```
**Explanation:** Calculate accuracy.

**accuracy_score(y_true, y_pred):**
- Compares actual vs predicted
- Returns fraction correct (0.0 to 1.0)

---

```python
lr_train_loss = log_loss(y_train_encoded, y_train_proba_lr)
```
**Explanation:** Calculate log loss (cross-entropy).

**Log Loss Formula:**
```
Loss = -Σ y_true * log(y_pred)
```

**Intuition:**
- Punishes confident wrong predictions heavily
- Rewards confident correct predictions
- Lower = better

---

```python
lr_train_prec = precision_score(y_train_encoded, y_train_pred_lr, average="macro")
```
**Explanation:** Calculate macro-averaged precision.

**`average="macro"`:**
- Calculate precision for each class
- Take unweighted average
- Treats all classes equally

**Alternative averages:**
| Average | Description |
|---------|-------------|
| "macro" | Simple average of all classes |
| "micro" | Global TP/FP/FN, then calculate |
| "weighted" | Weight by class frequency |

---

### Section 9: Learning Curves (Lines 161-220)

```python
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
```
**Explanation:** Generate learning curve data.

**Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `train_sizes=np.linspace(0.1, 1.0, 10)` | [0.1, 0.2, ..., 1.0] | Test at 10%, 20%, ... 100% of data |
| `cv=5` | 5-fold cross-validation | Split data 5 ways for robust estimate |
| `scoring="accuracy"` | Metric to evaluate | Could be "f1", "precision", etc. |

### Cross-Validation (cv=5) Explained:

**Problem:** How to evaluate model without using test set?

**Solution:** Split training into 5 parts (folds):
```
Fold 1: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [VALID]
Fold 2: [TRAIN] [TRAIN] [TRAIN] [VALID] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [VALID] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [VALID] [TRAIN] [TRAIN] [TRAIN]
Fold 5: [VALID] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
```

**Benefits:**
- Uses all data for both training and validation
- More robust metric estimate
- Detects overfitting better

**Output:**
- `train_scores_lr`: Shape (10, 5) - 10 sizes × 5 folds
- `val_scores_lr`: Same shape

---

```python
train_mean_lr = train_scores_lr.mean(axis=1)
train_std_lr = train_scores_lr.std(axis=1)
```
**Explanation:** Calculate mean and standard deviation across folds.

**axis=1:** Average across columns (folds), keep rows (sizes)

**Result:** 10 values (one per training size)

---

```python
plt.figure(figsize=(10, 6))
```
**Explanation:** Create figure 10 inches wide, 6 inches tall.

---

```python
plt.fill_between(
    train_sizes_lr,
    train_mean_lr - train_std_lr,
    train_mean_lr + train_std_lr,
    alpha=0.1,
    color="blue",
)
```
**Explanation:** Draw shaded confidence band.

**fill_between(x, y_low, y_high):**
- Fills area between two lines
- Shows variance in results
- `alpha=0.1` = 10% opacity (transparent)

---

```python
gap_lr = train_mean_lr[-1] - val_mean_lr[-1]
lr_fit_status = (
    "Overfitting"
    if gap_lr > 0.05
    else ("Underfitting" if val_mean_lr[-1] < 0.85 else "Good Fit")
)
```
**Explanation:** Determine model fit status.

**Logic:**
1. Calculate gap between training and validation accuracy
2. If gap > 5%: Overfitting
3. Else if validation < 85%: Underfitting  
4. Else: Good Fit

---

### Section 10: Random Forest Training (Lines 222-314)

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
```
**Explanation:** Create Random Forest with regularization.

**Hyperparameter explanations:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators=100` | 100 trees | Ensemble size (more = better, diminishing returns) |
| `max_depth=15` | Max 15 levels deep | Prevents memorization |
| `min_samples_leaf=5` | At least 5 samples per leaf | Stops creating tiny groups |
| `min_samples_split=10` | Need 10+ samples to split | Reduces tree complexity |

### Why These Values Prevent Overfitting:

**Without limits:**
```
Tree can grow until each leaf has 1 sample
→ Memorizes exact training points
→ Can't generalize
```

**With limits:**
```
Tree stops when nodes are "small enough"
→ Captures general patterns
→ Ignores noise
```

---

### Section 11: Saving Models (Lines 316-344)

```python
import joblib
```
**Explanation:** Import joblib for model serialization.

**Serialization:** Converting Python objects to bytes for storage.

---

```python
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_features.pkl"))
```
**Explanation:** Save scaler to disk.

**`.pkl` extension:** Pickle format (Python serialization)

**Why save preprocessors?**
- Need same transformation at prediction time
- Scaler stores training mean/std
- Must apply identical preprocessing

---

```python
joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
```
**Explanation:** Save trained model.

**What's saved:**
- Model architecture
- Learned weights/coefficients
- Configuration parameters

**To load later:**
```python
lr_model = joblib.load("logistic_regression.pkl")
```

---

# 3. Phase 2: Deep Learning

## File: `phase2_har.py`

### Section 1: Deep Learning Imports (Lines 1-24)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```
**Explanation:** Import TensorFlow and Keras components.

### TensorFlow/Keras Hierarchy:
```
TensorFlow (Low-level)
    └── Keras (High-level API)
            ├── Models (Sequential, Functional)
            ├── Layers (Dense, LSTM, Conv2D, etc.)
            ├── Optimizers (Adam, SGD)
            ├── Callbacks (EarlyStopping, etc.)
            └── Utilities (to_categorical)
```

### Deep Learning Key Concepts:

**Neural Network:**
- Layers of interconnected "neurons"
- Each neuron: weighted sum → activation function
- Learns by adjusting weights

**Layer Types Used:**

| Layer | Purpose |
|-------|---------|
| **LSTM** | Process sequential/time-series data |
| **Dense** | Fully connected layer (standard neural layer) |
| **Dropout** | Regularization (randomly zero neurons) |
| **Bidirectional** | Process sequence forward AND backward |

---

```python
warnings.filterwarnings("ignore")
```
**Explanation:** Suppress warning messages for cleaner output.

---

```python
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
```
**Explanation:** Set seeds for both NumPy and TensorFlow.

**Why both?**
- NumPy randomness: data shuffling, initialization
- TensorFlow randomness: weight initialization, dropout

---

### Section 2: Raw Data Configuration (Lines 50-66)

```python
SIGNAL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
```
**Explanation:** List of 9 raw sensor channels.

**Sensor types:**
| Sensor | Measures | Channels |
|--------|----------|----------|
| Body Accelerometer | Linear acceleration | x, y, z |
| Body Gyroscope | Angular velocity | x, y, z |
| Total Accelerometer | Gravity + body | x, y, z |

---

### Section 3: Data Loading Functions (Lines 90-119)

```python
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
```
**Explanation:** Load all 9 sensor files and combine.

**Step-by-step:**
1. Create empty list
2. Loop through 9 signal files
3. Load each file: shape (7352, 128)
4. After loop: list of 9 arrays
5. Stack: shape (9, 7352, 128)
6. Transpose: shape (7352, 128, 9)

**np.transpose explained:**
```
Original: (9, 7352, 128) = (channels, samples, timesteps)
After:    (7352, 128, 9) = (samples, timesteps, channels)
```

**Why this order?**
- LSTM expects: (samples, timesteps, features)
- Each sample = 128 time steps × 9 channels

---

### Section 4: LSTM Model Architecture (Lines 121-148)

```python
def build_lstm_model(n_timesteps, n_channels, n_classes, bidirectional=False):
    model = Sequential()
```
**Explanation:** Create function to build LSTM model.

**Sequential model:**
- Stack layers one after another
- Output of layer N → input of layer N+1
- Simplest Keras model type

---

```python
    if bidirectional:
        model.add(
            Bidirectional(
                LSTM(100, return_sequences=True), 
                input_shape=(n_timesteps, n_channels)
            )
        )
```
**Explanation:** Add Bidirectional LSTM layer.

### LSTM Deep Dive:

**Problem LSTMs solve:**
Regular neural networks have no "memory". They process each input independently. But time-series data has dependencies!

**Example:** To recognize "WALKING_DOWNSTAIRS", the model needs to see the pattern over time, not just one instant.

**LSTM Architecture:**
```
                    ┌─────────────────┐
                    │  LSTM Cell      │
                    │                 │
    input  ──────► │  forget gate    │ ──────► output
                    │  input gate     │
    prev_state ──► │  output gate    │ ──────► next_state
                    │  cell state     │
                    └─────────────────┘
```

**Three Gates:**
| Gate | Purpose |
|------|---------|
| **Forget Gate** | What to forget from memory |
| **Input Gate** | What new info to add |
| **Output Gate** | What to output |

**Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `100` | 100 LSTM units | Hidden state size |
| `return_sequences=True` | Return all timesteps | For stacking LSTMs |
| `input_shape` | (128, 9) | 128 timesteps, 9 features |

### Bidirectional LSTM:

**Standard LSTM:** Processes left→right only
```
t=1 → t=2 → t=3 → ... → t=128
```

**Bidirectional:** Processes both directions
```
Forward:  t=1 → t=2 → t=3 → ... → t=128
Backward: t=128 → t=127 → ... → t=1
```

**Why bidirectional?**
- Some patterns are easier to detect backwards
- Captures both past and future context
- **Downside:** 2× the parameters, more overfitting risk

---

```python
        model.add(Dropout(0.4))
```
**Explanation:** Add dropout regularization.

### Dropout Explained:

**What it does:** Randomly "turns off" neurons during training.

**Example with Dropout(0.4):**
```
Before: [n1, n2, n3, n4, n5]  (all active)
After:  [n1, 0, n3, 0, n5]    (40% zeroed randomly)
```

**Why it works:**
- Forces network to not rely on any single neuron
- Creates redundancy (multiple neurons learn same pattern)
- Acts like training many smaller networks and averaging

**During prediction:**
- All neurons active
- Weights scaled by (1 - dropout_rate)

---

```python
    model.add(Dense(n_classes, activation="softmax"))
```
**Explanation:** Add output layer.

### Dense Layer:
- Fully connected (every input connects to every output)
- 6 outputs (one per activity class)

### Softmax Activation:

**Formula:**
```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

**What it does:**
- Converts raw scores to probabilities
- All outputs sum to 1.0
- Largest score gets highest probability

**Example:**
```
Raw scores: [2.0, 1.0, 0.5, 0.5, 0.3, 0.1]
Softmax:    [0.52, 0.19, 0.12, 0.12, 0.09, 0.08]
Prediction: Class 0 (52% confidence)
```

---

```python
    model.compile(
        optimizer="adam", 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
```
**Explanation:** Configure training process.

### Optimizer: Adam

**Full name:** Adaptive Moment Estimation

**What it does:**
- Adjusts learning rate for each parameter
- Combines benefits of two other optimizers
- "Smart" gradient descent

**Key features:**
| Feature | Benefit |
|---------|---------|
| Momentum | Smooths updates, escapes local minima |
| Adaptive LR | Different learning rates per parameter |
| Bias correction | Better early training |

### Loss: Categorical Cross-Entropy

**Formula:**
```
Loss = -Σ y_true * log(y_pred)
```

**Why for classification:**
- Measures distance between probability distributions
- Heavily penalizes confident wrong predictions
- Works perfectly with softmax output

---

### Section 5: Training Function (Lines 151-189)

```python
def train_and_evaluate(model, X_train, y_train, X_test, y_test, y_test_encoded, model_name):
    early_stopping = EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        restore_best_weights=True, 
        verbose=1
    )
```
**Explanation:** Create early stopping callback.

### Early Stopping Explained:

**Problem:** Training too long leads to overfitting.

**Solution:** Stop when validation performance stops improving.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `monitor="val_loss"` | Watch validation loss | Stop when this stops improving |
| `patience=10` | Wait 10 epochs | Don't stop on small fluctuations |
| `restore_best_weights=True` | Restore best model | Don't use overfit final weights |

**Example:**
```
Epoch 1: val_loss=0.50
Epoch 5: val_loss=0.30 ← new best
Epoch 10: val_loss=0.31
Epoch 15: val_loss=0.35 ← 10 epochs without improvement, STOP!
Restore weights from Epoch 5
```

---

```python
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=5, 
        min_lr=0.0001, 
        verbose=1
    )
```
**Explanation:** Create learning rate reducer callback.

### ReduceLROnPlateau Explained:

**Problem:** Learning rate that's too high causes oscillation. Too low is slow.

**Solution:** Start high, reduce when stuck.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `factor=0.5` | Multiply LR by 0.5 | Halve the learning rate |
| `patience=5` | Wait 5 epochs | Before reducing |
| `min_lr=0.0001` | Don't go below this | Prevent too-small LR |

**Example:**
```
LR = 0.001, stuck for 5 epochs
→ LR = 0.0005, training continues
→ LR = 0.00025, fine-tuning
→ LR = 0.000125, final adjustments
→ LR = 0.0001, minimum reached
```

---

```python
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )
```
**Explanation:** Train the model.

### Training Parameters:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `epochs=50` | Maximum 50 passes through data | May stop early |
| `batch_size=64` | Process 64 samples at once | Balance speed/memory |
| `validation_split=0.2` | Use 20% for validation | Monitor overfitting |
| `callbacks` | [EarlyStopping, ReduceLR] | Automatic adjustments |

### Epoch vs Batch:

**Epoch:** One complete pass through ALL training data.
**Batch:** One subset processed at a time.

**Example:**
```
7352 samples, batch_size=64
→ 7352/64 = 115 batches per epoch
→ 115 weight updates per epoch
→ 50 epochs = 5750 total updates (max)
```

### Why Batches?

| Aspect | Full Data | Batches |
|--------|-----------|---------|
| Memory | Very high | Manageable |
| Gradient | Exact but slow | Noisy but fast |
| Updates | 1 per epoch | Many per epoch |

---

### Section 6: Data Preprocessing (Lines 192-264)

```python
X_train_2d = X_train.reshape(-1, n_channels)
```
**Explanation:** Reshape 3D data to 2D for scaling.

**Reshape operation:**
```
Original: (7352, 128, 9) = 7352 samples × 128 timesteps × 9 channels
Reshaped: (7352×128, 9) = (940736, 9)
```

**`-1` meaning:** "Calculate this dimension automatically"
```
-1 × 9 = 7352 × 128 × 9
-1 = 940736
```

**Why reshape?**
- StandardScaler expects 2D: (samples, features)
- We want to scale each channel independently
- After scaling, reshape back to 3D

---

```python
y_train_onehot = to_categorical(y_train_encoded, num_classes=n_classes)
```
**Explanation:** Convert labels to one-hot encoding.

### One-Hot Encoding Explained:

**Problem:** Class 2 > Class 1 mathematically, but WALKING isn't "greater than" SITTING!

**Solution:** Create binary vector for each class.

| Label | Integer | One-Hot |
|-------|---------|---------|
| WALKING | 0 | [1,0,0,0,0,0] |
| WALKING_UP | 1 | [0,1,0,0,0,0] |
| WALKING_DOWN | 2 | [0,0,1,0,0,0] |
| SITTING | 3 | [0,0,0,1,0,0] |
| STANDING | 4 | [0,0,0,0,1,0] |
| LAYING | 5 | [0,0,0,0,0,1] |

**Why needed for neural networks:**
- Softmax outputs 6 probabilities
- Compare with one-hot ground truth
- Categorical cross-entropy loss requires this format

---

### Section 7: Visualization (Lines 314-451)

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```
**Explanation:** Create 2×2 grid of subplots.

**Layout:**
```
┌─────────────────┬─────────────────┐
│ LSTM Loss       │ LSTM Accuracy   │
│ axes[0,0]       │ axes[0,1]       │
├─────────────────┼─────────────────┤
│ BiLSTM Loss     │ BiLSTM Accuracy │
│ axes[1,0]       │ axes[1,1]       │
└─────────────────┴─────────────────┘
```

---

```python
axes[0, 0].plot(
    lstm_results["history"].history["loss"], 
    "b-", 
    label="Training Loss", 
    linewidth=2
)
```
**Explanation:** Plot training loss curve.

**history.history:**
- Dictionary of metric values per epoch
- `history["loss"]` = list of training losses
- `history["val_loss"]` = list of validation losses

---

### Section 8: Saving Models (Lines 288-311)

```python
lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
```
**Explanation:** Save TensorFlow model.

**`.keras` format:**
- Native TensorFlow format
- Saves architecture + weights + optimizer state
- Can resume training later

**To load:**
```python
from tensorflow.keras.models import load_model
model = load_model("lstm_model.keras")
```

---

# 4. Model Comparison

## File: `model_comparison.py`

### Section 1: Loading Saved Models (Lines 112-138)

```python
lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
```
**Explanation:** Load previously trained models.

**Why separate loading?**
- scikit-learn: Use `joblib`
- TensorFlow/Keras: Use `load_model`
- Different serialization formats

---

### Section 2: Fit Status Function (Lines 74-81)

```python
def determine_fit_status(train_loss, test_loss, train_acc, test_acc):
    gap = test_loss - train_loss
    if gap > 0.2 or (train_acc - test_acc) > 0.1:
        return "Overfitting"
    elif train_acc < 0.75:
        return "Underfitting"
    else:
        return "Good Fit"
```
**Explanation:** Programmatically determine model fit.

**Logic breakdown:**

| Condition | Status | Reasoning |
|-----------|--------|-----------|
| Loss gap > 0.2 OR accuracy gap > 10% | Overfitting | Model much better on training |
| Training accuracy < 75% | Underfitting | Can't even fit training data |
| Otherwise | Good Fit | Healthy balance |

---

### Section 3: Unified Evaluation (Lines 173-212)

```python
def evaluate_classical(model, X_tr, y_tr, X_te, y_te, name):
    train_pred = model.predict(X_tr)
    train_proba = model.predict_proba(X_tr)
    train_acc = accuracy_score(y_tr, train_pred)
    train_loss = log_loss(y_tr, train_proba)
    # ... test metrics ...
```
**Explanation:** Evaluate classical ML models consistently.

**Return dictionary:** Standardized format for comparison.

---

### Section 4: Visualization (Lines 243-292)

```python
ax4.set_yscale('log')
```
**Explanation:** Use logarithmic scale for y-axis.

**Why log scale for parameters?**
- Models have vastly different sizes
- LR: 612, LSTM: 74,506
- Log scale makes comparison visible

**Linear scale:**
```
200K |                    ■
100K |
  0K |■                   
     LR                 LSTM
     (too small to see)
```

**Log scale:**
```
100K |                    ■
 10K |          ■
  1K |■
     LR       RF       LSTM
     (all visible)
```

---

# 5. Glossary of Terms

| Term | Definition |
|------|------------|
| **Accuracy** | Percentage of correct predictions |
| **Adam** | Adaptive optimizer combining momentum and RMSprop |
| **Batch** | Subset of data processed together |
| **Bidirectional** | Processing sequences in both directions |
| **Callback** | Function called during training at specific points |
| **Categorical Cross-Entropy** | Loss function for multi-class classification |
| **Cross-Validation** | Technique using multiple train/test splits |
| **Dense Layer** | Fully connected neural network layer |
| **Dropout** | Regularization by randomly zeroing neurons |
| **Early Stopping** | Stop training when validation stops improving |
| **Epoch** | One complete pass through training data |
| **Feature** | Input variable used for prediction |
| **Fit** | Train model on data |
| **Gradient Descent** | Optimization by following slope downhill |
| **Hyperparameter** | Configuration set before training |
| **Label** | Target variable to predict |
| **Learning Rate** | Step size for weight updates |
| **Log Loss** | Measures prediction probability quality |
| **LSTM** | Long Short-Term Memory (recurrent network) |
| **One-Hot Encoding** | Binary vector representation of categories |
| **Overfitting** | Memorizing training data, poor generalization |
| **PCA** | Principal Component Analysis (dimensionality reduction) |
| **Precision** | Of predicted positives, how many are correct |
| **Regularization** | Techniques to prevent overfitting |
| **Softmax** | Converts scores to probabilities (sum to 1) |
| **StandardScaler** | Normalizes to mean=0, std=1 |
| **Transform** | Apply learned transformation to data |
| **Underfitting** | Model too simple to learn patterns |
| **Validation Set** | Data used to tune hyperparameters |

---

## Summary

This project demonstrates:

1. **Classical ML (Phase 1):** Feature engineering + simple models can be very effective
2. **Deep Learning (Phase 2):** Learn from raw data but needs more compute/data
3. **Comparison:** Simpler isn't always worse; regularization is crucial

**Key Takeaway:** The best model depends on your data, constraints, and goals—not just complexity!
