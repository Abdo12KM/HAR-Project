# 🏃 Human Activity Recognition (HAR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Predict human activities from smartphone sensor data using Machine Learning and Deep Learning.

## 🎯 Features

- **4 Trained Models:** Logistic Regression, Random Forest, LSTM, Bidirectional LSTM
- **Interactive App:** Upload CSV, simulate data, or use manual sliders
- **Side-by-side Comparison:** See predictions from all models at once

## 🚀 Quick Start

### Local
```bash
pip install -r requirements.txt

# Train models
python phase1_har.py  # Classical ML (LR, RF)
python phase2_har.py  # Deep Learning (LSTM, Bi-LSTM)

# Run app
streamlit run streamlit_app.py
```

### Cloud Deployment
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect and deploy!

## 📊 Model Performance

| Model | Accuracy | Precision | Loss | Fit Status |
|-------|----------|-----------|------|------------|
| Logistic Regression | **93.28%** | 93.38% | 0.181 | Good Fit |
| Standard LSTM | **92.67%** | 92.80% | 0.204 | Good Fit |
| Bidirectional LSTM | 90.53% | 90.87% | 0.326 | Good Fit |
| Random Forest | 88.33% | 88.88% | 0.554 | Overfitting |

## 📁 Project Structure

```
├── streamlit_app.py      # Main Streamlit app
├── models/               # Trained model files
├── phase1_har.py         # Classical ML (LR, RF with tuned hyperparameters)
├── phase2_har.py         # Deep Learning (LSTM & Bidirectional LSTM comparison)
└── har_report.md         # Full project report
```

## 📚 Dataset

[UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) - 10,299 samples, 6 activities

## 🛠️ Technologies

- **ML:** scikit-learn (Logistic Regression, Random Forest, PCA)
- **DL:** TensorFlow/Keras (LSTM, Bidirectional LSTM)
- **App:** Streamlit
- **IDE:** PyCharm (per guidelines)
