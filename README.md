# 🏃 Human Activity Recognition (HAR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Predict human activities from smartphone sensor data using Machine Learning and Deep Learning.

## 🎯 Features

- **3 Trained Models:** Logistic Regression (93.04%), Random Forest (88.46%), LSTM (93.01%)
- **Interactive App:** Upload CSV, simulate data, or use manual sliders
- **Side-by-side Comparison:** See predictions from all models at once

## 🚀 Quick Start

### Local
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Cloud Deployment
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect and deploy!

## 📊 Model Performance

| Model | Accuracy | Precision | Loss |
|-------|----------|-----------|------|
| Logistic Regression | **93.04%** | 93.07% | 0.213 |
| LSTM | **93.01%** | 93.23% | 0.250 |
| Random Forest | 88.46% | 88.91% | 0.519 |

## 📁 Project Structure

```
├── streamlit_app.py      # Main Streamlit app
├── models/               # Trained model files
├── phase1_har.py         # Classical ML training
├── phase2_har.py         # LSTM training
├── model_comparison.py   # Comparison & visualization
└── har_report.md         # Full project report
```

## 📚 Dataset

[UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) - 10,299 samples, 6 activities
