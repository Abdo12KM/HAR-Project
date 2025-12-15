# 🏃 Human Activity Recognition (HAR)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://the-human-activity-recognition.streamlit.app/)

**🚀 Live Demo:** [http://the-human-activity-recognition.streamlit.app/](http://the-human-activity-recognition.streamlit.app/)

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
python model_comparison.py # Generate comparison artifacts

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
| Logistic Regression | **93.28%** | **93.38%** | **0.181** | Good Fit (Mild Overfitting) |
| Standard LSTM | 90.80% | 90.84% | 0.314 | Good Fit |
| Bidirectional LSTM | 89.28% | 89.50% | 0.366 | Overfitting |
| Random Forest | 88.33% | 88.88% | 0.554 | Overfitting |

## 💡 Key Findings

1.  **Classical ML Wins:** Logistic Regression outperformed Deep Learning (93.28% vs 90.80%) because well-engineered features captured the activity patterns effectively.
2.  **PCA Efficiency:** Reducing features from 561 to 102 retained 95% variance and significantly sped up training while preventing overfitting.
3.  **Simpler is Better:** The complex Bidirectional LSTM (189K params) performed worse than the simpler Standard LSTM (75K params) and Logistic Regression (612 params).

## 📁 Project Structure

```
├── streamlit_app.py      # Main Streamlit app
├── models/               # Trained model files
├── phase1_har.py         # Classical ML (LR, RF with tuned hyperparameters)
├── phase2_har.py         # Deep Learning (LSTM & Bidirectional LSTM comparison)
├── model_comparison.py   # Script to compare all models and generate plots
├── har_report.md         # Full project report
└── requirements.txt      # Python dependencies
```

## 📚 Dataset

[UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) - 10,299 samples, 6 activities

## 🛠️ Technologies

- **ML:** scikit-learn (Logistic Regression, Random Forest, PCA)
- **DL:** TensorFlow/Keras (LSTM, Bidirectional LSTM)
- **App:** Streamlit
- **IDE:** PyCharm (per guidelines)
