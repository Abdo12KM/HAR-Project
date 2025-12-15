"""
Human Activity Recognition (HAR) - Streamlit App
=================================================
Interactive app for activity prediction using all trained models
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# TensorFlow with reduced logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="HAR Activity Predictor",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================
MODELS_DIR = "models"
ACTIVITY_LABELS = {
    0: "🚶 WALKING",
    1: "⬆️ WALKING UPSTAIRS",
    2: "⬇️ WALKING DOWNSTAIRS",
    3: "🪑 SITTING",
    4: "🧍 STANDING",
    5: "🛏️ LAYING",
}

ACTIVITY_COLORS = {
    0: "#3498db",
    1: "#2ecc71",
    2: "#e74c3c",
    3: "#f39c12",
    4: "#9b59b6",
    5: "#1abc9c",
}

# Feature names for sliders (top 10 most important features)
TOP_FEATURES = [
    "tBodyAcc-mean()-X",
    "tBodyAcc-mean()-Y",
    "tBodyAcc-mean()-Z",
    "tGravityAcc-mean()-X",
    "tGravityAcc-mean()-Y",
    "tGravityAcc-mean()-Z",
    "tBodyAccJerk-mean()-X",
    "tBodyAccJerk-mean()-Y",
    "tBodyGyro-mean()-X",
    "tBodyGyro-mean()-Y",
]


# =============================================================================
# LOAD MODELS
# =============================================================================
@st.cache_resource
def load_models():
    """Load all trained models and preprocessors."""
    models = {}
    try:
        models['lr'] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
        models['rf'] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        models['lstm'] = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
        # Try to load Bidirectional LSTM if available
        bilstm_path = os.path.join(MODELS_DIR, "bilstm_model.keras")
        if os.path.exists(bilstm_path):
            models['bilstm'] = tf.keras.models.load_model(bilstm_path)
        models['scaler_feat'] = joblib.load(os.path.join(MODELS_DIR, "scaler_features.pkl"))
        models['scaler_raw'] = joblib.load(os.path.join(MODELS_DIR, "scaler_raw.pkl"))
        models['pca'] = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
        models['encoder'] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        return models, True
    except Exception as e:
        return {"error": str(e)}, False


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def predict_with_classical(model, features, scaler, pca):
    """Predict using classical ML models (LR or RF)."""
    # Scale and apply PCA
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_pca = pca.transform(features_scaled)
    
    # Predict
    prediction = model.predict(features_pca)[0]
    probabilities = model.predict_proba(features_pca)[0]
    
    return prediction, probabilities


def predict_with_lstm(model, raw_signals, scaler):
    """Predict using LSTM model with raw signals."""
    # raw_signals shape: (128, 9)
    n_timesteps, n_channels = raw_signals.shape
    
    # Scale
    raw_scaled = scaler.transform(raw_signals)
    raw_scaled = raw_scaled.reshape(1, n_timesteps, n_channels)
    
    # Predict
    probabilities = model.predict(raw_scaled, verbose=0)[0]
    prediction = np.argmax(probabilities)
    
    return prediction, probabilities


def generate_simulated_data(activity_type):
    """Generate simulated sensor data for a given activity."""
    np.random.seed(None)  # Random each time
    
    # Base patterns for different activities
    patterns = {
        0: {"acc_mean": 0.9, "acc_std": 0.3, "gyro_mean": 0.1, "gyro_std": 0.2},  # Walking
        1: {"acc_mean": 0.7, "acc_std": 0.4, "gyro_mean": 0.15, "gyro_std": 0.25},  # Up
        2: {"acc_mean": 1.1, "acc_std": 0.35, "gyro_mean": 0.12, "gyro_std": 0.22},  # Down
        3: {"acc_mean": 0.1, "acc_std": 0.05, "gyro_mean": 0.02, "gyro_std": 0.03},  # Sitting
        4: {"acc_mean": 0.15, "acc_std": 0.08, "gyro_mean": 0.03, "gyro_std": 0.04},  # Standing
        5: {"acc_mean": 0.05, "acc_std": 0.02, "gyro_mean": 0.01, "gyro_std": 0.02},  # Laying
    }
    
    p = patterns[activity_type]
    
    # Generate 561 features with activity-specific patterns
    features = np.zeros(561)
    
    # Mean features (indices 0-40 approximately)
    features[:20] = np.random.normal(p["acc_mean"], p["acc_std"], 20)
    features[20:40] = np.random.normal(p["gyro_mean"], p["gyro_std"], 20)
    
    # Std features
    features[40:80] = np.abs(np.random.normal(p["acc_std"], 0.1, 40))
    
    # Other features (random with some structure)
    features[80:] = np.random.normal(0, 0.5, 481)
    
    # Generate raw signals (128, 9)
    raw_signals = np.zeros((128, 9))
    
    # Body acceleration (channels 0-2)
    for i in range(3):
        raw_signals[:, i] = np.random.normal(p["acc_mean"], p["acc_std"], 128)
        if activity_type in [0, 1, 2]:  # Walking activities
            raw_signals[:, i] += 0.3 * np.sin(np.linspace(0, 4*np.pi, 128))
    
    # Body gyro (channels 3-5)
    for i in range(3, 6):
        raw_signals[:, i] = np.random.normal(p["gyro_mean"], p["gyro_std"], 128)
    
    # Total acceleration (channels 6-8)
    for i in range(6, 9):
        raw_signals[:, i] = raw_signals[:, i-6] + np.random.normal(0.98, 0.02, 128)
    
    return features, raw_signals


# =============================================================================
# UI COMPONENTS
# =============================================================================
def display_prediction_result(prediction, probabilities, model_name):
    """Display prediction result with confidence bars."""
    activity_name = ACTIVITY_LABELS[prediction]
    confidence = probabilities[prediction] * 100
    color = ACTIVITY_COLORS[prediction]
    
    st.markdown(f"### {model_name}")
    st.markdown(f"**Predicted Activity:** {activity_name}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    
    # Progress bar for confidence
    st.progress(confidence / 100)
    
    # All class probabilities
    with st.expander("View all class probabilities"):
        for i, prob in enumerate(probabilities):
            st.write(f"{ACTIVITY_LABELS[i]}: {prob*100:.1f}%")
            st.progress(prob)


def display_comparison(results):
    """Display side-by-side comparison of all models."""
    num_models = len(results)
    cols = st.columns(num_models)
    
    for i, (model_name, data) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"### {model_name}")
            st.markdown(f"**{ACTIVITY_LABELS[data['prediction']]}**")
            st.metric("Confidence", f"{data['confidence']:.1f}%")
            
            # Mini bar chart
            chart_data = pd.DataFrame({
                'Activity': [ACTIVITY_LABELS[j].split(' ')[0] for j in range(6)],
                'Probability': data['probabilities'] * 100
            })
            st.bar_chart(chart_data.set_index('Activity'), height=200)


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("🏃 Human Activity Recognition")
    st.markdown("### Predict activities using trained ML models")
    
    # Load models
    models, success = load_models()
    
    if not success:
        st.error(f"⚠️ Could not load models. Please run `phase1_har.py` and `phase2_har.py` first.")
        st.error(f"Error: {models.get('error', 'Unknown error')}")
        st.stop()
    
    # Show loaded models
    loaded_models = []
    if 'lr' in models: loaded_models.append("Logistic Regression")
    if 'rf' in models: loaded_models.append("Random Forest")
    if 'lstm' in models: loaded_models.append("LSTM")
    if 'bilstm' in models: loaded_models.append("Bidirectional LSTM")
    
    st.success(f"✅ Loaded models: {', '.join(loaded_models)}")
    
    # Sidebar - Input method selection
    st.sidebar.title("⚙️ Input Configuration")
    
    input_method = st.sidebar.radio(
        "Select Input Method",
        ["🎲 Simulated Data", "📁 Upload CSV", "🎛️ Manual Sliders"]
    )
    
    # Initialize variables
    features = None
    raw_signals = None
    
    # =============================================================================
    # INPUT METHOD: SIMULATED DATA
    # =============================================================================
    if input_method == "🎲 Simulated Data":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Simulate Activity Data")
        
        activity_to_simulate = st.sidebar.selectbox(
            "Select activity to simulate",
            options=list(ACTIVITY_LABELS.keys()),
            format_func=lambda x: ACTIVITY_LABELS[x]
        )
        
        if st.sidebar.button("🔄 Generate New Data", use_container_width=True):
            st.session_state['regenerate'] = True
        
        features, raw_signals = generate_simulated_data(activity_to_simulate)
        st.info(f"📊 Simulated data generated for: {ACTIVITY_LABELS[activity_to_simulate]}")
    
    # =============================================================================
    # INPUT METHOD: CSV UPLOAD
    # =============================================================================
    elif input_method == "📁 Upload CSV":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Upload Features CSV")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with 561 features",
            type=['csv', 'txt'],
            help="CSV file with 561 features per row"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None, sep=r'\s+|,', engine='python')
                if df.shape[1] == 561:
                    features = df.iloc[0].values
                    st.success(f"✅ Loaded {df.shape[0]} samples with 561 features")
                    
                    # Generate corresponding raw signals (approximate)
                    raw_signals = np.random.randn(128, 9) * 0.5
                else:
                    st.error(f"Expected 561 features, got {df.shape[1]}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.warning("Please upload a CSV file with 561 pre-extracted features")
    
    # =============================================================================
    # INPUT METHOD: MANUAL SLIDERS
    # =============================================================================
    elif input_method == "🎛️ Manual Sliders":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Adjust Key Features")
        
        slider_values = {}
        for feat_name in TOP_FEATURES:
            slider_values[feat_name] = st.sidebar.slider(
                feat_name,
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1
            )
        
        # Generate full feature vector with slider values
        features = np.zeros(561)
        for i, (name, val) in enumerate(slider_values.items()):
            features[i] = val
        features[len(slider_values):] = np.random.randn(561 - len(slider_values)) * 0.1
        
        # Generate raw signals based on slider values
        raw_signals = np.zeros((128, 9))
        for i in range(9):
            base_val = list(slider_values.values())[i] if i < len(slider_values) else 0
            raw_signals[:, i] = base_val + np.random.randn(128) * 0.2
    
    # =============================================================================
    # MAKE PREDICTIONS
    # =============================================================================
    if features is not None:
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Get predictions from all models
        results = {}
        
        # Logistic Regression
        lr_pred, lr_proba = predict_with_classical(
            models['lr'], features, models['scaler_feat'], models['pca']
        )
        results['Logistic Regression'] = {
            'prediction': lr_pred,
            'probabilities': lr_proba,
            'confidence': lr_proba[lr_pred] * 100
        }
        
        # Random Forest
        rf_pred, rf_proba = predict_with_classical(
            models['rf'], features, models['scaler_feat'], models['pca']
        )
        results['Random Forest'] = {
            'prediction': rf_pred,
            'probabilities': rf_proba,
            'confidence': rf_proba[rf_pred] * 100
        }
        
        # LSTM (only if raw signals available)
        if raw_signals is not None:
            lstm_pred, lstm_proba = predict_with_lstm(
                models['lstm'], raw_signals, models['scaler_raw']
            )
            results['LSTM'] = {
                'prediction': lstm_pred,
                'probabilities': lstm_proba,
                'confidence': lstm_proba[lstm_pred] * 100
            }
            
            # Bidirectional LSTM (if available)
            if 'bilstm' in models:
                bilstm_pred, bilstm_proba = predict_with_lstm(
                    models['bilstm'], raw_signals, models['scaler_raw']
                )
                results['Bi-LSTM'] = {
                    'prediction': bilstm_pred,
                    'probabilities': bilstm_proba,
                    'confidence': bilstm_proba[bilstm_pred] * 100
                }
        
        # Display comparison
        display_comparison(results)
        
        # Summary
        st.markdown("---")
        st.markdown("### 📋 Summary")
        
        predictions = [ACTIVITY_LABELS[r['prediction']] for r in results.values()]
        if len(set(predictions)) == 1:
            st.success(f"✅ All models agree: **{predictions[0]}**")
        else:
            st.warning("⚠️ Models disagree on the prediction")
            for name, data in results.items():
                st.write(f"- {name}: {ACTIVITY_LABELS[data['prediction']]}")
    
    # =============================================================================
    # FOOTER
    # =============================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>HAR Project</small><br>
        <small>Models: Logistic Regression, Random Forest, LSTM, Bidirectional LSTM</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
