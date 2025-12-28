import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Page Config
st.set_page_config(page_title="Diabetes Readmission Risk", page_icon="🏥", layout="wide")

# --- Load Resources ---
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'models', 'diabetes_model.joblib')
    encoders_path = os.path.join(base_dir, 'models', 'encoders.joblib')
    metadata_path = os.path.join(base_dir, 'models', 'metadata.joblib')
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    metadata = joblib.load(metadata_path)
    return model, encoders, metadata

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'diabetes_data.csv')
    return pd.read_csv(data_path)

try:
    model, encoders, metadata = load_model()
    df_raw = load_data()
except FileNotFoundError:
    st.error("Model or Data not found. Please run the training script first.")
    st.stop()

# --- Application Header ---
st.title("🏥 Diabetes Readmission Risk Predictor")
st.markdown("""
This application utilizes **Random Forest** machine learning to estimate the likelihood of a diabetic patient executing **readmission within 30 days**.
It identifies key risk factors based on the **UCI Diabetes 130-US Hospitals Dataset** (1999-2008).
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("Patient Data")

def user_input_features():
    # Defining input fields based on training features
    # 'race', 'gender', 'age', 'feature_names' in metadata
    
    race = st.sidebar.selectbox("Race", options=encoders['race'].classes_)
    gender = st.sidebar.selectbox("Gender", options=encoders['gender'].classes_)
    age = st.sidebar.selectbox("Age Group", options=encoders['age'].classes_)
    
    time_in_hospital = st.sidebar.slider("Time in Hospital (days)", 1, 14, 3)
    
    st.sidebar.subheader("Medical History")
    num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 132, 40)
    num_procedures = st.sidebar.slider("Number of Procedures", 0, 6, 0)
    num_medications = st.sidebar.slider("Number of Medications", 1, 81, 15)
    
    number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 16, 9)
    
    # Previous visits
    st.sidebar.subheader("Previous Visits (last year)")
    number_outpatient = st.sidebar.number_input("Outpatient Visits", 0, 50, 0)
    number_emergency = st.sidebar.number_input("Emergency Visits", 0, 80, 0)
    number_inpatient = st.sidebar.number_input("Inpatient Visits", 0, 25, 0)
    
    st.sidebar.subheader("Test Results & Treatment")
    max_glu_serum = st.sidebar.selectbox("Glucose Serum Test", options=encoders['max_glu_serum'].classes_)
    A1Cresult = st.sidebar.selectbox("A1C Result", options=encoders['A1Cresult'].classes_)
    
    # Check if insulin/diabetesMed are in encoders (safety check)
    insulin_opts = encoders.get('insulin', None)
    insulin = st.sidebar.selectbox("Insulin", options=insulin_opts.classes_) if insulin_opts else 'No'
    
    med_opts = encoders.get('diabetesMed', None)
    diabetesMed = st.sidebar.selectbox("Diabetes Med Prescribed", options=med_opts.classes_) if med_opts else 'No'
    
    change_opts = encoders.get('change', None)
    change = st.sidebar.selectbox("Change in Meds", options=change_opts.classes_) if change_opts else 'No'

    data = {
        'race': race,
        'gender': gender,
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'number_diagnoses': number_diagnoses,
        'max_glu_serum': max_glu_serum,
        'A1Cresult': A1Cresult,
        'diabetesMed': diabetesMed,
        'insulin': insulin,
        'change': change
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- tabs ---
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Insights", "Data Explorer"])

with tab1:
    st.subheader("Real-time Prediction")
    if st.button("Predict Risk"):
        # Apply Encoding
        processed_input = input_df.copy()
        for col, encoder in encoders.items():
            if col in processed_input.columns:
                processed_input[col] = encoder.transform(processed_input[col])
                
        prediction_proba = model.predict_proba(processed_input)[0][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Readmission Probability", f"{prediction_proba:.1%}")
            
        with col2:
            if prediction_proba > 0.5:
                st.error("High Risk of Readmission (< 30 days)")
            elif prediction_proba > 0.2:
                st.warning("Moderate Risk")
            else:
                st.success("Low Risk")

with tab2:
    st.subheader("Model Performance & Insights")
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_names = metadata.get('feature_names', input_df.columns)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis', ax=ax)
        ax.set_title("Feature Importance (Random Forest)")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
        st.info("This chart shows which factors most strongly influence the prediction model.")

    # Performance Metrics
    st.write("### Test Set Metrics")
    col_a, col_b = st.columns(2)
    col_a.metric("ROC AUC Score", f"{metadata['auc_score']:.3f}")
    
    st.json(metadata['classification_report'])
    
with tab3:
    st.header("Dataset Explorer")

    exp_tab1, exp_tab2 = st.tabs(["Readmission by Age", "Impact of Time in Hospital"])

    with exp_tab1:
        fig_age, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df_raw, x='age', hue='readmitted', palette='viridis', order=sorted(df_raw['age'].unique()))
        plt.xticks(rotation=45)
        plt.title("Readmission Distribution by Age Group")
        st.pyplot(fig_age)

    with exp_tab2:
        fig_hist, ax = plt.subplots(figsize=(8, 4))
        try:
            sns.kdeplot(data=df_raw, x='time_in_hospital', hue='readmitted', palette='coolwarm', fill=True)
            plt.title("Distribution of Time in Hospital vs Readmission")
            st.pyplot(fig_hist)
        except Exception as e:
            st.warning(f"Could not render plot: {e}")
    
    st.caption("Data Source: UCI Machine Learning Repository")
