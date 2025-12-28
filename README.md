# Diabetes Readmission Risk Prediction

## 🏥 Project Overview
This project leverages machine learning to predict the likelihood of a diabetes patient being readmitted to the hospital within 30 days. It analyzes the **UCI Diabetes 130-US Hospitals Dataset**, identifying key risk factors such as number of lab procedures, medications, and time spent in the hospital.

**Live App**: [https://diabetes-analysis-us.streamlit.app/](https://diabetes-analysis-us.streamlit.app/)

## 📊 Key Features
-   **Risk Prediction**: Interactive form to input patient data and generate a risk probability.
-   **Model Insights**: Visualizes Feature Importance (e.g., impact of Lab Procedures).
-   **Data Explorer**: Interactive charts showing readmission rates by age, race, and gender.

## 🛠️ Technology Stack
-   **Python 3.9+**
-   **Machine Learning**: `scikit-learn` (Random Forest Classifier)
-   **App Framework**: `streamlit`
-   **Data Processing**: `pandas`, `numpy`, `joblib`
-   **Visualization**: `matplotlib`, `seaborn`

## 📂 Project Structure
```text
├── data/           # Dataset files
├── notebooks/      # Jupyter notebooks for EDA and training
├── app/            # Streamlit application source code
│   └── app.py      # Main application file
├── models/         # Saved models (.joblib)
└── requirements.txt
```

## 🚀 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Diabetes_Risk_Prediction.git
    cd Diabetes_Risk_Prediction
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app/app.py
    ```
