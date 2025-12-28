import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_model():
    # Paths
    print("Loading data...")
    # Go up one level from notebooks/ to root, then to data/
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'diabetes_data.csv')
    
    df = pd.read_csv(data_path)
    
    # --- Preprocessing ---
    print("Preprocessing...")
    
    # Target: 1 if readmitted < 30 days, else 0
    df['target'] = (df['readmitted'] == '<30').astype(int)
    
    # Feature Selection
    features = [
        'race', 'gender', 'age', 'time_in_hospital', 
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult', 
        'diabetesMed', 'insulin', 'change'
    ]
    
    # Keep only selected features
    X = df[features].copy()
    y = df['target']
    
    # Handling missing values and encoding
    # '?' is missing. Replace with 'Unknown' for categorical
    cardinality_cols = [c for c in X.columns if X[c].dtype == 'object']
    
    encoders = {}
    for col in cardinality_cols:
        X[col] = X[col].replace('?', 'Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Train Model ---
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # --- Evaluate ---
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {auc:.4f}")
    
    # --- Save Model, Encoders & Metad data ---
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(clf, os.path.join(model_dir, 'diabetes_model.joblib'))
    joblib.dump(encoders, os.path.join(model_dir, 'encoders.joblib'))
    
    # Save metadata for the app
    metadata = {
        'feature_names': features,
        'auc_score': auc,
        'classification_report': report
    }
    joblib.dump(metadata, os.path.join(model_dir, 'metadata.joblib'))
    
    print(f"Model and artifacts saved to {model_dir}")

if __name__ == "__main__":
    train_model()
