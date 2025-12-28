import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_data():
    print("Attempting to download Diabetes 130-US Hospitals dataset...")
    try:
        # fetch dataset
        diabetes = fetch_ucirepo(id=296)
        
        # data (as pandas dataframes)
        X = diabetes.data.features
        y = diabetes.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), 'diabetes_data.csv')
        df.to_csv(output_path, index=False)
        print(f"Dataset successfully saved to {output_path}")
        print(f"Shape: {df.shape}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Creating synthetic dataset for demonstration purposes...")
        create_synthetic_data()

def create_synthetic_data():
    import numpy as np
    
    n_samples = 1000
    data = {
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Other'], n_samples),
        'gender': np.random.choice(['Female', 'Male'], n_samples),
        'age': np.random.choice(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_procedures': np.random.randint(0, 7, n_samples),
        'num_medications': np.random.randint(1, 40, n_samples),
        'number_outpatient': np.random.randint(0, 5, n_samples),
        'number_emergency': np.random.randint(0, 3, n_samples),
        'number_inpatient': np.random.randint(0, 5, n_samples),
        'diag_1': np.random.randint(250, 1000, n_samples), # roughly diabetes codes often start with 250
        'number_diagnoses': np.random.randint(1, 10, n_samples),
        'max_glu_serum': np.random.choice(['None', '>300', 'Norm', '>200'], n_samples),
        'A1Cresult': np.random.choice(['None', '>7', '>8', 'Norm'], n_samples),
        'diabetesMed': np.random.choice(['Yes', 'No'], n_samples),
        'readmitted': np.random.choice(['NO', '>30', '<30'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    output_path = os.path.join(os.path.dirname(__file__), 'diabetes_data_synthetic.csv')
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to {output_path}")

if __name__ == "__main__":
    download_data()
