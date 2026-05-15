"""
data_preprocessing.py - Data loading, cleaning, scaling, and SMOTE
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import joblib
def load_and_clean_data(filepath):
    """
    Load and clean the UCI Credit Default dataset
    Parameters:
    -----------
    filepath: str, path to Excel file
    Returns:
    --------
    X: DataFrame with features
    y: Series with target
    """
    # Load Excel
    df = pd.read_excel(filepath, engine='openpyxl', header=1)
    print(f"Initial shape: {df.shape}")
    print(df.head())
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Rename target column if needed
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
    
    # Set expected column names
    expected_cols = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                     'default']
    
    if len(df.columns) == 25:
        df.columns = expected_cols
    
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    
    # Drop ID column
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    
    print(df.head())
    
    # Clean EDUCATION
    count_ed = df['EDUCATION'].value_counts()
    print(count_ed)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    
    # Clean MARRIAGE
    count_mar = (df['MARRIAGE'] == 0).sum()
    print(count_mar)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

    print(df.head())
    # Check missing values
    print("\nMissing values:\n", df.isnull().sum())
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Remove duplicates
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")
    # Check data types
    print(df.dtypes)
    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(y.value_counts())
    print(f"Default Rate: {y.mean():.3f}")
    return X, y
def preprocess_with_scaling_and_smote(X, y, test_size=0.2, random_state=42):
    """
    Scale features, split train/test, apply SMOTE only on training set
    Parameters:
    -----------
    X: features DataFrame
    y: target Series
    test_size: float, proportion for testing
    random_state: int, for reproducibility
    Returns:
    --------
    X_train_resampled: scaled + SMOTE training features
    X_test_scaled: scaled test features
    y_train_resampled: balanced training labels
    y_test: test labels
    scaler: fitted StandardScaler
    """
    # Split first (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")
    print(f"Default rate in training: {y_train.mean():.3f}")
    print(f"Default rate in testing:  {y_test.mean():.3f}")
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("Scaler fitted on training data")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled training shape: {X_train_scaled.shape}")
    print(f"Scaled testing shape:  {X_test_scaled.shape}")
    # Convert to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    print("Mean of scaled training features (first 5):")
    print(X_train_scaled.mean().head())
    print("\nStd of scaled training features (first 5):")
    print(X_train_scaled.std().head())
    # Apply SMOTE
    print("\nBefore SMOTE:")
    print(f"Class 0 (No Default):  {(y_train == 0).sum()}")
    print(f"Class 1 (Default):     {(y_train == 1).sum()}")
    print(f"Default rate: {y_train.mean():.3f}")
    
    smote = SMOTE(sampling_strategy='auto', random_state=random_state, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print("\nAfter SMOTE:")
    print(f"Class 0 (No Default):  {(y_train_resampled == 0).sum()}")
    print(f"Class 1 (Default):     {(y_train_resampled == 1).sum()}")
    print(f"Default rate: {y_train_resampled.mean():.3f}")
    print(f"New training set size: {X_train_resampled.shape[0]} samples")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(X_train_resampled, 'data/processed/X_train.pkl')
    joblib.dump(y_train_resampled, 'data/processed/y_train.pkl')
    joblib.dump(X_test_scaled, 'data/processed/X_test.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    print("Data saved successfully")
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler
if __name__ == "__main__":
    print("Testing data preprocessing module")
    # Test with sample data path
    X, y = load_and_clean_data('data/raw/credit card clients.xlsx')
    X_train, X_test, y_train, y_test, scaler = preprocess_with_scaling_and_smote(X, y)
    print("\nPreprocessing test complete!")