import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys

def main():
    print("Starting Wine Quality Preprocessing")
    
    # Check if dataset exists
    dataset_path = '../namadataset_raw/winequality-red.csv'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    try:
        # Load data with correct separator
        df = pd.read_csv(dataset_path, sep=';')
        print(f"Data loaded successfully: {df.shape}")
        
        # Basic info
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Preprocessing
        df_processed = df.copy()
        
        # Handle missing values
        if df_processed.isnull().sum().sum() > 0:
            df_processed = df_processed.fillna(df_processed.median())
        
        # Create binary target
        df_processed['quality_category'] = df_processed['quality'].apply(
            lambda x: 'good' if x >= 7 else 'bad'
        )
        le = LabelEncoder()
        df_processed['quality_label'] = le.fit_transform(df_processed['quality_category'])
        
        # Scale numerical features
        numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 
                             'residual sugar', 'chlorides', 'free sulfur dioxide', 
                             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
        
        # Split data
        X = df_processed[numerical_features]
        y = df_processed['quality_label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data
        output_dir = '../namadataset_preprocessing'
        os.makedirs(output_dir, exist_ok=True)
        
        df_processed.to_csv(f'{output_dir}/wine_quality_processed.csv', index=False)
        
        # Save train/test sets
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(f'{output_dir}/wine_quality_train.csv', index=False)
        test_df.to_csv(f'{output_dir}/wine_quality_test.csv', index=False)
        
        print("Preprocessing completed successfully!")
        print(f"Files saved to {output_dir}/")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()