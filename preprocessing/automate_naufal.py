import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class WineQualityPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.numerical_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 
            'residual sugar', 'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]

    # =============================
    # FIXED LOAD DATA
    # =============================
    def load_data(self, file_path):
        """Load wine quality dataset with auto delimiter detection"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            print("Reading dataset...")
            
            # FIX: Auto detect delimiter
            df = pd.read_csv(file_path, sep=None, engine='python')

            print("Data loaded successfully, shape:", df.shape)

            # FIX: If only 1 column (bad format), split manually
            if df.shape[1] == 1:
                print("⚠ WARNING: Dataset appears incorrectly formatted (1 column only).")
                print("Attempting to split by comma...")

                df = df.iloc[:, 0].str.split(',', expand=True)

                # Rename columns properly
                df.columns = [
                    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                    'density', 'pH', 'sulphates', 'alcohol', 'quality'
                ]

                print("Dataset successfully fixed → new shape:", df.shape)

            return df

        except Exception as e:
            print("Error loading data:", e)
            return None

    # =============================
    # EDA
    # =============================
    def explore_data(self, df):
        """Perform automated EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)

        print("Dataset Shape:", df.shape)
        print("Features:", len(df.columns))
        print("Samples:", len(df))

        print("\nData Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        print(missing_values)

        print("\nStatistical Summary:")
        print(df.describe())

        # FIX: ensure 'quality' exists
        print("\nTarget Distribution (Quality):")
        if 'quality' in df.columns:
            quality_dist = df['quality'].value_counts().sort_index()
            print(quality_dist)
        else:
            print("⚠ WARNING: 'quality' column not found!")
            quality_dist = {}

        eda_info = {
            'shape': df.shape,
            'missing_values': missing_values.to_dict(),
            'quality_distribution': quality_dist,
            'features': list(df.columns)
        }

        return eda_info

    # =============================
    # PREPROCESSING
    # =============================
    def preprocess_data(self, df, classification_type='binary'):
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)

        df_processed = df.copy()

        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)

        # FIX: ensure quality exists
        if 'quality' not in df_processed.columns:
            raise KeyError("ERROR: Column 'quality' missing from dataset after loading!")

        # Create target variable
        if classification_type == 'binary':
            df_processed = self.create_binary_target(df_processed)
            target_column = 'quality_label'
        else:
            target_column = 'quality'

        # Scale numerical features
        df_processed = self.scale_features(df_processed)

        # Prepare features and target
        X = df_processed[self.numerical_features]
        y = df_processed[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Preprocessing completed")
        print("Training set:", X_train.shape)
        print("Test set:", X_test.shape)
        print("Target distribution - Train:", pd.Series(y_train).value_counts().to_dict())
        print("Target distribution - Test:", pd.Series(y_test).value_counts().to_dict())

        return X_train, X_test, y_train, y_test, df_processed

    def handle_missing_values(self, df):
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print("Handling", missing_count, "missing values")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        else:
            print("No missing values found")
        return df

    def create_binary_target(self, df):
        df['quality_category'] = df['quality'].apply(lambda x: 'good' if x >= 7 else 'bad')
        df['quality_label'] = self.label_encoder.fit_transform(df['quality_category'])

        good_count = (df['quality_label'] == 1).sum()
        bad_count = (df['quality_label'] == 0).sum()

        print("Binary target created:")
        print("Good wines (>=7):", good_count)
        print("Bad wines (<7):", bad_count)

        return df

    def scale_features(self, df):
        print("Scaling numerical features")
        df[self.numerical_features] = self.scaler.fit_transform(df[self.numerical_features])
        return df


# =======================================
# MAIN AUTOMATION FUNCTION (NO ERROR)
# =======================================
def automate_wine_preprocessing(input_path, output_path, classification_type='binary'):
    preprocessor = WineQualityPreprocessor()

    try:
        print("Starting automated wine quality preprocessing")
        print("Input path:", input_path)
        print("Output path:", output_path)

        df = preprocessor.load_data(input_path)
        if df is None:
            raise Exception("Failed to load dataset")

        eda_info = preprocessor.explore_data(df)

        X_train, X_test, y_train, y_test, df_processed = preprocessor.preprocess_data(
            df, classification_type=classification_type
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_processed.to_csv(output_path, index=False)
        print("Processed data saved to:", output_path)

        print("\nPREPROCESSING SUMMARY")
        print("="*50)
        print("Original samples:", len(df))
        print("Processed samples:", len(df_processed))

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'processed_data': df_processed,
            'preprocessor': preprocessor,
            'eda_info': eda_info,
        }

    except Exception as e:
        print("Error in automated preprocessing:", e)
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    input_path = '../namadataset_raw/winequality-red.csv'
    output_path = '../namadataset_preprocessing/wine_quality_processed.csv'

    result = automate_wine_preprocessing(
        input_path=input_path,
        output_path=output_path,
        classification_type='binary'
    )

    if result is None:
        sys.exit(1)
    else:
        print("Script executed successfully")
