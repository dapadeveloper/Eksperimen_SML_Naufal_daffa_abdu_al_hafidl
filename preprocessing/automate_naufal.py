import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys

def main():
    print("Starting Wine Quality Preprocessing")

    # Build dataset path safely
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../namadataset_raw/winequality-red.csv")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at: {dataset_path}")
        sys.exit(1)

    try:
        # Auto-detect delimiter (; or ,)
        print("Loading dataset...")
        df = pd.read_csv(dataset_path, sep=None, engine="python")
        print(f"Data loaded successfully: {df.shape}")

        # Detect case where CSV is read as one column
        if df.shape[1] == 1:
            print("ERROR: Dataset only has 1 column. Likely wrong delimiter.")
            print(f"Detected columns: {df.columns.tolist()}")
            sys.exit(1)

        print(f"Columns detected: {df.columns.tolist()}")
        print(f"Missing values: {df.isnull().sum().sum()}")

        df_processed = df.copy()

        # Handle missing values
        if df_processed.isnull().sum().sum() > 0:
            df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

        # Create binary target label
        if "quality" not in df_processed.columns:
            print("ERROR: Column 'quality' not found in dataset.")
            print(f"Available columns: {df_processed.columns.tolist()}")
            sys.exit(1)

        df_processed["quality_category"] = df_processed["quality"].apply(
            lambda x: "good" if x >= 7 else "bad"
        )

        le = LabelEncoder()
        df_processed["quality_label"] = le.fit_transform(df_processed["quality_category"])

        # Scale numerical features
        numerical_features = [
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH",
            "sulphates", "alcohol"
        ]

        missing_cols = [c for c in numerical_features if c not in df_processed.columns]
        if missing_cols:
            print(f"ERROR: The following required columns are missing: {missing_cols}")
            sys.exit(1)

        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

        # Train-test split
        X = df_processed[numerical_features]
        y = df_processed["quality_label"]

        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save preprocessed dataset
        output_dir = os.path.join(script_dir, "../namadataset_preprocessing")
        os.makedirs(output_dir, exist_ok=True)

        print("Saving processed files...")

        df_processed.to_csv(os.path.join(output_dir, "wine_quality_processed.csv"), index=False)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(os.path.join(output_dir, "wine_quality_train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "wine_quality_test.csv"), index=False)

        print("Preprocessing completed successfully.")
        print(f"Files saved to: {output_dir}")

    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
