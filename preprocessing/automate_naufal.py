import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys

def main():
    print("Starting Heart Dataset Preprocessing")

    # Build dataset path safely
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../namadataset_raw/heart.csv")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at: {dataset_path}")
        sys.exit(1)

    try:
        print("Loading dataset...")
        df = pd.read_csv(dataset_path, sep=None, engine="python")
        print(f"Data loaded successfully: {df.shape}")

        # Reject bad delimiter
        if df.shape[1] == 1:
            print("ERROR: Dataset has only 1 column (wrong delimiter).")
            print(f"Columns read: {df.columns.tolist()}")
            sys.exit(1)

        print(f"Columns detected: {df.columns.tolist()}")
        print(f"Missing values found: {df.isnull().sum().sum()}")

        df_processed = df.copy()

        # Fill missing numeric values
        df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

        # ==========================================
        # AUTO DETECT TARGET COLUMN
        # ==========================================
        possible_targets = ["target", "disease", "output", "label", "class", "heart_disease"]
        target_col = None

        for col in df_processed.columns:
            if col.lower() in [t.lower() for t in possible_targets]:
                target_col = col
                break

        if target_col is None:
            print("ERROR: Could not automatically detect target column.")
            print("Rename target column to `target`, `class`, or `label`.")
            sys.exit(1)

        print(f"Detected target column: {target_col}")

        # Encode target if needed
        if df_processed[target_col].dtype == "object":
            le = LabelEncoder()
            df_processed["target_label"] = le.fit_transform(df_processed[target_col])
        else:
            df_processed["target_label"] = df_processed[target_col]

        # ==========================================
        # AUTO DETECT NUMERIC FEATURES
        # ==========================================
        numerical_features = df_processed.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Remove encoded target
        if "target_label" in numerical_features:
            numerical_features.remove("target_label")

        print(f"Numeric features detected: {numerical_features}")

        # Scaling
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

        # ==========================================
        # TRAIN TEST SPLIT
        # ==========================================
        X = df_processed[numerical_features]
        y = df_processed["target_label"]

        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ==========================================
        # SAVE OUTPUT
        # ==========================================
        output_dir = os.path.join(script_dir, "../namadataset_preprocessing")
        os.makedirs(output_dir, exist_ok=True)

        print("Saving processed files...")

        df_processed.to_csv(os.path.join(output_dir, "heart_processed.csv"), index=False)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(os.path.join(output_dir, "heart_train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "heart_test.csv"), index=False)

        print("Preprocessing completed successfully.")
        print(f"Files saved to: {output_dir}")

    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()