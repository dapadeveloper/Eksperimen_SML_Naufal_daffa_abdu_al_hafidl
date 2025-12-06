# automate_Naufal.py

import pandas as pd
import joblib
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths absolut
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # folder script
RAW_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../namadataset_raw/banknote_authentication.csv"))
PREPROCESS_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
PREPROCESSED_PATH = os.path.join(PREPROCESS_DIR, "banknote_preprocessed.csv")
MODEL_PATH = os.path.join(PREPROCESS_DIR, "banknote_rf_model.pkl")

# Pastikan folder preprocessing ada
os.makedirs(PREPROCESS_DIR, exist_ok=True)

def load_data(path=RAW_DATA_PATH):
    """
    Load dataset dengan aman.
    Jika CSV sudah memiliki header, gunakan header=0.
    """
    cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(path, header=0, names=cols)  # header=0 untuk skip baris header asli
    print(f"Dataset loaded: {path} ({len(df)} rows)")
    print("Class distribution:\n", df['class'].value_counts())
    return df

def preprocessing(df):
    """Preprocessing: hapus duplikat & missing value"""
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Preprocessing completed. File saved to: {PREPROCESSED_PATH}")
    return df

def train_and_save(df):
    """Train RandomForest dan simpan model"""
    X = df[['variance','skewness','curtosis','entropy']]
    y = df['class']

    # Cek jumlah tiap kelas untuk stratify
    min_class_count = y.value_counts().min()
    stratify_option = y if min_class_count >= 2 else None
    if stratify_option is None:
        print("WARNING: Some class has <2 samples. Stratify disabled for train_test_split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print("=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

def predict_from_file(model_path, input_csv, output_csv):
    """Prediksi batch dari CSV input → output CSV"""
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    required_cols = ['variance','skewness','curtosis','entropy']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV harus memiliki kolom: {required_cols}")
    df['predicted_class'] = model.predict(df[required_cols])
    df.to_csv(output_csv, index=False)
    print(f"Prediction saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model & preprocess dataset")
    parser.add_argument("--predict", nargs=2, metavar=('INPUT_CSV', 'OUTPUT_CSV'), help="Predict batch data")
    args = parser.parse_args()

    if args.train:
        df = load_data()
        df = preprocessing(df)
        train_and_save(df)
    elif args.predict:
        input_csv, output_csv = args.predict
        predict_from_file(MODEL_PATH, input_csv, output_csv)
    else:
        print("No arguments provided. Use --train or --predict.")
