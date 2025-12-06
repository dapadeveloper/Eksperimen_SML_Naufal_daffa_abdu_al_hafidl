# automate_Naufal.py

"""
Automasi Eksperimen Banknote Authentication

Struktur:
- Dataset mentah: ../namadataset_raw/banknote_authentication.csv
- Output preprocessing: ../namadataset_preprocessing/
- Model tersimpan: banknote_rf_model.pkl
- Prediksi batch: dari file CSV input → output CSV
"""

import pandas as pd
import joblib
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
RAW_DATA_PATH = "../namadataset_raw/banknote_authentication.csv"
PREPROCESS_DIR = "../namadataset_preprocessing/"
MODEL_PATH = os.path.join(PREPROCESS_DIR, "banknote_rf_model.pkl")

# Pastikan folder preprocessing ada
os.makedirs(PREPROCESS_DIR, exist_ok=True)

def load_data(path=RAW_DATA_PATH):
    """Load dataset dari folder raw"""
    cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(path, header=None, names=cols)
    return df

def preprocessing(df):
    """Proses preprocessing sederhana"""
    # Hapus duplikat
    df = df.drop_duplicates()
    
    # Cek missing value (untuk dataset ini seharusnya tidak ada)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    # Simpan hasil preprocessing ke folder
    preprocessed_path = os.path.join(PREPROCESS_DIR, "banknote_preprocessed.csv")
    df.to_csv(preprocessed_path, index=False)
    print(f"Hasil preprocessing disimpan ke {preprocessed_path}")
    return df

def train_and_save(df, model_path=MODEL_PATH):
    """Train RandomForest dan simpan model"""
    X = df[['variance','skewness','curtosis','entropy']]
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    print("=== Evaluasi Model ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Simpan model
    joblib.dump(model, model_path)
    print(f"Model tersimpan ke {model_path}")

def predict_from_file(model_path, input_csv, output_csv):
    """Prediksi data baru dari CSV"""
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    required_cols = ['variance','skewness','curtosis','entropy']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV harus memiliki kolom: {required_cols}")
    X = df[required_cols]
    df['predicted_class'] = model.predict(X)
    df.to_csv(output_csv, index=False)
    print(f"Prediksi disimpan ke {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Banknote Authentication")
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model menggunakan dataset mentah + preprocessing"
    )
    
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=('INPUT_CSV', 'OUTPUT_CSV'),
        help="Prediksi kelas untuk input CSV dan simpan ke output CSV"
    )
    
    args = parser.parse_args()
    
    if args.train:
        df = load_data()
        df = preprocessing(df)
        train_and_save(df)
    elif args.predict:
        input_csv, output_csv = args.predict
        predict_from_file(MODEL_PATH, input_csv, output_csv)
    else:
        print("Tidak ada argumen diberikan. Gunakan --train atau --predict.")
