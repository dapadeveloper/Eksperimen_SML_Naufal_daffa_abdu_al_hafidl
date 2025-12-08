import pandas as pd
import argparse
import os
from datetime import datetime

# =========================
# PATH CONFIG
# =========================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # folder script

RAW_DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../namadataset_raw/banknote_authentication.csv")
)

PREPROCESS_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
PREPROCESSED_PATH = os.path.join(PREPROCESS_DIR, "banknote_preprocessed.csv")

#  Folder khusus untuk EXPORT dataset
EXPORT_DIR = os.path.join(BASE_DIR, "namadataset_export")

# Pastikan folder ada
os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_data(path=RAW_DATA_PATH):
    """
    Load dataset dengan aman.
    Jika CSV sudah memiliki header, gunakan header=0.
    """
    cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(path, header=0, names=cols)

    print(f"Dataset loaded: {path}")
    print(f"Jumlah data: {len(df)} baris")
    print("Distribusi kelas:\n", df['class'].value_counts())

    return df


# =========================
# PREPROCESSING
# =========================
def preprocessing(df):
    """
    Preprocessing:
    - Hapus duplikat
    - Hapus missing value
    - Simpan hasil preprocessing ke CSV
    - Export dataset preprocessing (timestamp)
    """

    print("\nMulai preprocessing...")

    # Hapus duplikat
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)

    print(f"Duplikat dihapus: {before_dup - after_dup}")

    # Hapus missing value jika ada
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        df = df.dropna()
        print(f"Missing value dihapus: {missing_total}")
    else:
        print("Tidak ada missing value")

    # =========================
    # SIMPAN PREPROCESSING UTAMA
    # =========================
    df.to_csv(PREPROCESSED_PATH, index=False)

    print("\n Preprocessing selesai.")
    print(f" File preprocessing tersimpan di:\n{PREPROCESSED_PATH}")

    # =========================
    # EXPORT DATASET (UNTUK EKSPERIMEN / MLflow)
    # =========================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"banknote_preprocessed_export_{timestamp}.csv"
    export_path = os.path.join(EXPORT_DIR, export_filename)

    df.to_csv(export_path, index=False)

    print(f" File export dataset berhasil dibuat di:\n{export_path}")

    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate preprocessing dataset + export")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Load raw data, preprocess, simpan, dan export dataset"
    )
    args = parser.parse_args()

    if args.preprocess:
        df = load_data()
        preprocessing(df)
    else:
        print("Gunakan perintah:")
        print("python automate_Naufal.py --preprocess")
