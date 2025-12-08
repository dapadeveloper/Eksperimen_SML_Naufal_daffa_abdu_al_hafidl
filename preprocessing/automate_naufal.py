import pandas as pd
import argparse
import os

# Paths absolut
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # folder script
RAW_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../namadataset_raw/banknote_authentication.csv"))
PREPROCESS_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
PREPROCESSED_PATH = os.path.join(PREPROCESS_DIR, "banknote_preprocessed.csv")

# Pastikan folder preprocessing ada
os.makedirs(PREPROCESS_DIR, exist_ok=True)

def load_data(path=RAW_DATA_PATH):
    """
    Load dataset dengan aman.
    Dataset Banknote Authentication sudah memiliki header asli,
    jadi kita pakai header=0 agar tidak duplikasi.
    """
    cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(path, header=0, names=cols)
    print(f"Dataset loaded: {path} ({len(df)} rows)")
    print("Class distribution:\n", df['class'].value_counts())
    return df

def preprocessing(df):
    """Preprocessing: hapus duplikat & missing value"""
    df = df.drop_duplicates()

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # Export hasil preprocessing (sesuai requirement Basic)
    df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Preprocessing completed. File saved to: {PREPROCESSED_PATH}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing only")
    args = parser.parse_args()

    if args.preprocess:
        df = load_data()
        preprocessing(df)
    else:
        print("No arguments provided. Use --preprocess.")