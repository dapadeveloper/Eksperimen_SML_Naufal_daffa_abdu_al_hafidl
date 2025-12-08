import pandas as pd
import argparse
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

RAW_PATH = os.path.join(BASE_DIR, "../namadataset_raw/banknote_authentication.csv")
PREP_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
PREP_PATH = os.path.join(PREP_DIR, "banknote_preprocessed.csv")

os.makedirs(PREP_DIR, exist_ok=True)

def load_raw():
    cols = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    df = pd.read_csv(RAW_PATH, header=0, names=cols)
    print(f"[INFO] Loaded raw dataset → {df.shape}")
    return df

def preprocess(df):
    df = df.drop_duplicates()
    df = df.dropna()

    df.to_csv(PREP_PATH, index=False)
    print(f"[INFO] Preprocessed dataset saved → {PREP_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run preprocessing")
    args = parser.parse_args()

    if args.run:
        df = load_raw()
        preprocess(df)
    else:
        print("Usage: python automate_naufal.py --run")
