import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ------------------------------------------------------------
# 1. File input dan output
# ------------------------------------------------------------

raw_file = r"C:\Users\USER\Documents\maggang\SML2_Naufaldaffa\Eksperimen_SML_Naufal_daffa_abdu_al_hafidl\namadataset_raw\banknote_authentication.csv"
output_folder = r"C:\Users\USER\Documents\maggang\SML2_Naufaldaffa\Eksperimen_SML_Naufal_daffa_abdu_al_hafidl\preprocessing\namadataset_preprocessing"
output_file = os.path.join(output_folder, "banknote_preprocessed.csv")

# Buat folder preprocessing jika belum ada
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------------------------
# 2. Load Dataset
# ------------------------------------------------------------

# Jika file raw belum ada, download otomatis
if not os.path.exists(raw_file):
    print("[INFO] File raw belum ada. Download dataset dari UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(url, header=None, names=columns)
    df.to_csv(raw_file, index=False)
    print("[INFO] Dataset berhasil disimpan ke:", raw_file)
else:
    df = pd.read_csv(raw_file)

print("[INFO] Dataset berhasil dibaca, 5 baris pertama:")
print(df.head())

# ------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------

# Hapus duplikasi
df = df.drop_duplicates()

# Tangani missing values (jika ada)
if df.isna().sum().sum() > 0:
    df = df.fillna(df.mean())

# Semua kolom numerik kecuali target
num_cols = ["variance", "skewness", "curtosis", "entropy"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------------------------------------
# 4. Simpan hasil preprocessing
# ------------------------------------------------------------

df.to_csv(output_file, index=False)
print(f"[INFO] Hasil preprocessing disimpan di: {output_file}")