import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

warnings.filterwarnings('ignore')

# ==========================================
# 1. DOWNLOAD AUTOMATIS DARI KAGGLE
# ==========================================
def download_dataset_from_kaggle():
    path_anime = "anime-data/anime.csv"
    path_rating = "anime-data/rating.csv"
    
    # Jika file csv belum ada, lakukan download otomatis
    if not os.path.exists(path_anime) or not os.path.exists(path_rating):
        print("Dataset tidak ditemukan secara lokal. Memulai download dari Kaggle...")
        
        # Mengatur kredensial Kaggle secara langsung via code (Sangat berguna untuk Streamlit)
        os.environ['KAGGLE_USERNAME'] = "username_kaggle_kamu" # <-- GANTI INI
        os.environ['KAGGLE_KEY'] = "api_key_kaggle_kamu"       # <-- Ganti INI
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download zip dataset dari Kaggle ke folder anime-data
            print("Downloading files...")
            api.dataset_download_files('CooperUnion/anime-recommendations-database', path='anime-data', unzip=True)
            print("Download dan Ekstrak selesai!")
            
        except Exception as e:
            print(f"Gagal mendownload dataset: {e}")
            print("Pastikan Username dan API Key Kaggle kamu sudah benar.")

# Jalankan fungsi download di awal program
download_dataset_from_kaggle()

# Memuat Data setelah dipastikan ter-download
try:
    anime = pd.read_csv("anime-data/anime.csv")
    rating = pd.read_csv("anime-data/rating.csv")
    print("Dataset berhasil dimuat ke dalam program!")
except FileNotFoundError:
    print("Error: Program tidak dapat melanjutkan karena dataset tidak ada.")
    anime = pd.DataFrame()
    rating = pd.DataFrame()

# ==========================================
# 2. PRA-PEMROSESAN & MODELLING
# ==========================================
sig = None
rec_indices = None
rec_data = pd.DataFrame()

if not anime.empty and not rating.empty:
    fulldata = pd.merge(anime, rating, on="anime_id", suffixes=[None, "_user"])
    fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

    rec_data = fulldata.copy()
    rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
    rec_data.reset_index(drop=True, inplace=True)

    # -------------------------------------------------------------
    # PERBAIKAN: Mengisi genre yang kosong (NaN) dengan string kosong ''
    # -------------------------------------------------------------
    rec_data["genre"] = rec_data["genre"].fillna("")

    genres = rec_data["genre"].str.split(", |, |,").astype(str)

    tfv = TfidfVectorizer(
        min_df=3, 
        max_features=None, 
        strip_accents="unicode", 
        analyzer="word", 
        token_pattern=r"\w{1,}", 
        ngram_range=(1, 3), 
        stop_words="english"
    )
    tfv_matrix = tfv.fit_transform(genres)

    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()
# ==========================================
# 3. FUNGSI REKOMENDASI
# ==========================================
def give_recommendation(search_query, sig=sig):
    if sig is None or rec_data.empty:
        print("Error: Model tidak siap karena data kosong.")
        return None
        
    matching_titles = rec_data[rec_data['name'].str.contains(search_query, case=False, na=False)]['name'].unique()
    
    if len(matching_titles) == 0:
        print(f"No anime found matching '{search_query}'.")
        return None
    elif len(matching_titles) > 1:
        print(f"Multiple anime found matching '{search_query}':")
        for title in matching_titles[:10]:
            print(f"- {title}")
        return None
    else:
        title = matching_titles[0]
        idx = rec_indices[title]
        sig_score = list(enumerate(sig[idx]))
        sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
        sig_score = sig_score[1:11]
        anime_indices = [i[0] for i in sig_score]
        
        rec_dic = {
            "No": range(1, 11),
            "Anime Name": anime["name"].iloc[anime_indices].values,
            "Rating": anime["rating"].iloc[anime_indices].values
        }
        dataframe = pd.DataFrame(data=rec_dic)
        dataframe.set_index("No", inplace=True)
        return dataframe

if __name__ == "__main__":
    query = "boku"  # Contoh query, bisa diganti dengan input dari pengguna
    hasil = give_recommendation(query)
    if hasil is not None:
        print(f"\nRekomendasi untuk penonton {query}:\n")
        print(hasil)
        