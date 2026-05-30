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
# 1. PENANGANAN DATASET (KAGGLE / LOKAL)
# ==========================================
# Catatan: Di Streamlit/VS Code lokal, sebaiknya dataset sudah di-ekstrak 
# di dalam folder proyek kamu agar tidak perlu download setiap saat.

zip_path = 'anime-data/anime-recommendations-database.zip'
extract_path = 'anime-data/'

# Pengecekan otomatis jika file belum di-ekstrak
if not os.path.exists(os.path.join(extract_path, 'anime.csv')):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset berhasil diekstrak!")
    else:
        print("Peringatan: File zip dataset tidak ditemukan di folder 'anime-data/'.")

# Memuat Data (Path disesuaikan ke folder lokal)
try:
    anime = pd.read_csv("anime-data/anime.csv")
    rating = pd.read_csv("anime-data/rating.csv")
except FileNotFoundError:
    print("Error: File anime.csv atau rating.csv tidak ditemukan. Pastikan folder 'anime-data' sudah ada.")
    anime = pd.DataFrame()
    rating = pd.DataFrame()

# ==========================================
# 2. PRA-PEMROSESAN & MODELLING
# ==========================================
if not anime.empty and not rating.empty:
    # Menggabungkan Dataset
    fulldata = pd.merge(anime, rating, on="anime_id", suffixes=[None, "_user"])
    fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

    # Ekstrak genre unik untuk TF-IDF
    rec_data = fulldata.copy()
    rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
    rec_data.reset_index(drop=True, inplace=True)

    genres = rec_data["genre"].str.split(", |, |,").astype(str)

    # TF-IDF Vectorization
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

    # Membaca Sigmoid Kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()

# ==========================================
# 3. FUNGSI REKOMENDASI (VERSI TERMINAL/VS CODE)
# ==========================================
def give_recommendation(search_query, sig=sig):
    matching_titles = rec_data[rec_data['name'].str.contains(search_query, case=False, na=False)]['name'].unique()
    
    if len(matching_titles) == 0:
        print(f"No anime found matching '{search_query}'. Please try a different search term.")
        return None
    elif len(matching_titles) > 1:
        print(f"Multiple anime found matching '{search_query}':")
        for title in matching_titles[:10]: # Limbat 10 pilihan agar tidak penuh
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

# ==========================================
# 4. CARA MENJALANKAN DI VS CODE
# ==========================================
if __name__ == "__main__":
    # Ganti dengan nama anime yang ingin kamu uji di terminal VS Code
    query = "Cyborg 009" 
    print(f"Mencari rekomendasi untuk: {query}\n")
    
    hasil = give_recommendation(query)
    if hasil is not None:
        print(hasil)