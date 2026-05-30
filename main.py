import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st  # <-- Menggunakan Streamlit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

warnings.filterwarnings('ignore')

# Konfigurasi Tampilan Halaman Web Streamlit
st.set_page_config(page_title="Anime Recommender System", page_icon="🎬", layout="centered")

# ==========================================
# 1. DOWNLOAD OTOMATIS DARI KAGGLE
# ==========================================
@st.cache_resource # Mempercepat loading agar tidak download ulang setiap web direfresh
def download_dataset_from_kaggle():
    path_anime = "anime-data/anime.csv"
    path_rating = "anime-data/rating.csv"
    
    if not os.path.exists(path_anime) or not os.path.exists(path_rating):
        with st.spinner("Sedang mengunduh dataset dari Kaggle (Proses ini hanya berjalan sekali)..."):
            # ISI DENGAN API KEY KAGGLE KAMU SEPERTI SEBELUMNYA
            os.environ['KAGGLE_USERNAME'] = "username_kaggle_kamu" 
            os.environ['KAGGLE_KEY'] = "api_key_kaggle_kamu"       
            
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                api.dataset_download_files('CooperUnion/anime-recommendations-database', path='anime-data', unzip=True)
            except Exception as e:
                st.error(f"Gagal mendownload dataset: {e}")

download_dataset_from_kaggle()

# Memuat Data & Membuat Model (Menggunakan Cache agar Web Ringan & Cepat)
@st.cache_data
def load_and_process_data():
    try:
        anime = pd.read_csv("anime-data/anime.csv")
        rating = pd.read_csv("anime-data/rating.csv")
        
        fulldata = pd.merge(anime, rating, on="anime_id", suffixes=[None, "_user"])
        fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

        rec_data = fulldata.copy()
        rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
        rec_data.reset_index(drop=True, inplace=True)
        rec_data["genre"] = rec_data["genre"].fillna("")

        genres = rec_data["genre"].str.split(", |, |,").astype(str)

        tfv = TfidfVectorizer(
            min_df=3, max_features=None, strip_accents="unicode", 
            analyzer="word", token_pattern=r"\w{1,}", 
            ngram_range=(1, 3), stop_words="english"
        )
        tfv_matrix = tfv.fit_transform(genres)
        sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
        rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()
        
        return anime, rec_data, sig, rec_indices
    except:
        return pd.DataFrame(), pd.DataFrame(), None, None

anime, rec_data, sig, rec_indices = load_and_process_data()

# ==========================================
# 2. TAMPILAN ANTARMUKA WEB (UI)
# ==========================================
st.title("🎬 Anime Recommendation System")
st.write("Dapatkan 10 rekomendasi anime terbaik berdasarkan kemiripan genre!")

if sig is not None:
    # Menggunakan Selectbox agar pengguna tidak salah ketik nama anime
    list_anime = rec_data['name'].unique()
    search_query = st.selectbox("Pilih atau ketik nama anime yang kamu sukai:", list_anime)

    if st.button("Cari Rekomendasi"):
        idx = rec_indices[search_query]
        sig_score = list(enumerate(sig[idx]))
        sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
        sig_score = sig_score[1:11]
        anime_indices = [i[0] for i in sig_score]
        
        rec_dic = {
            "No": range(1, 11),
            "Judul Anime": anime["name"].iloc[anime_indices].values,
            "Rating Global": anime["rating"].iloc[anime_indices].values
        }
        dataframe = pd.DataFrame(data=rec_dic)
        dataframe.set_index("No", inplace=True)
        
        st.success(f"Berikut adalah 10 rekomendasi anime bagi penonton **{search_query}**:")
        # Menampilkan tabel interaktif di web
        st.dataframe(dataframe, use_container_width=True)
else:
    st.error("Gagal memuat model. Pastikan dataset terunduh dengan benar.")