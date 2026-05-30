import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Anime Recommender System", page_icon="🎬", layout="centered")

# ==========================================
# 1. DOWNLOAD OTOMATIS DARI KAGGLE
# ==========================================
@st.cache_resource
def download_dataset_from_kaggle():
    path_anime = "anime-data/anime.csv"
    
    if not os.path.exists(path_anime):
        with st.spinner("Sedang mengunduh dataset dari Kaggle (Proses ini hanya berjalan sekali)..."):
            os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
            os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
            
            try:
                os.makedirs("anime-data", exist_ok=True)
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                # Mendownload paket dataset seutuhnya
                api.dataset_download_files('CooperUnion/anime-recommendations-database', path='anime-data', unzip=True)
            except Exception as e:
                st.error(f"Gagal mendownload dataset: {e}")

download_dataset_from_kaggle()

# Memuat Data & Membuat Model (Sangat Hemat RAM)
@st.cache_data
def load_and_process_data():
    try:
        # HANYA MEMBACA anime.csv (Jangan rating.csv agar RAM 1GB Streamlit tidak jebol!)
        anime = pd.read_csv("anime-data/anime.csv")
        
        rec_data = anime.dropna(subset=['name']).copy()
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
        
        return rec_data, sig, rec_indices
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return pd.DataFrame(), None, None

rec_data, sig, rec_indices = load_and_process_data()

# ==========================================
# 2. TAMPILAN ANTARMUKA WEB (UI)
# ==========================================
st.title("🎬 Anime Recommendation System")
st.write("Dapatkan 10 rekomendasi anime terbaik berdasarkan kemiripan genre!")

if sig is not None and not rec_data.empty:
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
            "Judul Anime": rec_data["name"].iloc[anime_indices].values,
            "Genre": rec_data["genre"].iloc[anime_indices].values,
            "Rating": rec_data["rating"].iloc[anime_indices].values
        }
        dataframe = pd.DataFrame(data=rec_dic)
        dataframe.set_index("No", inplace=True)
        
        st.success(f"Berikut adalah 10 rekomendasi anime bagi penonton **{search_query}**:")
        st.dataframe(dataframe, use_container_width=True)
else:
    st.error("Gagal memuat model atau dataset belum siap.")