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
# 1. DOWNLOAD OTOMATIS DARI KAGGLE (VERSI AMAN SERVER)
# ==========================================
@st.cache_resource
def download_dataset_from_kaggle():
    path_anime = "anime.csv"
    zip_target = "anime-recommendations-database.zip"
    
    if not os.path.exists(path_anime):
        # Mengatur kredensial Kaggle
        os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
        os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            # Download langsung ke folder utama
            api.dataset_download_files('CooperUnion/anime-recommendations-database', path='.', unzip=False)
            
            if os.path.exists(zip_target):
                with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                    zip_ref.extractall(".")
                return True
        except Exception as e:
            st.error(f"Gagal mendownload dataset: {e}")
            return False
    return True

# Memuat Data & Membuat Model
@st.cache_data
def load_and_process_data():
    try:
        if os.path.exists("anime.csv"):
            anime = pd.read_csv("anime.csv")
            
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
        else:
            return pd.DataFrame(), None, None
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return pd.DataFrame(), None, None

# ==========================================
# 2. TAMPILAN ANTARMUKA WEB (UI)
# ==========================================
st.title("🎬 Anime Recommendation System")
st.write("Dapatkan 10 rekomendasi anime terbaik berdasarkan kemiripan genre!")

# Pemicu download diletakkan di dalam UI agar tidak menyumbat booting server
if download_dataset_from_kaggle():
    rec_data, sig, rec_indices = load_and_process_data()

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
        st.info("Sedang menyiapkan model data, mohon tunggu sebentar...")
