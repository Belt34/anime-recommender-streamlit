import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Anime Recommender System", page_icon="🎬", layout="centered")

# Memuat Data Langsung Menggunakan URL Jalur Pipa Instan
@st.cache_data
def load_and_process_data():
    try:
        # Menembak langsung file anime.csv publik yang super stabil lewat internet
      url_anime = "https://raw.githubusercontent.com/Mayank-Tyagi/Anime-Recommendation-System/master/anime.csv"
        
        with st.spinner("Sedang menghubungkan ke basis data anime..."):
            anime = pd.read_csv(url_anime)
        
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

# ==========================================
# TAMPILAN ANTARMUKA WEB (UI)
# ==========================================
st.title("🎬 Anime Recommendation System")
st.write("Dapatkan 10 rekomendasi anime terbaik berdasarkan kemiripan genre!")

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
