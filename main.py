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
# 1. EKSTRAK DATASET LOKAL DARI GITHUB
# ==========================================
@st.cache_resource
def extract_local_dataset():
    path_anime = "anime.csv"
    zip_target = "anime.zip"
    
    # Jika file csv belum ada tapi zip-nya ada, langsung ekstrak
    if not os.path.exists(path_anime) and os.path.exists(zip_target):
        try:
            with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Ekstrak file anime.zip lokal berhasil!")
        except Exception as e:
            st.error(f"Gagal mengekstrak file lokal: {e}")

# Jalankan ekstraksi file zip yang ada di reponmu
extract_local_dataset()

# ==========================================
# 2. MEMUAT DATA & MEMBUAT MODEL (OPTIMAL & HEMAT RAM)
# ==========================================
@st.cache_data
def load_and_process_data():
    try:
        if os.path.exists("anime.csv"):
            # 1. Optimasi Tipe Data saat membaca CSV untuk menghemat RAM hingga 50%
            anime_dtypes = {
                'anime_id': 'int32', 
                'rating': 'float32', 
                'members': 'int32'
            }
            anime = pd.read_csv("anime.csv", dtype=anime_dtypes)
            
            # 2. Pembersihan awal bawaan kode Anda
            rec_data = anime.dropna(subset=['name']).copy()
            rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
            rec_data.reset_index(drop=True, inplace=True)
            
            # 3. BATASI DATASET (Crucial untuk Streamlit Cloud agar tidak Out of Memory)
            # Mengambil 5.000 anime terpopuler berdasarkan jumlah members
            if len(rec_data) > 5000:
                rec_data = rec_data.sort_values(by='members', ascending=False).head(5000)
                rec_data.reset_index(drop=True, inplace=True)
            
            rec_data["genre"] = rec_data["genre"].fillna("")
            genres = rec_data["genre"].str.split(", |, |,").astype(str)
            
            # 4. Batasi max_features pada TF-IDF agar ukuran matriks terkontrol
            tfv = TfidfVectorizer(
                min_df=3, 
                max_features=3000,  # Membatasi maksimal 3.000 kosakata unik
                strip_accents="unicode", 
                analyzer="word", 
                token_pattern=r"\w{1,}", 
                ngram_range=(1, 3), 
                stop_words="english"
            )
            
            tfv_matrix = tfv.fit_transform(genres)
            
            # Komputasi sigmoid kernel pada data yang sudah dibatasi (aman dari crash)
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
            
            rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()
            
            return rec_data, sig, rec_indices
        else:
            st.error("File anime.csv tidak ditemukan setelah ekstraksi!")
            return pd.DataFrame(), None, None
            
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return pd.DataFrame(), None, None

# ==========================================
# 3. TAMPILAN ANTARMUKA WEB (UI)
# ==========================================
st.title("🎬 Anime Recommendation System")
st.write("Dapatkan 10 rekomendasi anime terbaik berdasarkan kemiripan genre!")

# Memuat data hasil optimasi
rec_data, sig, rec_indices = load_and_process_data()

if sig is not None and not rec_data.empty:
    list_anime = rec_data['name'].unique()
    search_query = st.selectbox("Pilih atau ketik nama anime yang kamu sukai:", list_anime)
    
    # Inisialisasi state untuk menyimpan genre yang dipilih via tombol
    if "selected_genre" not in st.session_state:
        st.session_state.selected_genre = None

    if st.button("Cari Rekomendasi"):
        # Reset filter genre setiap kali mencari anime baru
        st.session_state.selected_genre = None
        
        idx = rec_indices[search_query]
        
        # Hitung skor kemiripan
        sig_score = list(enumerate(sig[idx]))
        sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
        sig_score = sig_score[1:11]  # Ambil peringkat 2 sampai 11
        
        anime_indices = [i[0] for i in sig_score]
        
        # Menyusun DataFrame hasil untuk ditampilkan
        rec_dic = {
            "No": range(1, 11),
            "Judul Anime": rec_data["name"].iloc[anime_indices].values,
            "Genre": rec_data["genre"].iloc[anime_indices].values,
            "Rating": rec_data["rating"].iloc[anime_indices].values
        }
        
        # Simpan hasil rekomendasi ke session state agar tidak hilang saat tombol genre diklik
        st.session_state.df_result = pd.DataFrame(data=rec_dic).set_index("No")
        st.session_state.search_done = True
        st.session_state.current_anime = search_query

    # Jika pencarian sudah dilakukan, tampilkan hasil dan menu eksplorasi genre
    if st.session_state.get("search_done", False):
        st.success(f"Berikut adalah 10 rekomendasi anime bagi penonton **{st.session_state.current_anime}**:")
        st.dataframe(st.session_state.df_result, use_container_width=True)
        
        st.write("---")
        st.subheader("🔍 Eksplorasi Genre Lebih Lanjut")
        st.write("Klik salah satu genre di bawah ini untuk melihat anime sejenis dari daftar rekomendasi:")
        
        # Ambil semua genre unik dari 10 anime hasil rekomendasi tersebut
        all_genres = set()
        for g_str in st.session_state.df_result["Genre"]:
            genres_list = [g.strip() for g in g_str.split(",")]
            all_genres.update(genres_list)
        
        # Urutkan nama genre secara alfabetis
        sorted_genres = sorted(list(all_genres))
        
        # Membuat tombol pill horizontal yang bisa diklik
        genre_click = st.pills("Pilih Genre:", sorted_genres, selection_mode="single")
        
        # Jika ada genre yang diklik, lakukan filter dari seluruh dataset rec_data
        if genre_click:
            st.session_state.selected_genre = genre_click
            
            # Filter rec_data (5000 data terpopuler) yang mengandung genre terpilih
            filtered_anime = rec_data[rec_data['genre'].str.contains(genre_click, case=False, na=False)]
            
            st.write(f"### 📋 Daftar Anime dengan Genre: **{genre_click}**")
            
            # Format tampilan tabel filter
            filter_display = filtered_anime[['name', 'genre', 'rating']].copy()
            filter_display.columns = ['Judul Anime', 'Genre', 'Rating']
            filter_display.insert(0, 'No', range(1, len(filter_display) + 1))
            filter_display.set_index('No', inplace=True)
            
            # Batasi tampilan maksimal 15 anime terpopuler di genre tersebut agar rapi
            st.dataframe(filter_display.head(15), use_container_width=True)

else:
    st.info("Sedang memuat data, silakan tunggu sebentar...")
