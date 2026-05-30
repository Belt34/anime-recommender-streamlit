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
st.write("Dapatkan rekomendasi anime terbaik berdasarkan kemiripan genre!")

# Memuat data hasil optimasi
rec_data, sig, rec_indices = load_and_process_data()

if sig is not None and not rec_data.empty:
    list_anime = rec_data['name'].unique()
    search_query = st.selectbox("Pilih atau ketik nama anime yang kamu sukai:", list_anime)
    
    # Inisialisasi semua session state yang diperlukan
    if "selected_genre" not in st.session_state:
        st.session_state.selected_genre = None
    if "recommendation_page" not in st.session_state:
        st.session_state.recommendation_page = 1  # Halaman 1 = peringkat 1-10, Halaman 2 = peringkat 11-20

    # Membuat dua kolom berdampingan untuk tombol utama
    col1, col2 = st.columns([1, 1])

    # Tombol 1: Cari Rekomendasi Awal (Reset ke halaman 1)
    with col1:
        if st.button("🔮 Cari Rekomendasi", use_container_width=True):
            st.session_state.recommendation_page = 1
            st.session_state.selected_genre = None
            st.session_state.search_done = True
            st.session_state.current_anime = search_query

    # Tombol 2: Tampilkan 10 Anime Berbeda Lainnya (Pindah ke halaman 2)
    with col2:
        # Tombol ini hanya aktif/muncul jika user sudah pernah menekan tombol pertama sebelumnya
        if st.session_state.get("search_done", False) and search_query == st.session_state.get("current_anime"):
            tombol_teks = "🔄 Tampilkan 10 Lainnya" if st.session_state.recommendation_page == 1 else "↩️ Kembali ke 10 Awal"
            if st.button(tombol_teks, use_container_width=True):
                # Tukar halaman antara 1 dan 2
                st.session_state.recommendation_page = 2 if st.session_state.recommendation_page == 1 else 1
                st.session_state.selected_genre = None # Reset filter genre saat ganti batch

    # Logika eksekusi rekomendasi berdasarkan halaman aktif
    if st.session_state.get("search_done", False):
        # Selalu hitung skor kemiripan dari anime aktif
        active_anime = st.session_state.current_anime
        idx = rec_indices[active_anime]
        
        sig_score = list(enumerate(sig[idx]))
        sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
        
        # Tentukan slicing index berdasarkan halaman
        if st.session_state.recommendation_page == 1:
            start_rank, end_rank = 1, 11   # Peringkat 1 sampai 10 teratas
            nomor_urut = range(1, 11)
            label_sukses = f"Berikut adalah 10 rekomendasi anime teratas bagi penonton **{active_anime}**:"
        else:
            start_rank, end_rank = 11, 21  # Peringkat 11 sampai 20 (10 anime berbeda lainnya)
            nomor_urut = range(11, 21)
            label_sukses = f"Berikut adalah 10 rekomendasi alternatif berikutnya (Peringkat 11-20) bagi penonton **{active_anime}**:"
            
        sig_score = sig_score[start_rank:end_rank]
        anime_indices = [i[0] for i in sig_score]
        
        # Menyusun DataFrame hasil untuk ditampilkan
        rec_dic = {
            "No": nomor_urut,
            "Judul Anime": rec_data["name"].iloc[anime_indices].values,
            "Genre": rec_data["genre"].iloc[anime_indices].values,
            "Rating": rec_data["rating"].iloc[anime_indices].values
        }
        
        st.session_state.df_result = pd.DataFrame(data=rec_dic).set_index("No")
        
        # Tampilkan Pesan Sukses & Tabel Utama
        st.success(label_sukses)
        st.dataframe(
            st.session_state.df_result, 
            use_container_width=True,
            column_config={
                "Judul Anime": st.column_config.TextColumn("Judul Anime", width="large"),
                "Genre": st.column_config.TextColumn("Genre", width="large"),
                "Rating": st.column_config.NumberColumn("Rating", width="small")
            }
        )
        
        # --- Bagian Eksplorasi Genre (Tetap Sinkron Otomatis dengan Batch yang Aktif) ---
        st.write("---")
        st.subheader("🔍 Eksplorasi Genre Lebih Lanjut")
        st.write("Klik salah satu genre di bawah ini untuk melihat anime sejenis dari daftar di atas:")
        
        all_genres = set()
        for g_str in st.session_state.df_result["Genre"]:
            genres_list = [g.strip() for g in g_str.split(",")]
            all_genres.update(genres_list)
        
        sorted_genres = sorted(list(all_genres))
        genre_click = st.pills("Pilih Genre:", sorted_genres, selection_mode="single")
        
        if genre_click:
            st.session_state.selected_genre = genre_click
            filtered_anime = rec_data[rec_data['genre'].str.contains(genre_click, case=False, na=False)]
            
            st.write(f"### 📋 Daftar Anime dengan Genre: **{genre_click}**")
            
            filter_display = filtered_anime[['name', 'genre', 'rating']].copy()
            filter_display.columns = ['Judul Anime', 'Genre', 'Rating']
            filter_display.insert(0, 'No', range(1, len(filter_display) + 1))
            filter_display.set_index('No', inplace=True)
            
            st.dataframe(
                filter_display.head(15), 
                use_container_width=True,
                column_config={
                    "Judul Anime": st.column_config.TextColumn("Judul Anime", width="large"),
                    "Genre": st.column_config.TextColumn("Genre", width="large"),
