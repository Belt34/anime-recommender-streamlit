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
            anime_dtypes = {'anime_id': 'int32', 'rating': 'float32', 'members': 'int32'}
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

# Memuat data hasil optimasi
rec_data, sig, rec_indices = load_and_process_data()

# ==========================================
# 3. TAMPILAN SIDEBAR (TAB SLIDE DARI KIRI)
# ==========================================
with st.sidebar:
    st.title("⛩️ Anime Recommendation")
    st.write("---")
    st.markdown("### Navigation")
    
    # Menu Navigasi sesuai request kamu
    menu = st.radio(
        label="Navigation Menu",
        options=[
            "Home", 
            "EDA", 
            "Description Page", 
            "Data Preprocessing", 
            "Train Model", 
            "Result / Prediction Demo", 
            "Feature Importance", 
            "About Us"
        ],
        label_visibility="collapsed" # Menyembunyikan label bawaan agar rapi
    )
    
    st.write("---")
    st.markdown("**MyAnimeList Dataset**\n\n*Anime Recommendation Engine*\n\n---\nBuilt with ❤️ using Streamlit")

# ==========================================
# 4. LOGIKA HALAMAN / KONTEN UTAMA
# ==========================================
if menu == "Home":
    st.title("🏠 Home")
    st.write("Selamat datang di aplikasi Anime Recommendation System!")
    st.write("Silakan pilih menu di sidebar sebelah kiri untuk menjelajahi aplikasi.")

elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Halaman ini nantinya bisa kamu gunakan untuk menampilkan grafik visualisasi data anime.")

elif menu == "Description Page":
    st.title("📝 Description Page")
    st.write("Penjelasan detail mengenai proyek, dataset, dan sistem rekomendasi yang dibangun.")

elif menu == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing")
    st.write("Proses pembersihan data, penanganan nilai kosong, dan penyiapan teks fitur.")

elif menu == "Train Model":
    st.title("🧠 Train Model")
    st.write("Proses pembentukan matriks TF-IDF dan kalkulasi skor kemiripan menggunakan Sigmoid Kernel.")

elif menu == "Result / Prediction Demo":
    # --- SELURUH KODE ASLI KAMU MASUK KE SINI ---
    st.title("🎬 Anime Recommendation System")
    st.write("Dapatkan rekomendasi anime terbaik berdasarkan kemiripan genre!")
    
    if sig is not None and not rec_data.empty:
        list_anime = rec_data['name'].unique()
        search_query = st.selectbox("Pilih atau ketik nama anime yang kamu sukai:", list_anime)
        
        # Inisialisasi semua session state yang diperlukan
        if "selected_genre" not in st.session_state:
            st.session_state.selected_genre = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1  # Melacak halaman aktif
            
        # Tombol Utama: Cari Rekomendasi Awal
        if st.button("🔮 Cari Rekomendasi", use_container_width=True):
            st.session_state.current_page = 1  # Reset ke halaman 1 saat cari anime baru
            st.session_state.selected_genre = None
            st.session_state.search_done = True
            st.session_state.current_anime = search_query
            
        # Logika eksekusi & navigasi rekomendasi
        if st.session_state.get("search_done", False):
            active_anime = st.session_state.current_anime
            idx = rec_indices[active_anime]
            
            # Hitung seluruh skor kemiripan
            sig_score = list(enumerate(sig[idx]))
            sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
            
            # --- SISTEM NAVIGASI PAGINASI (< Halaman >) ---
            st.write("")
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            
            with nav_col1:
                if st.session_state.current_page > 1:
                    if st.button("⬅️ `<`", use_container_width=True):
                        st.session_state.current_page -= 1
                        st.session_state.selected_genre = None
                        st.rerun()
                else:
                    st.button("⬅️ `<`", disabled=True, use_container_width=True)
                    
            with nav_col2:
                current_start = ((st.session_state.current_page - 1) * 10) + 1
                current_end = st.session_state.current_page * 10
                st.markdown(f"<h5 style='text-align: center; margin-top: 5px;'>Halaman {st.session_state.current_page} (Peringkat {current_start} - {current_end})</h5>", unsafe_allow_html=True)
                
            with nav_col3:
                max_pilihan_anime = len(sig_score) - 1
                max_halaman = min(5, max_pilihan_anime // 10)
                if st.session_state.current_page < max_halaman:
                    if st.button("`>` ➡️", use_container_width=True):
                        st.session_state.current_page += 1
                        st.session_state.selected_genre = None
                        st.rerun()
                else:
                    st.button("`>` ➡️", disabled=True, use_container_width=True)
            
            # Menghitung slicing indeks berdasarkan halaman aktif
            start_rank = ((st.session_state.current_page - 1) * 10) + 1
            end_rank = (st.session_state.current_page * 10) + 1
            nomor_urut = range(start_rank, end_rank)
            
            page_sig_score = sig_score[start_rank:end_rank]
            anime_indices = [i[0] for i in page_sig_score]
            
            # Menyusun DataFrame hasil untuk ditampilkan
            rec_dic = {
                "No": nomor_urut, 
                "Judul Anime": rec_data["name"].iloc[anime_indices].values, 
                "Genre": rec_data["genre"].iloc[anime_indices].values, 
                "Rating": rec_data["rating"].iloc[anime_indices].values
            }
            st.session_state.df_result = pd.DataFrame(data=rec_dic).set_index("No")
            
            # Tampilkan Pesan Sukses & Tabel Utama
            st.success(f"Menampilkan 10 rekomendasi anime sejenis untuk **{active_anime}**:")
            st.dataframe(
                st.session_state.df_result, 
                use_container_width=True, 
                column_config={
                    "Judul Anime": st.column_config.TextColumn("Judul Anime", width="large"), 
                    "Genre": st.column_config.TextColumn("Genre", width="large"), 
                    "Rating": st.column_config.NumberColumn("Rating", width="small")
                }
            )
            
            # --- Bagian Eksplorasi Genre ---
            st.write("---")
            st.subheader("🔍 Pt. 2: Eksplorasi Genre Lebih Lanjut")
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
                        "Rating": st.column_config.NumberColumn("Rating", width="small")
                    }
                )
    else:
        st.info("Sedang memuat data, silakan tunggu sebentar...")

elif menu == "Feature Importance":
    st.title("📊 Feature Importance")
    st.write("Menampilkan bobot fitur kata (N-Gram) dari genre yang paling memengaruhi model rekomendasi.")

elif menu == "About Us":
    st.title("👥 About Us")
    st.write("Aplikasi ini dikembangkan sebagai sistem rekomendasi berbasis konten menggunakan TF-IDF dan Sigmoid Kernel.")
