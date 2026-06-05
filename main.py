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

# --- SAKURA MOCHI THEME CUSTOMIZATION ---
st.markdown("""
    <style>
        /* 1. Mengubah latar belakang utama menjadi Putih Mochi */
        .stApp {
            background-color: #FFFFFF;
        }
        
        /* 2. Mengubah warna judul utama (H1) menjadi Pink Sakura */
        h1 {
            color: #FFB7B2 !important;
            font-weight: 700;
        }
        
        /* 3. Mengubah warna sub-judul/deskripsi langsung di bawah judul menjadi Hijau Matcha */
        h1 + p, .stMarkdown p:first-of-type {
            color: #A8E6CF !important;
            font-size: 1.1rem;
        }
        
        /* 4. Aksen Coklat Kayu untuk Sub-header (H2, H3) dan teks umum */
        h2, h3, h4, h5, h6 {
            color: #6F4E37 !important;
        }
        
        div[data-testid="stMarkdownContainer"] p {
            color: #5C4033; /* Coklat kayu gelap agar teks deskripsi tetap kontras dan terbaca */
        }
        
        /* 5. Kostumisasi Sidebar (Opsional: dibuat senada dengan tema) */
        [data-testid="stSidebar"] {
            background-color: #F9F1F0; /* Putih sedikit semburat pink lembut */
            border-right: 2px solid #6F4E37; /* Garis pembatas coklat kayu */
        }
        
        [data-testid="stSidebar"] * {
            color: #6F4E37 !important; /* Semua teks di sidebar berwarna coklat kayu */
        }
        
        /* 6. Mengubah warna komponen interaktif seperti Radio Button / Slider */
        .st-bd, .st-ae, .st-af {
            color: #FFB7B2 !important;
        }
    </style>
""", unsafe_allow_html=True)
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
    
    # Sambutan Utama
    st.markdown("""
    ### Selamat datang di aplikasi **Anime Recommendation System**! 👋
    
    Sistem rekomendasi ini dirancang interaktif untuk membantu para penggemar menemukan tontonan baru yang relevan secara instan tanpa harus kebingungan memilih di antara ribuan judul.
    """)
    
    # Kotak Deskripsi Singkat Proyek
    st.success("""
    💡 **Deskripsi Singkat Proyek:**
    
    Aplikasi ini menerapkan pilar *Unsupervised Learning* melalui metode **Content-Based Filtering**. Dengan mengekstrak karakteristik teks fitur dari komponen `genre`, model secara cerdas mengukur kedekatan antar-anime menggunakan pendekatan **TF-IDF Vectorizer** dan **Sigmoid Kernel Similarity**. 
    
    Hasil akhirnya adalah mesin pencari cerdas yang mampu merekomendasikan daftar anime masa depan yang paling mendekati preferensi dan kemiripan corak cerita dari anime yang kamu sukai.
    """)
    
    st.write("---")
    
    # Petunjuk Navigasi (Membimbing user untuk mencoba menu lain)
    st.markdown("""
    #### 🚀 Cara Menjelajahi Aplikasi:
    Silakan gunakan menu **Navigation** di sidebar sebelah kiri untuk mengakses seluruh rangkaian eksperimen data kami:
    1. **EDA:** Melihat visualisasi dan pola penyebaran data mentah.
    2. **Description Page & Preprocessing:** Memahami struktur data awal dan proses pembersihannya.
    3. **Train Model & Feature Importance:** Mengintip konfigurasi model dan bobot genre terpenting.
    4. **Result / Prediction Demo:** Mencoba langsung mesin rekomendasi anime secara *live*!
    5. **About Us:** Informasi profil lengkap tim **Group 6 (Binus University)**.
    """)
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Jelajahi karakteristik data dari katalog anime secara interaktif.")
    
    if rec_data.empty:
        st.warning("Data anime belum dimuat. Silakan periksa file dataset kamu.")
    else:
        # Buat sub-menu tab di dalam halaman EDA agar rapi seperti aplikasi energi
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔢 Statistik Deskriptif", 
            "📈 Distribusi Fitur", 
            "🔥 Korelasi & Hubungan", 
            "🎭 Wawasan Genre"
        ])
        
        # --- TAB 1: STATISTIK DESKRIPTIF ---
        with tab1:
            st.subheader("1. Ringkasan Statistik Data")
            st.write("Berikut adalah gambaran umum angka mentah dari dataset anime yang telah difilter:")
            st.dataframe(rec_data.describe(), use_container_width=True)
            
            # Ringkasan Cepat (Insights Card)
            st.write("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anime di Analisis", f"{len(rec_data):,}")
            with col2:
                st.metric("Rata-rata Rating Global", f"{rec_data['rating'].mean():.2f} / 10")
            with col3:
                st.metric("Total Komunitas Terbesar", f"{rec_data['members'].max():,}")

        # --- TAB 2: DISTRIBUSI FITUR (HISTOGRAM & BOXPLOT) ---
        with tab2:
            st.subheader("2. Generator Distribusi Data")
            feature_choice = st.selectbox("Pilih fitur yang ingin dilihat distribusinya:", ["rating", "members"])
            bins_choice = st.slider("Jumlah Bins (Batang Histogram):", min_value=5, max_value=50, value=20)
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot Histogram & KDE
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(rec_data[feature_choice], bins=bins_choice, kde=True, ax=ax, color="#FF4B4B")
            ax.set_title(f"Grafik Distribusi Fitur: {feature_choice.capitalize()}")
            ax.set_xlabel(feature_choice.capitalize())
            ax.set_ylabel("Jumlah Anime")
            st.pyplot(fig)
            
            # Plot Boxplot untuk melihat Outliers
            st.write("---")
            st.write(f"**Boxplot Fitur: {feature_choice.capitalize()} (Deteksi Outliers)**")
            fig_box, ax_box = plt.subplots(figsize=(8, 2))
            sns.boxplot(x=rec_data[feature_choice], ax=ax_box, color="#4682B4")
            st.pyplot(fig_box)

        # --- TAB 3: KORELASI & HUBUNGAN DUA VARIABEL ---
        with tab3:
            st.subheader("3. Analisis Hubungan Antar Fitur")
            st.write("Apakah anime yang populer (banyak members) otomatis memiliki rating yang tinggi?")
            
            # Scatter Plot: Members vs Rating
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rec_data, x="members", y="rating", alpha=0.5, color="#1f77b4", ax=ax_scatter)
            ax_scatter.set_title("Scatter Plot: Popularitas (Members) vs Kualitas (Rating)")
            ax_scatter.set_xlabel("Jumlah Members")
            ax_scatter.set_ylabel("Rating Global")
            st.pyplot(fig_scatter)
            
            st.info("💡 **Insight:** Jika grafik condong berkumpul di bagian kanan atas, artinya terdapat korelasi positif di mana anime populer cenderung mempertahankan rating yang baik karena besarnya basis penggemar.")

        # --- TAB 4: WAWASAN GENRE ---
        with tab4:
            st.subheader("4. Sebaran Tipe Tayangan Anime")
            st.write("Distribusi tipe penayangan anime di dalam dataset:")
            
            # Bar Chart untuk variabel kategori 'type'
            if 'type' in rec_data.columns:
                type_counts = rec_data['type'].value_counts()
                
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis", ax=ax_bar)
                ax_bar.set_title("Jumlah Anime Berdasarkan Tipe Tayangan")
                ax_bar.set_xlabel("Tipe")
                ax_bar.set_ylabel("Jumlah")
                st.pyplot(fig_bar)
            else:
                st.write("Fitur 'type' tidak ditemukan dalam data.")
elif menu == "Description Page":
    st.title("📝 Dataset Description")
    
    st.markdown("""
    ## Dataset Overview
    - **Source:** MyAnimeList (via Kaggle)
    - **Dataset Name:** [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
    - **Creators:** CooperUnion
    - **Scope:** Data preferensi dari 73.516 user untuk 12.294 anime.
    """)
    
    # --- FEATURE DICTIONARY ---
    st.subheader("📋 Feature Dictionary")
    st.write("Dataset ini terdiri dari dua file utama dengan detail kolom sebagai berikut:")
    
    with st.expander("📂 1. Keterangan Kolom anime.csv (Informasi Katalog)"):
        st.markdown("""
        * **anime_id:** ID unik dari myanimelist.net untuk mengidentifikasi setiap judul anime.
        * **name:** Nama lengkap atau judul resmi dari anime.
        * **genre:** Daftar genre yang melekat pada anime tersebut (dipisahkan oleh koma).
        * **type:** Format penayangan anime (contoh: TV, Movie, OVA, Special).
        * **episodes:** Jumlah total episode tayangan (bernilai `1` jika bertipe Movie).
        * **rating:** Rata-rata nilai rating global (skala 1-10) yang dihitung secara agregat.
        * **members:** Jumlah total anggota komunitas yang memasukkan anime ini ke dalam daftar mereka (indikator popularitas).
        """)
        
    with st.expander("📂 2. Keterangan Kolom rating.csv (Preferensi User)"):
        st.markdown("""
        * **user_id:** ID acak terenkripsi untuk mengidentifikasi user unik secara anonim.
        * **anime_id:** ID anime yang telah berinteraksi dengan user tersebut.
        * **rating:** Nilai rating (skala 1-10) yang diberikan oleh user. 
          * *Catatan:* Nilai `-1` menandakan user tersebut telah menonton anime-nya namun tidak memberikan nilai rating numerik.
        """)

    # --- DATA SHAPE INFO ---
    st.write("---")
    st.subheader("📊 Data Shape & Quick Info")
    
    if not rec_data.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris Data (Katalog)", f"{rec_data.shape[0]:,}")
        with col2:
            st.metric("Total Fitur/Kolom", f"{rec_data.shape[1]}")
        with col3:
            st.metric("Missing Values Terdeteksi", f"{rec_data.isnull().sum().sum()}")
            
        st.write("**Pratinjau Data Mentah (Raw Data Preview):**")
        st.dataframe(rec_data.head(10), use_container_width=True)
    else:
        st.warning("Data anime belum dimuat. Pratinjau tidak dapat ditampilkan.")

    # --- RECOMMANDATION WORKFLOW ---
    st.write("---")
    st.subheader("⚙️ Recommendation Workflow")
    
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue, fontname="Helvetica"];
        "Raw Data (Kaggle)" -> "EDA & Cleaning" -> "TF-IDF / Matrix Processing" -> "Cosine Similarity" -> "Generate Recommendation";
    }
    """)
    
    st.caption("Alur pemrosesan data dari pembacaan dataset hingga menghasilkan rekomendasi anime kustom kepada user.")
elif menu == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing Pipeline")
    st.write("Tahapan pembersihan data dan transformasi fitur sebelum dimasukkan ke dalam model TF-IDF.")

    if rec_data.empty:
        st.warning("Data anime tidak tersedia untuk diproses.")
    else:
        # Membuat Tab seperti halaman sebelumnya agar rapi
        tab_clean1, tab_clean2 = st.tabs(["🧹 Data Cleaning", "🚀 Feature Engineering (Text)"])

        with tab_clean1:
            st.subheader("1. Pembersihan Data Mentah (Data Cleaning)")
            
            # Info 1: Drop Missing Values & Duplicates
            st.markdown("""
            * **Penanganan Missing Values:** Menghapus baris data anime yang tidak memiliki informasi nama (`name`).
            * **Pembersihan Data Duplikat:** Menghapus baris duplikat berdasarkan judul anime dan hanya mempertahankan baris pertama (`keep='first'`).
            """)
            
            # Info 2: Optimasi Memori & Pembatasan Data (Crucial Insight)
            st.info("⚡ **Strategi Optimasi Memori (Anti-Crash):**\n\n"
                    "Streamlit Cloud membatasi penggunaan RAM maksimal 1 GB. Karena komputasi matriks kesamaan "
                    "(*Sigmoid Kernel*) membutuhkan memori besar, dataset disaring secara otomatis dengan hanya mengambil "
                    "**5.000 anime terpopuler** berdasarkan jumlah komunitas (`members`).")
            
            # Tampilkan statistik sederhana data setelah preprocessing
            col_pre1, col_pre2 = st.columns(2)
            with col_pre1:
                st.metric("Jumlah Baris Akhir (Terpopuler)", f"{rec_data.shape[0]:,}")
            with col_pre2:
                st.metric("Nilai Kosong Tersisa (Genre)", f"{rec_data['genre'].isnull().sum()}")

        with tab_clean2:
            st.subheader("2. Penyiapan Fitur Teks untuk TF-IDF")
            st.write("Sebelum teks genre diubah menjadi angka oleh matriks TF-IDF, dilakukan transformasi bentuk string:")
            
            # Tampilkan kode pemrosesan teks yang kamu gunakan di fungsi load data
            st.code("""
# Mengisi nilai genre yang kosong dengan string kosong
rec_data["genre"] = rec_data["genre"].fillna("")

# Melakukan split teks genre dan mengubahnya menjadi format string array untuk dibaca TF-IDF
genres = rec_data["genre"].str.split(", |, |,").astype(str)
            """, language="python")
            
            st.write("**Hasil Pemrosesan Fitur Teks (Siap untuk Tokenisasi):**")
            
            # Membuat contoh preview teks genre yang sudah siap di-vektorisasi
            preview_df = rec_data[['name', 'genre']].head(5).copy()
            preview_df['Processed Token String'] = preview_df['genre'].str.split(", |, |,").astype(str)
            st.dataframe(preview_df, use_container_width=True)

elif menu == "Train Model":
    st.title("🧠 Model Training & Configuration")
    st.write("Halaman ini menjelaskan bagaimana model *Content-Based Filtering* mempelajari karakteristik genre anime.")

    if rec_data.empty:
        st.warning("Data belum siap. Silakan periksa kembali file dataset kamu.")
    else:
        # Layout Dua Kolom untuk Parameter
        st.subheader("🛠️ 1. Model Hyperparameters")
        st.write("Konfigurasi yang digunakan pada algoritma `TfidfVectorizer` untuk mengekstrak fitur genre:")
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            st.markdown("""
            - **N-Gram Range:** `(1, 3)` *(Membaca kombinasi 1 sampai 3 kata)*
            - **Stop Words:** `English` *(Mengabaikan kata hubung bawaan)*
            """)
        with col_param2:
            st.markdown("""
            - **Max Features:** `3,000` *(Membatasi maksimal 3.000 kosakata unik)*
            - **Similarity Metric:** `Sigmoid Kernel`
            """)

        st.write("---")
        st.subheader("📐 2. Hasil Komputasi Matriks Kedekatan")
        st.write("Setelah model 'dilatih', bentuk dimensi data berubah menjadi matriks matematika:")

        # Menghitung bentuk matriks secara tiruan berdasarkan bentuk data asli untuk simulasi visual
        num_anime = rec_data.shape[0]
        
        col_mat1, col_mat2 = st.columns(2)
        with col_mat1:
            st.metric("Dimensi Matriks TF-IDF", f"{num_anime} × 3,000")
            st.caption("Baris mewakili judul anime, kolom mewakili bobot kata genre.")
        with col_mat2:
            st.metric("Dimensi Kernel Kesamaan (Sigmoid)", f"{num_anime} × {num_anime}")
            st.caption("Matriks persegi yang menyimpan skor kemiripan antar setiap anime.")

        st.write("---")
        st.subheader("💾 3. Status Model Deployment")
        
        # Indikator status apakah model berhasil dimuat di memori ram atau tidak
        if sig is not None:
            st.success("✅ **Model Status: Trained & Active**")
            st.info("💡 **Informasi:** Matriks kesamaan telah berhasil disimpan ke dalam *cache memory* Streamlit. Sistem siap memberikan rekomendasi instan pada menu **Result / Prediction Demo** tanpa perlu melatih ulang data dari awal.")
        else:
            st.error("❌ **Model Status: Not Trained / Error**")

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
    st.title("📊 Feature Importance Analysis")
    st.write("Menampilkan bobot fitur kata (N-Gram) dari genre yang paling memengaruhi model rekomendasi.")

    if rec_data.empty:
        st.warning("Data anime tidak tersedia untuk dianalisis.")
    else:
        st.subheader("🧐 Bagaimana Model Menilai Sebuah Genre?")
        st.write("Model menggunakan metode **TF-IDF**. Semakin tinggi nilai skor sebuah kata/genre, "
                 "artinya kata tersebut sifatnya unik dan menjadi pembeda kuat antar-anime di dalam sistem.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Kita lakukan proses ekstraksi fitur secara terpisah untuk visualisasi
            # Menggunakan parameter yang persis sama dengan dapur modelmu
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            genres_clean = rec_data["genre"].str.split(", |, |,").astype(str)
            tfv_viz = TfidfVectorizer(min_df=3, max_features=3000, 
                                      strip_accents="unicode", analyzer="word", 
                                      token_pattern=r"\w{1,}", ngram_range=(1, 3), 
                                      stop_words="english")
            tfv_viz.fit(genres_clean)
            
            # Mengambil daftar kata (feature names) dan nilai IDF-nya
            feature_names = tfv_viz.get_feature_names_out()
            idfs = tfv_viz.idf_
            
            # Membuat DataFrame untuk menampung skor kepentingan fitur
            importance_df = pd.DataFrame({
                'Genre Feature': feature_names,
                'Importance Score (IDF)': idfs
            }).sort_values(by='Importance Score (IDF)', ascending=False) # Urutkan dari yang tertinggi
            
            # Slider Interaktif untuk menentukan jumlah fitur yang ingin dilihat user
            num_features = st.slider("Pilih jumlah fitur terpenting yang ingin ditampilkan:", 
                                     min_value=10, max_value=40, value=15)
            
            # Ambil data top N sesuai pilihan slider
            top_features = importance_df.head(num_features)
            
            st.write(f"### 🔝 Top {num_features} Fitur Genre Paling Berpengaruh")
            
            # Pembuatan Grafik Bar Chart Horizontal menggunakan Seaborn
            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_features, 
                        x='Importance Score (IDF)', 
                        y='Genre Feature', 
                        palette='flare', 
                        ax=ax_importance)
            
            ax_importance.set_title(f"Top {num_features} Feature Importance Berdasarkan Skor IDF", fontsize=14)
            ax_importance.set_xlabel("Skor Kepentingan (Semakin Tinggi = Semakin Unik/Spesifik)", fontsize=11)
            ax_importance.set_ylabel("Fitur Teks / N-Gram Genre", fontsize=11)
            
            # Tampilkan Grafik di Streamlit
            st.pyplot(fig_importance)
            
            # Tampilkan data dalam bentuk tabel interaktif di bawahnya
            with st.expander("📄 Lihat Seluruh Daftar Tabel Bobot Fitur"):
                st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
                
        except Exception as e:
            st.error(f"Gagal memproses visualisasi Feature Importance: {e}")
elif menu == "About Us":
    st.title("👥 About Us")
    
    # --- HEADER GROUP ---
    st.markdown("""
    <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-bottom: 25px;">
        <h2 style="margin: 0; color: white;">Group 6</h2>
        <p style="margin: 5px 0 0 0; color: #B0B0B0;">Machine Learning Assignment — Binus University</p>
        <p style="margin: 0; color: #888888; font-size: 0.9em;">Binusian 2028</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- TEAM MEMBERS SECTION ---
    st.subheader("🎓 Team Members")
    
    # Membuat 3 kolom untuk 3 anggota tim
    col_member1, col_member2, col_member3 = st.columns(3)
    
    with col_member1:
        st.markdown("""
        <div style="background-color: #262626; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0;">Jimmy Stephen</h4>
            <p style="color: #FF4B4B; margin: 0; font-weight: bold;">2802461151</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_member2:
        st.markdown("""
        <div style="background-color: #262626; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0;">Nicholas Lee</h4>
            <p style="color: #FF4B4B; margin: 0; font-weight: bold;">2802450721</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_member3:
        st.markdown("""
        <div style="background-color: #262626; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0;">C. Darryl Witono</h4>
            <p style="color: #FF4B4B; margin: 0; font-weight: bold;">2802465420</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    # --- UNIVERSITY & COURSE INFO ---
    st.subheader("🏛️ University & Course Detail")
    
    info_univ = {
        "Detail": [
            "University", 
            "Program", 
            "Batch", 
            "Course", 
            "Assignment", 
            "Semester"
        ],
        "Information": [
            "Binus University", 
            "Computer Science", 
            "Binusian 2028", 
            "Machine Learning", 
            "Group Project — Recommendation System", 
            "Even Semester 2025/2026"
        ]
    }
    st.table(pd.DataFrame(info_univ).set_index("Detail"))

    st.write("---")

    # --- PROJECT INFORMATION ---
    st.subheader("📋 Project Information")
    
    info_project = {
        "Item": [
            "Project Title", 
            "Type", 
            "Dataset Source", 
            "Models / Core Logic", 
            "Framework", 
            "Visualization"
        ],
        "Project Detail": [
            "Machine Learning-Based Analysis of User Preferences for Recommending Future Anime & Manga", 
            "Content-Based Filtering (Unsupervised Learning)", 
            "Kaggle — Anime Recommendations Database", 
            "TF-IDF Vectorizer & Sigmoid Kernel Similarity", 
            "Streamlit (Python)", 
            "Matplotlib, Seaborn"
        ]
    }
    st.table(pd.DataFrame(info_project).set_index("Item"))
    
    # --- FOOTER ---
    st.write("---")
    st.caption("© 2026 Group 6 — Binus University | Machine Learning Assignment. Built with ❤️ using Python & Streamlit")
