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
# 0. SAKURA MOCHI THEME CUSTOMIZATION (CSS GLOBAL)
# ==========================================
st.markdown("""
    <style>
        /* 1. Mengubah latar belakang utama menjadi Pink */
        .stApp {
            background-color: #FFB7C5;
        }
        
        /* 2. Mengubah warna judul utama (H1) menjadi Pink Sakura */
        h1 {
            color: ##000000 !important;
            font-weight: 700;
        }
        
        /* 3. Mengubah warna sub-judul/deskripsi langsung di bawah judul menjadi Hijau Matcha */
        h1 + p, .stMarkdown p:first-of-type {
            color: #01691b !important;
            font-size: 1.1rem;
        }
        
        /* 4. Aksen Coklat Kayu untuk Sub-header (H2, H3, dll) dan teks umum */
        h2, h3, h4, h5, h6 {
            color: #5C4033 !important;
        }
        
        div[data-testid="stMarkdownContainer"] p {
            color: #5C4033; /* Coklat kayu gelap agar teks deskripsi tetap kontras dan terbaca */
        }
        
        /* 5. Kostumisasi Sidebar */
        [data-testid="stSidebar"] {
            background-color: #F9F1F0; /* Putih sedikit semburat pink lembut */
            border-right: 2px solid #6F4E37; /* Garis pembatas coklat kayu */
        }
        
        [data-testid="stSidebar"] * {
            color: #6F4E37 !important; /* Semua teks di sidebar berwarna coklat kayu */
        }
        
        /* 6. Mengubah warna komponen interaktif & MEMBUAT TEKS PILLS MENJADI PUTIH */
        .st-bd, .st-ae, .st-af {
            color: #FFB7B2 !important;
        }
        
        /* Mengubah warna teks di dalam st.pills (tombol pilihan hitam) menjadi putih */
        div[data-testid="stPills"] button p {
            color: ##000000 !important;
        }

        /* 7. KUSTOMISASI TABEL: Warna Merah Gelap / Isian Pasta Kacang (Anko) */
        .stTable table {
            color: ##000000 !important; /* MENGUBAH TEKS ISI TABEL MENJADI PUTIH BERSIH */
            border: 2px solid #FFFFFF !important; /* Garis luar merah gelap/maroon */
        }
        
        .stTable th {
            background-color: #FFFFFF !important; /* Latar belakang header menjadi merah gelap */
            color: ##000000 !important; /* Teks header menjadi putih agar kontras */
            font-weight: bold;
        }
        
        .stTable td {
            border-bottom: 1px solid #E0B0FF !important; /* Garis pembatas baris pink tipis */
            color: ##000000 !important; /* Memastikan teks di dalam cell td berwarna putih */
        }
        
        /* Mengubah warna teks di dalam komponen dataframe/tabel interaktif agar ikut menjadi putih */
        div[data-testid="stDataFrame"] {
            border: 2px solid #FFFFFF !important;
        }
        
        div[data-testid="stDataFrame"] div {
            color: ##000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. LOCAL DATASET EXTRACTION FROM GITHUB
# ==========================================
@st.cache_resource
def extract_local_dataset():
    path_anime = "anime.csv"
    zip_target = "anime.zip"
    if not os.path.exists(path_anime) and os.path.exists(zip_target):
        try:
            with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Local anime.zip extracted successfully!")
        except Exception as e:
            st.error(f"Failed to extract local files: {e}")

extract_local_dataset()

# ==========================================
# 2. DATA LOADING & MODEL GENERATION (RAM OPTIMIZED)
# ==========================================
@st.cache_data
def load_and_process_data():
    try:
        if os.path.exists("anime.csv"):
            anime_dtypes = {'anime_id': 'int32', 'rating': 'float32', 'members': 'int32'}
            anime = pd.read_csv("anime.csv", dtype=anime_dtypes)
            
            rec_data = anime.dropna(subset=['name']).copy()
            rec_data.drop_duplicates(subset="name", keep="first", inplace=True)
            rec_data.reset_index(drop=True, inplace=True)
            
            # Restrict dataset to top 5,000 most popular titles to prevent RAM crash on Streamlit Cloud
            if len(rec_data) > 5000:
                rec_data = rec_data.sort_values(by='members', ascending=False).head(5000)
                rec_data.reset_index(drop=True, inplace=True)
                
            rec_data["genre"] = rec_data["genre"].fillna("")
            genres = rec_data["genre"].str.split(", |, |,").astype(str)
            
            tfv = TfidfVectorizer(min_df=3, max_features=3000, 
                                  strip_accents="unicode", analyzer="word", 
                                  token_pattern=r"\w{1,}", ngram_range=(1, 3), 
                                  stop_words="english")
            tfv_matrix = tfv.fit_transform(genres)
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
            rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()
            
            return rec_data, sig, rec_indices
        else:
            st.error("anime.csv file not found after extraction!")
            return pd.DataFrame(), None, None
    except Exception as e:
        st.error(f"Failed to process data: {e}")
        return pd.DataFrame(), None, None

rec_data, sig, rec_indices = load_and_process_data()

# ==========================================
# 3. SIDEBAR LAYOUT (NAVIGATION)
# ==========================================
with st.sidebar:
    st.title("⛩️ Anime Recommendation")
    st.write("---")
    st.markdown("### Navigation")
    menu = st.radio(
        label="Navigation Menu",
        options=["Home", "EDA", "Description Page", "Data Preprocessing", "Train Model", "Result / Prediction Demo", "Feature Importance", "About Us"],
        label_visibility="collapsed"
    )
    st.sidebar.write("---")
    st.sidebar.markdown("**MyAnimeList Dataset**\n\n*Anime Recommendation Engine*\n\n---\nBuilt with ❤️ using Streamlit")

# ==========================================
# 4. MAIN PAGE CONTENT LOGIC
# ==========================================

# --- HOME PAGE ---
if menu == "Home":
    st.title("🏠 Home")
    st.markdown("""
    ### Welcome to the **Anime Recommendation System** app! 👋
    
    This recommendation system is designed interactively to help anime fans instantly find relevant new series or movies without getting lost in thousands of available options.
    """)
    
    st.success("""
    💡 **Short Project Summary:**
    
    This application utilizes *Unsupervised Learning* principles through a **Content-Based Filtering** algorithm. By extracting text features from the `genre` metadata, the model measures structural similarities between anime using a combined **TF-IDF Vectorizer** and **Sigmoid Kernel Similarity Matrix** approach.
    
    The final result is a smart engine capable of recommending future anime titles that align perfectly with the genre patterns and narrative styles of your favorite shows.
    """)
    
    st.write("---")
    st.markdown("""
    #### 🚀 Explore the Application:
    Use the **Navigation** menu on the left sidebar to access our complete data experiments:
    1. **EDA:** View visual distributions and insights from the raw catalog dataset.
    2. **Description Page & Preprocessing:** Understand the core data structures and cleaning operations.
    3. **Train Model & Feature Importance:** Check model parameters and top genre weights.
    4. **Result / Prediction Demo:** Test the live recommendation search engine!
    5. **About Us:** Complete team profile for **Group 6 (Binus University)**.
    """)

# --- EDA PAGE ---
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("Explore the underlying traits and distributions of the anime catalog interactively.")
    
    if rec_data.empty:
        st.warning("Anime dataset is empty. Please verify your data source file.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🔢 Descriptive Stats", "📈 Feature Distributions", "🔥 Correlations & Relations", "🎭 Genre Insights"])
        
        with tab1:
            st.subheader("1. Data Summary Statistics")
            st.write("Overview of numerical measurements from the filtered dataset:")
            st.dataframe(rec_data.describe(), use_container_width=True)
            st.write("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analysed Anime", f"{len(rec_data):,}")
            with col2:
                st.metric("Average Global Rating", f"{rec_data['rating'].mean():.2f} / 10")
            with col3:
                st.metric("Peak Community Members", f"{rec_data['members'].max():,}")

        with tab2:
            st.subheader("2. Feature Distribution Generator")
            feature_choice = st.selectbox("Select a feature to plot its distribution:", ["rating", "members"])
            bins_choice = st.slider("Number of Bins (Histogram Bars):", min_value=5, max_value=50, value=20)
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(rec_data[feature_choice], bins=bins_choice, kde=True, ax=ax, color="#FF4B4B")
            ax.set_title(f"Distribution Chart: {feature_choice.capitalize()}")
            ax.set_xlabel(feature_choice.capitalize())
            ax.set_ylabel("Anime Count")
            st.pyplot(fig)
            
            st.write("---")
            st.write(f"**Boxplot of {feature_choice.capitalize()} (Outlier Detection)**")
            fig_box, ax_box = plt.subplots(figsize=(8, 2))
            sns.boxplot(x=rec_data[feature_choice], ax=ax_box, color="#4682B4")
            st.pyplot(fig_box)

        with tab3:
            st.subheader("3. Inter-Feature Analysis")
            st.write("Does a highly popular anime (large member base) automatically guarantee a top-tier rating?")
            
            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=rec_data, x="members", y="rating", alpha=0.5, color="#1f77b4", ax=ax_scatter)
            ax_scatter.set_title("Scatter Plot: Popularity (Members) vs Quality (Rating)")
            ax_scatter.set_xlabel("Community Member Count")
            ax_scatter.set_ylabel("Global Score")
            st.pyplot(fig_scatter)
            st.info("💡 **Insight:** Strong clustering towards the top-right quadrant demonstrates a positive correlation: mass-market appeal frequently sustains high scoring due to large fanbase backing.")

        with tab4:
            st.subheader("4. Anime Format Distribution")
            st.write("Breakdown of presentation formats across the catalog data:")
            if 'type' in rec_data.columns:
                type_counts = rec_data['type'].value_counts()
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis", ax=ax_bar)
                ax_bar.set_title("Anime Count by Media Type Format")
                ax_bar.set_xlabel("Media Type")
                ax_bar.set_ylabel("Count")
                st.pyplot(fig_bar)
            else:
                st.write("Feature variable 'type' is missing from the active dataset.")

# --- DESCRIPTION PAGE ---
elif menu == "Description Page":
    st.title("📝 Dataset Description")
    st.markdown("""
    ## Dataset Overview
    - **Source:** MyAnimeList (via Kaggle)
    - **Dataset Name:** [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
    - **Creators:** CooperUnion
    - **Scope:** Preferences and ratings data from 73,516 users on 12,294 unique anime titles.
    """)
    
    st.subheader("📋 Feature Dictionary")
    st.write("The dataset structure consists of two core relational files with columns described below:")
    
    with st.expander("📂 1. Column Metadata for anime.csv (Catalog Information)"):
        st.markdown("""
        <div style="background-color: #FFF0F0; padding: 20px; border-radius: 8px; border: 1px solid #FFB7B2; margin-top: 10px;">
            <ul style="color: #5C4033; margin-bottom: 0;">
                <li><b>anime_id:</b> Unique numerical ID from myanimelist.net assigned to identify each anime title.</li>
                <li><b>name:</b> Official full name or broadcast title of the anime.</li>
                <li><b>genre:</b> Comma-separated listing of thematic category tags belonging to the title.</li>
                <li><b>type:</b> Release broadcast format of the anime (e.g., TV, Movie, OVA, Special).</li>
                <li><b>episodes:</b> Total count of episodes produced (defaults to 1 for theatrical Movies).</li>
                <li><b>rating:</b> Overall global average user review score computed out of 10 points.</li>
                <li><b>members:</b> Aggregate count of community user accounts that have added this item to their library lists (popularity proxy).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with st.expander("📂 2. Column Metadata for rating.csv (User Preferences)"):
        st.markdown("""
        <div style="background-color: #FFF0F0; padding: 20px; border-radius: 8px; border: 1px solid #FFB7B2; margin-top: 10px;">
            <ul style="color: #5C4033; margin-bottom: 0;">
                <li><b>user_id:</b> Randomly generated, anonymized string used to identify a single unique reviewer.</li>
                <li><b>anime_id:</b> Relational reference matching the specific anime rated by the user.</li>
                <li><b>rating:</b> Explicit review score (1-10 scale) assigned by the individual user.</li>
                <li style="list-style-type: none; margin-top: 10px; font-style: italic; color: #FFFFFF;">
                    ⚠️ <b>Special Value Note:</b> A value of -1 indicates that the user logged the title as watched but chose not to input an explicit numerical score.
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.subheader("📊 Data Shape & Quick Info")
    if not rec_data.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows Matrix (Catalog)", f"{rec_data.shape[0]:,}")
        with col2:
            st.metric("Total Structural Features", f"{rec_data.shape[1]}")
        with col3:
            st.metric("Detected Null / Missing Values", f"{rec_data.isnull().sum().sum()}")
        st.write("**Processed Data Snapshot (Raw Data Preview):**")
        st.dataframe(rec_data.head(10), use_container_width=True)
    else:
        st.warning("Data matrices are uninitialized. Snapshot preview unavailable.")

    st.write("---")
    st.subheader("⚙️ Recommendation Workflow")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color=lightblue, fontname="Helvetica"];
        "Raw Data (Kaggle)" -> "EDA & Cleaning" -> "TF-IDF / Matrix Processing" -> "Cosine Similarity" -> "Generate Recommendation";
    }
    """)
    st.caption("Step-by-step pipeline mapping data from initial repository download to deployment-ready cosine similarity calculations.")

# --- DATA PREPROCESSING PAGE ---
elif menu == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing Pipeline")
    st.write("Data pipeline cleaning stages executed before features can be fed into the TF-IDF Vectorizer model.")
    
    if rec_data.empty:
        st.warning("Active tables unavailable for preprocessing analysis.")
    else:
        tab_clean1, tab_clean2 = st.tabs(["🧹 Data Cleaning", "🚀 Feature Engineering (Text)"])
        with tab_clean1:
            st.subheader("1. General Cleaning & Null Handling")
            st.markdown("""
            * **Missing Values Drop:** Drop rows missing critical title identifiers (`name`) to protect lookup joins.
            * **Deduplication:** Scan for duplicate title logs and drop overlapping records, retaining only initial entries (`keep='first'`).
            """)
            st.info("⚡ **Memory Scaling Allocation Strategy (Anti-Crash Safeguard):**\n\n"
                    "Streamlit Cloud infrastructure limits system memory consumption to 1 GB RAM. Because multi-dimensional "
                    "array similarity computing (*Sigmoid Kernel*) expands exponentially, the dataset isolates the "
                    "**top 5,000 most popular titles** (`members`) to maximize processing efficiency without breaking container constraints.")
            col_pre1, col_pre2 = st.columns(2)
            with col_pre1:
                st.metric("Final Shape (Row Matrix)", f"{rec_data.shape[0]:,}")
            with col_pre2:
                st.metric("Unresolved Nulls (Genre Vector)", f"{rec_data['genre'].isnull().sum()}")
                
        with tab_clean2:
            st.subheader("2. Text Fitting & String Tokenization Prep")
            st.write("Prior to numerical translation, string parameters undergo parsing formatting transformations:")
            st.code("""
# Impute empty genre records with uniform blank string indicators
rec_data["genre"] = rec_data["genre"].fillna("")

# Split comma-separated genre attributes into structured string representations for the TF-IDF array
genres = rec_data["genre"].str.split(", |, |,").astype(str)
            """, language="python")
            st.write("**Transformed Text Feature Output (Ready for Bag-of-Words Vectorisation):**")
            preview_df = rec_data[['name', 'genre']].head(5).copy()
            preview_df['Processed Token String'] = preview_df['genre'].str.split(", |, |,").astype(str)
            st.dataframe(preview_df, use_container_width=True)

# --- TRAIN MODEL PAGE ---
elif menu == "Train Model":
    st.title("🧠 Model Training & Configuration")
    st.write("Technical review explaining how the unsupervised Content-Based Filter analyzes mathematical vector distributions of genre maps.")
    
    if rec_data.empty:
        st.warning("Data structures uninitialized. Model cannot verify parameters.")
    else:
        st.subheader("🛠️ 1. Vectorizer Hyperparameters")
        st.write("Structural parameters defined inside the `TfidfVectorizer` mapping configuration block:")
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            st.markdown("""
            - **N-Gram Token Range:** `(1, 3)` *(Evaluates sequences containing 1 to 3 combined words)*
            - **Stop Words Library:** `English` *(Filters out structural vocabulary noise)*
            """)
        with col_param2:
            st.markdown("""
            - **Max Features Limit:** `3,000` *(Constrains vocabulary dictionary bounds to 3k distinct attributes)*
            - **Similarity Function:** `Sigmoid Kernel Metric`
            """)
        st.write("---")
        st.subheader("📐 2. Computed Matrix Dimensional Profiles")
        st.write("Post-execution array alignment dimensions outputted by the vector learning model pipeline:")
        num_anime = rec_data.shape[0]
        col_mat1, col_mat2 = st.columns(2)
        with col_mat1:
            st.metric("TF-IDF Matrix Shape", f"{num_anime} × 3,000")
            st.caption("Rows represent structural anime records; columns represent computed keyword item weights.")
        with col_mat2:
            st.metric("Sigmoid Similarity Array", f"{num_anime} × {num_anime}")
            st.caption("Symmetrical cross-product space storing exact pair-wise score distances between every catalog title.")
        st.write("---")
        st.subheader("💾 3. Live Model Deployment Status")
        if sig is not None:
            st.success("✅ **Model Status: Trained & Active**")
            st.info("💡 **System Note:** Symmetrical similarity arrays are persistently mapped within memory cache layers. Real-time recommendation retrieval calculations on the **Result / Prediction Demo** menu execute immediately without recalculating base model weights.")
        else:
            st.error("❌ **Model Status: Not Trained / Error Detected**")

# --- RESULT / PREDICTION DEMO PAGE ---
elif menu == "Result / Prediction Demo":
    st.title("🎬 Anime Recommendation System")
    st.write("Generate dynamic anime recommendations driven completely by textual genre mapping models!")
    
    if sig is not None and not rec_data.empty:
        list_anime = rec_data['name'].unique()
        search_query = st.selectbox("Select or search for an anime title you enjoy:", list_anime)
        
        if "selected_genre" not in st.session_state:
            st.session_state.selected_genre = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
            
        if st.button("🔮 Generate Recommendations", use_container_width=True):
            st.session_state.current_page = 1
            st.session_state.selected_genre = None
            st.session_state.search_done = True
            st.session_state.current_anime = search_query
            
        if st.session_state.get("search_done", False):
            active_anime = st.session_state.current_anime
            idx = rec_indices[active_anime]
            sig_score = list(enumerate(sig[idx]))
            sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
            
            st.write("")
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            with nav_col1:
                if st.session_state.current_page > 1:
                    if st.button("Back ⬅️ `<`", use_container_width=True):
                        st.session_state.current_page -= 1
                        st.session_state.selected_genre = None
                        st.rerun()
                else:
                    st.button("⬅️ `<`", disabled=True, use_container_width=True)
            with nav_col2:
                current_start = ((st.session_state.current_page - 1) * 10) + 1
                current_end = st.session_state.current_page * 10
                st.markdown(f"<h5 style='text-align: center; margin-top: 5px;'>Page {st.session_state.current_page} (Rankings {current_start} - {current_end})</h5>", unsafe_allow_html=True)
            with nav_col3:
                max_pilihan_anime = len(sig_score) - 1
                max_halaman = min(5, max_pilihan_anime // 10)
                if st.session_state.current_page < max_halaman:
                    if st.button("`>` ➡️ Next", use_container_width=True):
                        st.session_state.current_page += 1
                        st.session_state.selected_genre = None
                        st.rerun()
                else:
                    st.button("`>` ➡️", disabled=True, use_container_width=True)
                    
            start_rank = ((st.session_state.current_page - 1) * 10) + 1
            end_rank = (st.session_state.current_page * 10) + 1
            nomor_urut = range(start_rank, end_rank)
            page_sig_score = sig_score[start_rank:end_rank]
            anime_indices = [i[0] for i in page_sig_score]
            
            rec_dic = {
                "No": nomor_urut,
                "Anime Title": rec_data["name"].iloc[anime_indices].values,
                "Genre": rec_data["genre"].iloc[anime_indices].values,
                "Rating": rec_data["rating"].iloc[anime_indices].values
            }
            st.session_state.df_result = pd.DataFrame(data=rec_dic).set_index("No")
            st.success(f"Displaying top 10 calculated matches for: **{active_anime}**")
            st.dataframe(st.session_state.df_result, use_container_width=True,
                         column_config={
                             "Anime Title": st.column_config.TextColumn("Anime Title", width="large"),
                             "Genre": st.column_config.TextColumn("Genre", width="large"),
                             "Rating": st.column_config.NumberColumn("Rating", width="small")
                         })
            
            st.write("---")
            st.subheader("🔍 Pt. 2: Deeper Genre Discovery")
            st.write("Isolate and explore matching titles by filtering specifically for an extracted keyword below:")
            all_genres = set()
            for g_str in st.session_state.df_result["Genre"]:
                genres_list = [g.strip() for g in g_str.split(",")]
                all_genres.update(genres_list)
            sorted_genres = sorted(list(all_genres))
            genre_click = st.pills("Filter by Theme tag:", sorted_genres, selection_mode="single")
            
            if genre_click:
                st.session_state.selected_genre = genre_click
                filtered_anime = rec_data[rec_data['genre'].str.contains(genre_click, case=False, na=False)]
                st.write(f"### 📋 Catalog Profiles Containing Theme: **{genre_click}**")
                filter_display = filtered_anime[['name', 'genre', 'rating']].copy()
                filter_display.columns = ['Anime Title', 'Genre', 'Rating']
                filter_display.insert(0, 'No', range(1, len(filter_display) + 1))
                filter_display.set_index('No', inplace=True)
                st.dataframe(filter_display.head(15), use_container_width=True,
                             column_config={
                                 "Anime Title": st.column_config.TextColumn("Anime Title", width="large"),
                                 "Genre": st.column_config.TextColumn("Genre", width="large"),
                                 "Rating": st.column_config.NumberColumn("Rating", width="small")
                             })
    else:
        st.info("Constructing analytical model dependencies, standby...")

# --- FEATURE IMPORTANCE PAGE ---
elif menu == "Feature Importance":
    st.title("📊 Feature Importance Analysis")
    st.write("Inspect structural vocabulary weight scores (N-Grams) indicating which descriptive themes carry maximum influence within the model system.")
    
    if rec_data.empty:
        st.warning("Missing required token dependencies to perform evaluation.")
    else:
        st.subheader("🧐 How Does the Vector Model Grade Token Importance?")
        st.write("Utilizing standard **TF-IDF Matrix Weights**: words receiving maximum evaluation metrics scores are highly specific, acting as distinct profile grouping separators across catalog similarities.")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            genres_clean = rec_data["genre"].str.split(", |, |,").astype(str)
            tfv_viz = TfidfVectorizer(min_df=3, max_features=3000, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 3), stop_words="english")
            tfv_viz.fit(genres_clean)
            
            feature_names = tfv_viz.get_feature_names_out()
            idfs = tfv_viz.idf_
            
            importance_df = pd.DataFrame({'Genre Feature': feature_names, 'Importance Score (IDF)': idfs}).sort_values(by='Importance Score (IDF)', ascending=False)
            num_features = st.slider("Select maximum features count to evaluate:", min_value=10, max_value=40, value=15)
            top_features = importance_df.head(num_features)
            
            st.write(f"### 🔝 Top {num_features} High-Impact Structural N-Grams")
            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_features, x='Importance Score (IDF)', y='Genre Feature', palette='flare', ax=ax_importance)
            ax_importance.set_title(f"Top {num_features} Feature Weights via Evaluated Inverse Document Frequency (IDF) Scores", fontsize=14)
            ax_importance.set_xlabel("Significance Weight Metrics (Higher Weight = High Specificity Profile Purity Marker)", fontsize=11)
            ax_importance.set_ylabel("Extracted Vocabulary Token N-Grams", fontsize=11)
            st.pyplot(fig_importance)
            
            with st.expander("📄 Review Full Token Dictionary Weight Array Sheet"):
                st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to compile feature importance charts visualization: {e}")

# --- ABOUT US PAGE ---
elif menu == "About Us":
    st.title("👥 About Us")
    
    st.markdown("""
    <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #FFB7B2; margin-bottom: 25px;">
        <h2 style="margin: 0; color: white;">Group 6</h2>
        <p style="margin: 5px 0 0 0; color: #B0B0B0;">Machine Learning Assignment — Binus University</p>
        <p style="margin: 0; color: #888888; font-size: 0.9em;">Binusian 2028</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎓 Team Members")
    col_member1, col_member2, col_member3 = st.columns(3)
    with col_member1:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0; color: #FFB7B2 !important;">Jimmy Stephen</h4>
            <p style="color: #A8E6CF; margin: 0; font-weight: bold;">2802461151</p>
        </div>
        """, unsafe_allow_html=True)
    with col_member2:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0; color: #FFB7B2 !important;">Nicholas Lee</h4>
            <p style="color: #A8E6CF; margin: 0; font-weight: bold;">2802450721</p>
        </div>
        """, unsafe_allow_html=True)
    with col_member3:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #444;">
            <span style="font-size: 40px;">👨‍💻</span>
            <h4 style="margin: 10px 0 5px 0; color: #FFB7B2 !important;">C. Darryl Witono</h4>
            <p style="color: #A8E6CF; margin: 0; font-weight: bold;">2802465420</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    st.subheader("🏛 *University & Course Detail*")
    info_univ = {
        "Detail": ["University", "Program", "Batch", "Course", "Assignment", "Semester"],
        "Information": ["Binus University", "Computer Science", "Binusian 2028", "Machine Learning", "Group Project — Recommendation System", "Even Semester 2025/2026"]
    }
    st.table(pd.DataFrame(info_univ).set_index("Detail"))
    
    st.write("---")
    st.subheader("📋 Project Information")
    info_project = {
        "Item": ["Project Title", "Type", "Dataset Source", "Models / Core Logic", "Framework", "Visualization"],
        "Project Detail": ["Machine Learning-Based Analysis of User Preferences for Recommending Future Anime & Manga", "Content-Based Filtering (Unsupervised Learning)", "Kaggle — Anime Recommendations Database", "TF-IDF Vectorizer & Sigmoid Kernel Similarity", "Streamlit (Python)", "Matplotlib, Seaborn"]
    }
    st.table(pd.DataFrame(info_project).set_index("Item"))
    
    st.write("---")
    st.caption("© 2026 Group 6 — Binus University | Machine Learning Assignment. Built with ❤️ using Python & Streamlit")
