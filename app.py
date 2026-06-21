import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oscar Predictor",
    page_icon="🏆",
    layout="centered"
)

# ---------------- BACKGROUND ----------------

def add_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
        background-size: cover;
        background-position: center;
    }
    .main {
        background-color: rgba(0,0,0,0.75);
        padding: 30px;
        border-radius: 15px;
    }
    h1,h2,h3,p,label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

add_bg()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("movie_metadata.csv")  # 👈 CSV NAME

df = load_data()

# ---------------- UI ----------------
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    st.title("🏆 Oscar Winner Predictor")
    st.write("Machine Learning Based Oscar Prediction System")

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATA PREP ----------------
    df_model = df[
        ['imdb_score',
         'num_voted_users',
         'budget',
         'gross',
         'duration',
         'title_year',
         'genres',
         'country',
         'language']
    ].dropna()

    # Create Oscar-like target
    df_model['oscar_win'] = (
        (df_model['imdb_score'] > 7.2) &
        (df_model['num_voted_users'] > 30000)
    ).astype(int)

    # Encode categorical
    df_model = pd.get_dummies(
        df_model,
        columns=['genres', 'country', 'language']
    )

    X = df_model.drop('oscar_win', axis=1)
    y = df_model['oscar_win']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"✅ Model Trained | Accuracy: {acc:.2f}")

    # ---------------- USER INPUT ----------------
    st.markdown("---")
    st.subheader("🎬 Enter Movie Details")

    rating = st.slider("IMDb Rating", 0.0, 10.0, 7.5)
    votes = st.number_input("IMDb Votes", 1000, 5000000, 100000)
    budget = st.number_input("Budget (Million $)", 1, 500, 50)
    gross = st.number_input("Box Office (Million $)", 0, 10000, 100)
    runtime = st.number_input("Runtime (Minutes)", 60, 300, 120)
    year = st.number_input("Release Year", 1920, 2026, 2024)

    genre = st.selectbox("Genre", sorted(df['genres'].dropna().unique()))
    country = st.selectbox("Country", sorted(df['country'].dropna().unique()))
    language = st.selectbox("Language", sorted(df['language'].dropna().unique()))

    # ---------------- INPUT PROCESSING ----------------
    input_df = pd.DataFrame([{
        'imdb_score': rating,
        'num_voted_users': votes,
        'budget': budget,
        'gross': gross,
        'duration': runtime,
        'title_year': year,
        'genres': genre,
        'country': country,
        'language': language
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # ---------------- PREDICTION ----------------
    if st.button("🎖️ Predict Oscar Chance"):
        prob = model.predict_proba(input_df)[0][1] * 100
        st.success(f"🏆 Oscar Winning Probability: **{prob:.2f}%**")

    st.info("📌 Prediction is based on historical movie metadata patterns")

    st.markdown("</div>", unsafe_allow_html=True)
