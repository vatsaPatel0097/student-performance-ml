import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ---------- Load Model ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please train the model first.")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# ---------- App UI ----------
st.set_page_config(page_title="Math Score Predictor", layout="centered")

st.title("📊 Student Math Score Predictor")
st.write(
    "Predict a student's **math score** based on background information "
    "and reading/writing performance."
)

st.divider()


# ---------- User Inputs ----------
gender = st.selectbox("Gender", ["male", "female"])

race = st.selectbox(
    "Race / Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parent_edu = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

test_prep = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

reading_score = st.slider("Reading Score", 0, 100, 70)
writing_score = st.slider("Writing Score", 0, 100, 70)


# ---------- Prediction ----------
if st.button("Predict Math Score"):
    if model is None:
        st.stop()

    input_data = pd.DataFrame(
        [{
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parent_edu,
            "lunch": lunch,
            "test preparation course": test_prep,
            "reading score": reading_score,
            "writing score": writing_score
        }]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Math Score: **{prediction:.2f}**")
