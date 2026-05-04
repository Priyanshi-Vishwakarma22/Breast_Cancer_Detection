import streamlit as st
import pandas as pd
import dill
from src.components.data_transformation import DateFeatureExtractor

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    model = dill.load(open("artifacts/model.pkl", "rb"))
    preprocessor = dill.load(open("artifacts/preprocessor.pkl", "rb"))
    return model, preprocessor

model, preprocessor = load_model()

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# ---------------- UI DESIGN ---------------- #
st.title("🧬 Breast Cancer Survival Prediction")
st.markdown("### Enter patient details")

# ---------- INPUT SECTION ---------- #
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    protein1 = st.number_input("Protein 1", value=0.0)
    protein2 = st.number_input("Protein 2", value=0.0)
    tumour_stage = st.selectbox("Tumour Stage", ["I", "II", "III"])

with col3:
    protein3 = st.number_input("Protein 3", value=0.0)
    protein4 = st.number_input("Protein 4", value=0.0)
    surgery = st.selectbox("Surgery Type", ["Lumpectomy", "Simple Mastectomy", "Other"])

# Extra fields
histology = st.selectbox("Histology", [
    "Infiltrating Ductal Carcinoma",
    "Infiltrating Lobular Carcinoma",
    "Mucinous Carcinoma"
])

her2 = st.selectbox("HER2 Status", ["Positive", "Negative"])

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict"):

    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Protein1': [protein1],
        'Protein2': [protein2],
        'Protein3': [protein3],
        'Protein4': [protein4],
        'Tumour_Stage': [tumour_stage],
        'Histology': [histology],
        'HER2 status': [her2],
        'Surgery_type': [surgery]
    })

    try:
        data = preprocessor.transform(input_df)

        # 🔥 CUSTOM THRESHOLD LOGIC
        proba = model.predict_proba(data)[0][1]

        if proba > 0.6:
            prediction = 1
        else:
            prediction = 0

        st.markdown("---")

        # ---------- OUTPUT ---------- #
        if prediction == 1:
            st.success(f"✅ Patient likely to SURVIVE")
            st.write(f"Confidence: {proba:.2f}")
        else:
            st.error(f"⚠️ High Risk Patient (DEAD)")
            st.write(f"Confidence: {proba:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")