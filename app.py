import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Diabetes",
    layout="centered",
)

# MEMUAT MODEL, SCALER, DAN NAMA FITUR
BASE = Path(__file__).parent
MODELS_DIR = BASE / "TM8" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE / "models"


@st.cache_resource
def load_assets():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feats = joblib.load(MODELS_DIR / "feature_names.joblib")
    return model, scaler, feats


try:
    model, scaler, feats = load_assets()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan. Jalankan `classification.py` terlebih dahulu "
        "agar folder `TM8/models/` (atau `models/`) terbentuk."
    )
    st.stop()

# TAMPILAN UTAMA
st.title("Prediksi Risiko Diabetes")
st.write(
    "Masukkan data pemeriksaan pasien, lalu tekan tombol **Prediksi** "
    "untuk melihat hasil klasifikasi."
)
st.info(f"Model yang digunakan: **{type(model).__name__}** (akurasi tertinggi pada pengujian)")

# FORM INPUT
st.subheader("Data Pasien")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input(
        "Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=1, step=1
    )
    glu = st.number_input(
        "Glucose (mg/dL)", min_value=1, max_value=300, value=120, step=1
    )
    bp = st.number_input(
        "Blood Pressure (mm Hg)", min_value=1, max_value=150, value=70, step=1
    )
    skin = st.number_input(
        "Skin Thickness (mm)", min_value=1, max_value=100, value=29, step=1
    )

with col2:
    ins = st.number_input(
        "Insulin (µU/mL)", min_value=1, max_value=900, value=125, step=1
    )
    bmi = st.number_input(
        "BMI (kg/m²)", min_value=1.0, max_value=70.0, value=32.0, step=0.1, format="%.1f"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05,
        max_value=2.5,
        value=0.47,
        step=0.01,
        format="%.3f",
    )
    age = st.number_input(
        "Age (tahun)", min_value=21, max_value=100, value=33, step=1
    )

# PREDIKSI
if st.button("Prediksi", type="primary", use_container_width=True):
    # Urutan kolom harus sama dengan saat training
    data = {
        "Pregnancies": preg,
        "Glucose": glu,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": ins,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    row = pd.DataFrame([data])[feats]

    # Normalisasi dengan scaler hasil training, lalu prediksi
    x_scaled = scaler.transform(row)
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]
    p_diabetes = proba[1]

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error(f"**Berisiko Diabetes** (probabilitas diabetes {p_diabetes:.1%})")
    else:
        st.success(f"**Non-Diabetes** (probabilitas diabetes {p_diabetes:.1%})")

    st.progress(float(p_diabetes))

    c1, c2 = st.columns(2)
    c1.metric("Probabilitas Non-Diabetes", f"{proba[0]:.1%}")
    c2.metric("Probabilitas Diabetes", f"{proba[1]:.1%}")

    with st.expander("Lihat data yang dimasukkan"):
        st.dataframe(row, use_container_width=True)

    st.caption(
        "Hasil ini hanya untuk keperluan tugas pembelajaran mesin, "
        "bukan pengganti diagnosis medis."
    )

# SIDEBAR
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.write(
        "Aplikasi ini memprediksi apakah pasien berisiko diabetes "
        "berdasarkan 8 fitur klinis dari dataset Pima Indians Diabetes."
    )
    st.write("**Tahapan model:**")
    st.write(
        "- Imputasi nilai 0 dengan median\n"
        "- Capping outlier (IQR)\n"
        "- Normalisasi Min-Max\n"
        "- Oversampling SMOTE\n"
        "- Klasifikasi"
    )