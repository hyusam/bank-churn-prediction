import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_assets():
    # Menyesuaikan path relatif berdasarkan struktur project
    # Jika app.py ada di root directory, ubah menjadi 'assets/xgb_model.pkl'
    base_path = os.path.join(os.path.dirname(__file__), '../assets')
    model_path = os.path.join(base_path, 'xgb_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    # Jika gagal mencari dengan path relatif, fallback ke folder 'assets' di direktori yang sama
    if not os.path.exists(model_path):
        model_path = 'assets/xgb_model.pkl'
        scaler_path = 'assets/scaler.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    xgb_tuned, scaler = load_assets()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model atau scaler. Pastikan folder 'assets' tersedia. Error: {e}")
    st.stop()

# ==========================================
# 3. HEADER & DESKRIPSI
# ==========================================
st.title("Bank Customer Churn Prediction")
st.markdown("""
Aplikasi AI ini menggunakan model **XGBoost Tuned** untuk memprediksi probabilitas seorang nasabah akan meninggalkan bank (*churn*) atau tetap bertahan (*retained*).
""")
st.divider()

# ==========================================
# 4. FORM INPUT DATA NASABAH
# ==========================================
st.subheader("Data Profil Nasabah")

with st.form(key="customer_data_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        geography = st.selectbox("Geography (Negara)", ["France", "Spain", "Germany"])
        gender = st.radio("Gender (Jenis Kelamin)", ["Male", "Female"])
        age = st.number_input("Age (Usia)", min_value=18, max_value=100, value=35, step=1)
        tenure = st.slider("Tenure (Lama Bergabung - Tahun)", min_value=0, max_value=10, value=5)
        
    with col2:
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"], index=0)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔍 Analisis & Prediksi Churn")

# ==========================================
# 5. PREPROCESSING & PREDIKSI
# ==========================================
if submit_button:
    # 5.1. Encoding Manual Sesuai get_dummies(drop_first=True)
    # France dan Female adalah baseline (di-drop saat get_dummies di Notebook)
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0
    
    # Konversi Yes/No ke Biner
    has_cr_card_bin = 1 if has_cr_card == "Yes" else 0
    is_active_member_bin = 1 if is_active_member == "Yes" else 0
    
    # 5.2. Susun Dataframe (Urutan HARUS sama persis dengan X_train di notebook)
    # Urutan di Notebook: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, 
    # IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain, Gender_Male
    feature_names = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]
    
    input_data = pd.DataFrame([[
        credit_score, age, tenure, balance, num_of_products,
        has_cr_card_bin, is_active_member_bin, estimated_salary,
        geo_germany, geo_spain, gender_male
    ]], columns=feature_names)
    
    # 5.3. Scalling dengan StandardScaler yang sudah di-load
    input_scaled = scaler.transform(input_data)
    
    # 5.4. Lakukan Prediksi
    prediction = xgb_tuned.predict(input_scaled)[0]
    prediction_proba = xgb_tuned.predict_proba(input_scaled)[0]
    
    # Probabilitas kelas 1 (Churn)
    churn_risk = prediction_proba[1] * 100

# ==========================================
# 6. TAMPILKAN HASIL
# ==========================================
    st.divider()
    st.subheader("Hasil Prediksi")
    
    # Layout visual untuk hasil
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        # Menampilkan Gauge / Metric untuk Probabilitas
        st.metric(label="Risiko Churn", value=f"{churn_risk:.2f} %")
        
    with res_col2:
        if prediction == 1:
            st.error("🚨 **NASABAH BERISIKO TINGGI UNTUK CHURN**")
            st.write("Berdasarkan pola data, nasabah ini memiliki karakteristik yang sangat mirip dengan nasabah yang telah meninggalkan bank.")
            st.info("**Saran Tindakan:** Segera hubungi nasabah untuk menawarkan program retensi, tanyakan keluhan layanan, atau berikan penawaran khusus.")
        else:
            st.success("✅ **NASABAH DIPREDIKSI LOYAL (RETAINED)**")
            st.write("Nasabah ini kemungkinan besar akan tetap menggunakan layanan bank.")
            st.info("**Saran Tindakan:** Jaga kualitas layanan dan pertimbangkan untuk melakukan *cross-selling* produk lainnya.")