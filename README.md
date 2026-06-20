# Bank Customer Churn Prediction & Retention Strategy

---

## Project Background
Dalam industri perbankan, biaya untuk mengakuisisi nasabah baru jauh lebih mahal dibandingkan mempertahankan nasabah yang sudah ada. Kehilangan nasabah (*churn*) berdampak langsung pada penurunan profitabilitas. Oleh karena itu, penting bagi pihak bank untuk dapat memprediksi nasabah mana yang berniat pergi sebelum hal itu benar-benar terjadi agar intervensi yang tepat dapat diberikan.

## Problem Statement
* **Masalah:** Tingkat *churn* nasabah yang tidak terdeteksi secara dini menyebabkan hilangnya potensi pendapatan berkelanjutan bagi bank.
* **Tujuan:** Membangun model prediktif klasifikasi untuk mendeteksi nasabah yang berisiko *churn* dengan metrik evaluasi yang berfokus pada **Recall** (untuk meminimalkan jumlah nasabah *churn* yang gagal terdeteksi oleh sistem).

---

## Data Understanding
* **Sumber Data:** Dataset "Bank Customer Churn Prediction" dari Kaggle.
* **Karakteristik Data:** Terdiri dari 10.000 baris histori nasabah dengan 14 kolom. Target label adalah `Exited` (1 = Churn, 0 = Retained).
* **Kualitas Data:** Dataset dalam kondisi sangat bersih (0 *missing values* dan 0 data duplikat).

## Exploratory Data Analysis (EDA)
Temuan utama dari proses eksplorasi data:
1. **Target Imbalance:** Terdapat ketidakseimbangan data di mana tingkat *churn* keseluruhan adalah 20.4%.
2. **Distribusi Usia (Age):** Nasabah berusia 40-50 tahun memiliki rasio *churn* yang paling tinggi.
3. **Keaktifan (IsActiveMember):** Nasabah yang pasif/tidak aktif secara signifikan lebih rentan untuk meninggalkan bank.
4. **Jumlah Produk (NumOfProducts):** Nasabah yang memiliki 3 atau 4 produk bank memiliki probabilitas *churn* mendekati 100%, mengindikasikan bahwa memiliki terlalu banyak produk bank justru memberatkan nasabah.
5. **Saldo (Balance):** Churn lebih banyak terkonsentrasi pada nasabah dengan saldo menengah ke atas (100k-150k), sedangkan nasabah bersaldo 0 cenderung lebih loyal.

## Machine Learning Process
* **Preprocessing:** Menghapus kolom identifier yang tidak relevan (`RowNumber`, `CustomerId`, `Surname`). Menerapkan *One-Hot Encoding* untuk data kategorikal, melakukan *SMOTE* untuk menyeimbangkan data dan *StandardScaler* untuk menormalisasi distribusi fitur numerik.
* **Modeling:** Menggunakan algoritma *tree-based* tingkat lanjut, **XGBoost Classifier**.
* **Evaluasi:** Model di-tune untuk memaksimalkan *Recall* kelas positif (churn), karena tujuan utama bisnis adalah "menangkap" sebanyak mungkin nasabah yang berisiko pergi.
* **Feature Importance:** Fitur `Age`, `IsActiveMember`, dan `NumOfProducts` menjadi penentu paling vital bagi model dalam melakukan klasifikasi.

---

## Conclusion & Recommendation
Berdasarkan hasil pemodelan dan EDA, berikut adalah rekomendasi strategi retensi:
1. **Targeted Loyalty Program:** Rancang paket loyalitas atau layanan premium khusus untuk demografi nasabah berusia 40-50 tahun.
2. **Re-engagement Campaign:** Kirimkan promosi (seperti *cashback* atau bebas biaya admin) kepada nasabah yang mulai terdeteksi sebagai *inactive member*.
3. **Evaluasi Produk:** Tinjau ulang beban biaya atau kompleksitas nasabah yang memiliki lebih dari 2 produk bank, tawarkan konsolidasi produk untuk mempermudah pengalaman finansial mereka.

---

## Project Structure
```
bank_customer_churn_prediction/
│
├── assets/                  # Folder menyimpan grafik EDA, scaler, & XGBoost model
├── data/                    # Dataset mentah (diabaikan oleh gitignore)
├── notebook/                # Jupyter notebook untuk proses EDA & ML Training
├── app.py                   # Script utama aplikasi Streamlit
├── requirements.txt         # Daftar dependencies library Python
├── README.md                # Dokumentasi proyek
└── .gitignore               # Konfigurasi file yang diabaikan oleh Git
```

---

## How to Run Locally
1. **Clone Repository**
```bash
git clone https://github.com/hyusam/bank-churn-prediction.git
cd bank_customer_churn_prediction
```
2. **Buat Virtual Environment**
```bash
python -m venv .venv
source venv/Scripts/activate  # Untuk Windows
# atau
source venv/bin/activate      # Untuk Mac/Linux
```
3. **Install Requirements**
```bash
pip install -r requirements.txt
```
4. **Jalankan Aplikasi Streamlit**
```bash
streamlit run app.py
```