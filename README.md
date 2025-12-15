# 👶 Stunting Prediction MLOps Project

![CI/CD Pipeline](https://github.com/sains-data/Stunting_MLOps/actions/workflows/main.yml/badge.svg)

Proyek ini adalah implementasi *End-to-End Machine Learning Operations (MLOps)* untuk mendeteksi status stunting pada balita. Proyek ini mencakup pipeline mulai dari *Data Ingestion*, *Preprocessing*, *Model Training*, *Experiment Tracking*, hingga *Model Serving* menggunakan API.

Tugas Besar Mata Kuliah *Machine Learning Operations* Sains Data - ITERA 2025.

---

## 📋 Machine Learning Canvas

| Bagian | Deskripsi |
| :--- | :--- |
| **🔍 Background** | Stunting adalah gangguan pertumbuhan kronis pada anak akibat kurang gizi. Deteksi dini seringkali terlambat karena kurangnya alat bantu prediksi cepat di tingkat Posyandu. |
| **💎 Value Proposition** | Menyediakan API prediksi otomatis yang dapat membantu tenaga kesehatan/orang tua mengetahui status gizi balita secara *real-time* dan akurat. |
| **🎯 Objectives** | Membangun model klasifikasi dengan akurasi tinggi dan men-deploy-nya ke sistem produksi yang terotomatisasi. |
| **🧠 Solution** | Menggunakan algoritma **Support Vector Machine (SVM)** dengan pipeline preprocessing otomatis (OneHotEncoding & Scaling). |
| **📊 Data** | Dataset Balita yang mencakup fitur: `Umur (bulan)`, `Jenis Kelamin`, `Tinggi Badan (cm)`. Target: `Status Gizi`. |
| **📈 Metrics** | Evaluasi model menggunakan **Accuracy**, **Precision**, dan **Recall** untuk meminimalkan kesalahan deteksi. |
| **🚀 Inference** | **Online Inference** menggunakan REST API (FastAPI). |

---

## 🛠️ Tech Stack & Tools

* **Bahasa Pemrograman:** Python 3.9+
* **Data & Modeling:** Pandas, Scikit-Learn
* **Experiment Tracking:** MLFlow (Mencatat metrik akurasi & parameter tiap training)
* **Command Line Interface (CLI):** Typer
* **API / Model Serving:** FastAPI & Uvicorn
* **Version Control:** Git & GitHub
* **CI/CD Automation:** GitHub Actions (Automated Testing)

---

## 📂 Struktur Project

```text
├── .github/workflows/   # Konfigurasi CI/CD (GitHub Actions)
├── data/                # Dataset (data_balita.csv)
├── models/              # Tempat penyimpanan model (.pkl)
├── src/                 # Source Code utama
│   ├── app.py           # Kode untuk API / Deployment (FastAPI)
│   └── train.py         # Kode untuk Training & Experiment Tracking (MLFlow)
├── requirements.txt     # Daftar library yang dibutuhkan
├── Dockerfile           # Konfigurasi Docker Build
└── README.md            # Dokumentasi Proyek
```

---

##requirement.txt
[cite_start]pandas [cite: 1]
[cite_start]scikit-learn [cite: 1]
[cite_start]mlflow [cite: 1]
[cite_start]typer [cite: 1]
[cite_start]fastapi [cite: 1]
[cite_start]uvicorn [cite: 1]
[cite_start]joblib [cite: 1]
[cite_start]pydantic [cite: 1]

##dockerfile
# Gunakan base image Python yang ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file kode, model, dan data
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/ /app/data/

# Tentukan command untuk menjalankan API server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

## 🚀 Cara Menjalankan Project (Reproducibility)

Ikuti langkah ini untuk menjalankan proyek di komputer lokal Anda:

### 1️⃣ Clone Repository
```bash
Copy code
git clone https://github.com/sains-data/Stunting_MLOps.git
cd Stunting_MLOps
```

### 2️⃣ Install Dependencies

Disarankan menggunakan virtual environment:

```bash
Copy code
pip install -r requirements.txt
```

### 3️⃣ Training Model (Build Pipeline)

Jalankan perintah berikut untuk melatih model. Hasil eksperimen akan tercatat oleh MLFlow, dan model akan disimpan di folder models/.

```bash
Copy code
python src/train.py
Output: Model tersimpan sebagai models/model_stunting.pkl.
```

### 4️⃣ Menjalankan API (Deployment)

Aktifkan server FastAPI untuk melakukan prediksi:

```bash
Copy code
uvicorn src.app:app --reload
Server akan berjalan di: http://127.0.0.1:8000
```

### 5️⃣ Uji Coba (Testing)

Buka Swagger UI untuk mencoba prediksi secara interaktif: http://127.0.0.1:8000/docs

Contoh Input JSON:

```json
{
  "umur_bulan": 24,
  "jenis_kelamin": "Laki-laki",
  "tinggi_badan": 85.5
}
```

---

## 🤖 CI/CD Automation

Proyek ini telah dilengkapi dengan GitHub Actions. Setiap kali ada push ke branch main, sistem akan otomatis:

* Menginstall environment Python.
* Menginstall dependencies.
* Menjalankan tes training untuk memastikan kode tidak error.
* Status build terakhir dapat dilihat di badge di atas.

------

## 🔗 Publikasi dan Deployment Eksternal

Bagian ini menyoroti hasil deployment dan otomatisasi yang telah diterapkan sesuai standar MLOps:

* **Hugging Face Space (Deployment Production):**
    Model telah di-deploy dan dapat diakses publik melalui link berikut. Anda dapat melakukan pengujian API secara langsung pada URL ini (FastAPI Swagger UI).
    
    > **[Hugging Face API Docs]**
    > **https://dwiratna-prediksi-stunting.hf.space/docs** 

* **GitHub Actions:**
    Status CI/CD (Continuous Integration/Continuous Deployment) dan pengujian otomatis dapat dipantau melalui *badge* di bagian atas *README*.

---

## 👥 Tim Pengembang

Proyek ini dikerjakan oleh tim dari Mata Kuliah MLOps ITERA 2025:

* Siti Nur Aarifah (122450006)
* Dwi Ratna Anggareni (122450008)
* Cyntia Kristina Sidauruk (122450023)
* Priska Silvi Ferantiana (122450053)

***Created for Project MLOps 2025***
