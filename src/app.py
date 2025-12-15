import pandas as pd
import joblib
from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel

# 1. Load Model (Sama seperti sebelumnya)
model_path = "models/model_stunting.pkl"
try:
    model = joblib.load(model_path)
    print("✅ Model berhasil dimuat.")
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    model = None

# 2. Inisialisasi FastAPI
app = FastAPI(title="Prediksi Stunting API")

# --- BAGIAN API (Untuk Tugas MLOps) ---
class BalitaData(BaseModel):
    umur_bulan: int
    jenis_kelamin: str  # "Laki-laki" atau "Perempuan"
    tinggi_badan: float

@app.get("/")
def home():
    return {"message": "Halo! Buka /docs untuk API atau langsung gunakan form di bawah."}

@app.post("/predict")
def predict_api(data: BalitaData):
    # Logika prediksi untuk API
    df_input = pd.DataFrame([{
        'Umur (bulan)': data.umur_bulan,
        'Jenis Kelamin': data.jenis_kelamin,
        'Tinggi Badan (cm)': data.tinggi_badan
    }])
    prediction = model.predict(df_input)
    return {"status_gizi": prediction[0]}

# --- BAGIAN UI (GRADIO - Untuk Tampilan Cantik) ---
def prediksi_gradio(umur, gender, tinggi):
    # Fungsi ini khusus dipanggil oleh tampilan web
    df_input = pd.DataFrame([{
        'Umur (bulan)': int(umur),
        'Jenis Kelamin': gender,
        'Tinggi Badan (cm)': float(tinggi)
    }])
    try:
        hasil = model.predict(df_input)[0]
        return f"Hasil Prediksi: {hasil}"
    except Exception as e:
        return f"Error: {str(e)}"

# Membuat Layout Tampilan
ui = gr.Interface(
    fn=prediksi_gradio,
    inputs=[
        gr.Number(label="Umur (bulan)", value=12),
        gr.Radio(["Laki-laki", "Perempuan"], label="Jenis Kelamin"),
        gr.Number(label="Tinggi Badan (cm)", value=75.0)
    ],
    outputs="text",
    title="Aplikasi Deteksi Stunting 👶",
    description="Masukkan data balita untuk melihat prediksi status gizi."
)

# 3. TEMPELKAN UI KE FASTAPI
# UI akan muncul di alamat utama, API tetap jalan di background
app = gr.mount_gradio_app(app, ui, path="/")