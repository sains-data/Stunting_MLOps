from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# 1. Inisialisasi Aplikasi
app = FastAPI(title="API Prediksi Stunting", description="MLOps Project ITERA 2025")

# 2. Load Model yang sudah ditrain
model = joblib.load("models/model_stunting.pkl")

# 3. Definisikan Format Data Input (Validasi Data)
class DataBalita(BaseModel):
    umur_bulan: int
    jenis_kelamin: str  # 'Laki-laki' atau 'Perempuan'
    tinggi_badan: float

@app.get("/")
def home():
    return {"message": "Selamat Datang di API Prediksi Stunting!"}

@app.post("/predict")
def predict_stunting(data: DataBalita):
    # Ubah data input user menjadi DataFrame (agar mirip format CSV training)
    df_input = pd.DataFrame([{
        'Umur (bulan)': data.umur_bulan,
        'Jenis Kelamin': data.jenis_kelamin,
        'Tinggi Badan (cm)': data.tinggi_badan
    }])

    # Lakukan Prediksi
    # Pipeline akan otomatis mengurus scaling & encoding
    prediction = model.predict(df_input)
    
    return {
        "status": "Sukses",
        "input_data": data,
        "hasil_prediksi": prediction[0]  # Misal: "Sangat Pendek" atau "Normal"
    }