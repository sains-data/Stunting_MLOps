import pandas as pd
import typer
import joblib
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Inisialisasi Aplikasi CLI
app = typer.Typer()

def load_and_merge_data(original_path: str, new_data_path: str):
    """
    Fungsi ini menggabungkan data dataset asli dengan data baru dari monitoring (log).
    """
    # 1. Load Data Asli
    print(f"[INFO] Memuat data asli dari: {original_path}")
    try:
        df_old = pd.read_csv(original_path)
    except FileNotFoundError:
        print(f"[ERROR] File {original_path} tidak ditemukan!")
        return pd.DataFrame()

    # 2. Cek apakah ada Data Baru (Log)
    if os.path.exists(new_data_path):
        print(f"[INFO] Ditemukan data baru di: {new_data_path}. Sedang menggabungkan...")
        try:
            df_new = pd.read_csv(new_data_path)
            
            # Mapping nama kolom dari Log (Inggris/kecil) ke Dataset Asli (Indonesia/Kapital)
            rename_mapping = {
                "umur_bulan": "Umur (bulan)",
                "jenis_kelamin": "Jenis Kelamin",
                "tinggi_badan": "Tinggi Badan (cm)",
                "prediksi": "Status Gizi" 
            }
            df_new = df_new.rename(columns=rename_mapping)
            
            # Hapus kolom timestamp karena tidak dipakai training
            if "timestamp" in df_new.columns:
                df_new = df_new.drop(columns=["timestamp"])
            
            # Pastikan hanya kolom yang relevan yang diambil
            required_cols = ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)", "Status Gizi"]
            # Filter hanya kolom yang ada
            df_new = df_new[df_new.columns.intersection(required_cols)]
            
            # Gabungkan (Concatenate)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            print(f"[SUCCESS] Berhasil digabung! Total data sekarang: {len(df_final)} baris.")
            return df_final
            
        except Exception as e:
            print(f"[WARNING] Gagal menggabungkan data baru: {e}")
            return df_old
    else:
        print("[INFO] Tidak ada data baru ditemukan. Menggunakan data asli saja.")
        return df_old

@app.command()
def train(
    data_path: str = "data/data_balita.csv", 
    log_path: str = "monitoring_log.csv", 
    model_path: str = "models/model_stunting.pkl",
    kernel: str = "rbf",
    c_param: float = 1.0
):
    # 1. Load & Merge Data
    df = load_and_merge_data(data_path, log_path)
    
    if df.empty:
        print("[ERROR] Data kosong. Training dibatalkan.")
        return

    # Hitung total data untuk dilaporkan
    TOTAL_DATA = len(df)

    # Pisahkan Fitur (X) dan Target (y)
    try:
        X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
        y = df['Status Gizi']
    except KeyError as e:
        print(f"[ERROR] Kolom Tidak Ditemukan: {e}")
        return

    # 2. Preprocessing Pipeline
    numeric_features = ['Umur (bulan)', 'Tinggi Badan (cm)']
    categorical_features = ['Jenis Kelamin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 3. Setup Model & MLFlow
    mlflow.set_experiment("FIX_FINAL_TEST")
    
    with mlflow.start_run():
        print("[START] Sedang training model...")
        
        # Buat Pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', SVC(kernel=kernel, C=c_param))])
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"[SUCCESS] Training Selesai! Akurasi: {accuracy:.4f}")
        print(f"[DEBUG] Data Count yang dikirim ke MLFlow: {TOTAL_DATA}")
        
        # --- LOGGING KE MLFLOW ---
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("C", c_param)
        
        # INI PARAMETER YANG KAMU CARI
        mlflow.log_param("data_count", TOTAL_DATA) 
        
        mlflow.log_metric("accuracy", accuracy)
        
        # Simpan Model
        joblib.dump(pipeline, model_path)
        print(f"[SAVED] Model diperbarui dan disimpan di: {model_path}")

if __name__ == "__main__":
    app()