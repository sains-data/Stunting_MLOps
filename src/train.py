import pandas as pd
import typer
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

app = typer.Typer()

@app.command()
def train(
    data_path: str = "data/data_balita.csv", 
    model_path: str = "models/model_stunting.pkl",
    kernel: str = "rbf",
    c_param: float = 1.0
):
    # 1. Load Data
    print("⏳ Memuat data...")
    df = pd.read_csv(data_path)
    
    # PERBAIKAN DI SINI: Sesuaikan nama kolom dengan dataset asli
    X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
    y = df['Status Gizi'] # <-- Ganti dari 'Status Stunting' ke 'Status Gizi'

    # 2. Preprocessing Pipeline
    numeric_features = ['Umur (bulan)', 'Tinggi Badan (cm)']
    categorical_features = ['Jenis Kelamin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # 3. Setup Model & MLFlow
    mlflow.set_experiment("Eksperimen Stunting")
    
    with mlflow.start_run():
        print("🚀 Sedang training model...")
        
        # Pipeline: Preprocessing + Model SVM
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', SVC(kernel=kernel, C=c_param))])
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluasi
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"✅ Training Selesai! Akurasi: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

        # 4. Log Metrics ke MLFlow
        mlflow.log_param("kernel", kernel)
        mlflow.log_param("C", c_param)
        mlflow.log_metric("accuracy", accuracy)
        
        # 5. Simpan Model
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "model")
        print(f"💾 Model disimpan di: {model_path}")

if __name__ == "__main__":
    app()