import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURACIÓN INICIAL DE MLFLOW ---
# Asegúrate de que esta ruta coincida con la configuración de tu entorno
MLFLOW_TRACKING_URI = "mlflow-deploy\mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("diabetes-logistic-model")


def prepare_data():
    """Carga los datos y binariza el target para la clasificación."""
    X, y = load_diabetes(return_X_y=True)
    
    # La Regresión Logística es para CLASIFICACIÓN.
    # El dataset de Diabetes es de REGRESIÓN.
    # Para poder usar LogisticRegression, binarizamos el target (y) con un umbral.
    # Si la progresión es superior a la mediana, es clase 1; sino, es clase 0.
    median_y = np.median(y)
    y_binary = (y > median_y).astype(int)
    
    # División de datos
    return train_test_split(X, y_binary, test_size=0.2, random_state=42)


def train_and_log_logistic_model():
    """Entrena el modelo de Regresión Logística y registra los resultados en MLflow."""
    
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Definición de hiperparámetros del modelo
    params = {
        "solver": "liblinear", 
        "C": 0.1,  # Parámetro de regularización (ejemplo)
        "random_state": 42
    }
    
    # --- 2. INICIO DEL MLFLOW RUN ---
    with mlflow.start_run():
        
        # 2.1. Entrenamiento
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        
        # 2.2. Predicción y Métricas
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy (Regresión Logística): {accuracy:.4f}")
        
        # 2.3. Registro en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        # Registra el modelo completo. "model" es la carpeta de artefactos
        mlflow.sklearn.log_model(lr, "model")
        
        print(f"Modelo 'LogisticRegression' registrado con éxito en MLflow.")


if __name__ == "__main__":
    train_and_log_logistic_model()