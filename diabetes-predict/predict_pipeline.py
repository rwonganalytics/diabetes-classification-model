"""
Este script carga el pipeline ya entrenado y realiza predicciones sobre los datos de prueba.
"""

import os
import pickle
from datetime import datetime
import shutil
import pandas as pd
import mlflow

def predict_pipeline():
    """
    Carga el pipeline entrenado y realiza predicciones sobre los datos de prueba.
    Las predicciones se almacenan en un archivo CSV y se registran en MLflow.
    """
    # Configuración de rutas
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_path, "data", "raw", "diabetes.csv")
    predictions_path = os.path.join(project_path, "data", "predictions")
    os.makedirs(predictions_path, exist_ok=True)

    # Carga del pipeline entrenado
    pipeline_path = os.path.join(project_path, "artifacts", "trainded_pipeline.pkl")
    with open(pipeline_path, 'rb') as f:
        trained_pipeline = pickle.load(f)

    # Carga de los datos de prueba
    test_data = pd.read_csv(data_path)

    # Realizar predicciones
    predictions = trained_pipeline.predict(test_data)

    # Guardar predicciones en un archivo CSV con nombre dinámico
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    predictions_file = os.path.join(predictions_path, f"predictions-{timestamp}.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_file, index=False)

    # Crear una copia del archivo con un nombre fijo para DVC
    predictions_file_static = os.path.join(predictions_path, "predictions.csv")
    shutil.copy(predictions_file, predictions_file_static)

    # Registro de predicciones en MLflow
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("Diabetes Predict Model")
    with mlflow.start_run(run_name=f"Predictions-{timestamp}"):
        mlflow.log_artifact(predictions_file)

    print(f"Predicciones guardadas en: {predictions_file}")
    print(f"Copia estática para DVC: {predictions_file_static}")
    print("Predicciones registradas en MLflow.")

if __name__ == "__main__":
    predict_pipeline()
