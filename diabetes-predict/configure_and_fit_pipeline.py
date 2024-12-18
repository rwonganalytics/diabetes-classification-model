"""
Este script define el mejor modelo de predicción y lo agrega al pipeline de ML
"""

import os
import pickle
import pandas as pd
import mlflow

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def configure_and_fit_pipeline():
    """
    Esta función define el mejor modelo de predicción y lo agrega al pipeline de ML
    """
    project_path = os.path.dirname(os.getcwd())
    dataset = pd.read_csv(os.path.join(project_path, "data", "raw", "diabetes.csv"))

    # configuración del Pipeline
    target = 'Outcome'
    mflow_url = 'http://127.0.0.1:5000'

    # division en train y test
    x_features = dataset
    y_target = dataset[target]
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2,
                                                        shuffle=True, random_state=42)

    # cargamos el pipeline previamente definido.
    with open(os.path.join(project_path, 'artifacts', 'pipeline.pkl'), 'rb') as f:
        diabetes_predict_model = pickle.load(f)

    # dataset de entrenamiento.
    x_features_train = diabetes_predict_model.fit_transform(x_train)

    # dataset para seleccion de modelo según métrica.
    x_features_test = diabetes_predict_model.transform(x_test)

    # configuracion de MLFlow
    mlflow.set_tracking_uri(mflow_url)
    mlflow.set_experiment('Diabetes Predict Model')

    # Definición de modelos y parámetros
    models_and_params = {
        "Model 1": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                                random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.1}
        },
        "Model 2": {
            "model": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, 
                                                random_state=42),
            "params": {"n_estimators": 50, "learning_rate": 0.1}
        },
        "Model 3": {
            "model": GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, 
                                                random_state=42),
            "params": {"n_estimators": 25, "learning_rate": 0.05}
        },
        "Model 4": {
            "model": KNeighborsClassifier(n_neighbors=3),
            "params": {"n_neighbors": 3}
        },
        "Model 5": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "params": {"n_neighbors": 5}
        },
        "Model 6": {
            "model": KNeighborsClassifier(n_neighbors=2),
            "params": {"n_neighbors": 2}
        },
        "Model 7": {
            "model": LogisticRegression(random_state=42, solver='lbfgs', max_iter=100),
            "params": {"solver": "lbfgs", "max_iter": 100}
        },
        "Model 8": {
            "model": LogisticRegression(random_state=42, solver='lbfgs', max_iter=200),
            "params": {"solver": "lbfgs", "max_iter": 200}
        },
        "Model 9": {
            "model": LogisticRegression(random_state=42, solver='liblinear', max_iter=100),
            "params": {"solver": "liblinear", "max_iter": 100}
        },
        "Model 10": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            "params": {"n_estimators": 100, "max_depth": None}
        },
        "Model 11": {
            "model": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
            "params": {"n_estimators": 200, "max_depth": None}
        },
        "Model 12": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10}
        },
        "Model 13": {
            "model": SVC(kernel='linear', random_state=42),
            "params": {"kernel": "linear"}
        },
        "Model 14": {
            "model": SVC(kernel='rbf', random_state=42),
            "params": {"kernel": "rbf"}
        },
        "Model 15": {
            "model": SVC(kernel='poly', random_state=42),
            "params": {"kernel": "poly"}
        }
    }

    # array para almacenar accuracy score de cada modelo
    resultados_acc = []

    # Iteramos sobre los diferentes modelos
    for model_name, model_info in models_and_params.items():
        with mlflow.start_run(run_name=model_name):
            model = model_info["model"]
            params = model_info["params"]

            # Entrenamiento del modelo
            model.fit(x_features_train, y_train)

            # Predicciones
            y_pred = model.predict(x_features_test)

            # Cálculo de métricas
            accuracy = accuracy_score(y_test, y_pred)

            # Registro de parámetros, métrica y modelos
            mlflow.log_params(params)
            mlflow.log_metric("accuracy score", accuracy)
            mlflow.sklearn.log_model(model, model_name)

            resultados_acc.append({"model_name": model_name, "accuracy_score": accuracy})

            mlflow.end_run()

    resultados_acc = pd.DataFrame(resultados_acc)

    # obtenemos modelo con mayor accuracy score
    best_model_name = resultados_acc.loc[resultados_acc["accuracy_score"].idxmax()]["model_name"]
    best_model = models_and_params[best_model_name]["model"]

    # agregamso paso de prediccion al pipeline
    diabetes_predict_model.steps.append(
                ('modelo_prediccion', best_model))

    # configuración y entrenamiento del modelo final
    diabetes_predict_model.fit(x_train, y_train)

    # modelo entrenado y configurado.
    with open(os.path.join(project_path, 'artifacts', 'trainded_pipeline.pkl'), 'wb') as f:
        pickle.dump(diabetes_predict_model, f)

configure_and_fit_pipeline()
