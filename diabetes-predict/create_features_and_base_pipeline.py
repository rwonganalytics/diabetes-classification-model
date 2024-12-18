"""
Este script crea las características para el modelo de machine learning.
"""

import pickle
import pandas as pd

from feature_engine.imputation import MeanMedianImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def create_features_and_base_pipeline():
    """
    Esta función genera las características para el modelo de ML.
    """
    # Carga del dataset
    dataset = pd.read_csv("../data/raw/diabetes.csv")

    # Configuración del pipeline
    target = "Outcome"
    vars_to_drop = ["SkinThickness", "Insulin", target]
    continuous_vars_to_impute = ["Glucose", "BloodPressure", "BMI"]

    # Eliminación de variables no deseadas
    x_features = dataset.drop(labels=vars_to_drop, axis=1)
    y_target = dataset[target]

    # Separación del dataset en train y test
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_target, test_size=0.2, shuffle=True, random_state=42
    )

    # Creación del pipeline
    diabetes_predict_pipeline = Pipeline([
        # Imputación de variables continuas
        ("continuous_var_mean_imputation", MeanMedianImputer(
            imputation_method="mean", variables=continuous_vars_to_impute
        )),
        # Estandarización de variables
        ("feature_scaling", MinMaxScaler())
    ])

    # Ajustamos usando solo datos de train
    diabetes_predict_pipeline.fit(x_train)
    x_features_processed = diabetes_predict_pipeline.transform(x_train)
    df_features_processed = pd.DataFrame(x_features_processed, columns=x_train.columns)
    df_features_processed[target] = y_train.reset_index(drop=True)

    # Guardamos los datos procesados para entrenar modelos
    df_features_processed.to_csv("../data/processed/features_for_model.csv", index=False)

    # Guardamos el pipeline
    with open("../artifacts/pipeline.pkl", "wb") as f:
        pickle.dump(diabetes_predict_pipeline, f)

    # Guardamos dataset de test
    df_test = pd.DataFrame(x_test, columns=x_test.columns)
    df_test[target] = y_test
    df_test.to_csv("../data/processed/test_dataset.csv", index=False)
