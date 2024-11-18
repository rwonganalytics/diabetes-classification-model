import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

def create_model_features():
    # cargamos dataset
    df = pd.read_csv('../data/processed/df_test.csv')

    # eliminamos variables con muchos faltantes
    df.drop(['SkinThickness', 'Insulin'], axis=1, inplace=True)

    # imputación de variables
    cols_imputacion = ["Glucose", "BloodPressure", "BMI"]
    feature_eng_configs = {}
    for col in cols_imputacion:
        media = int(df[col].mean())
        imputed_key = f"{col}_imputed_value"  
        feature_eng_configs[imputed_key] = media
        df[col] = df[col].replace(0, media).astype(int)

    # Guardamos valores imputados como artefacto
      with open("../artifacts/feature_eng_configs.pkl", "wb") as f:
        pickle.dump(feature_eng_configs, f)

    # Estandarización de variables
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(df)
    df_scaled = pd.DataFrame(mm_scaler.transform(df), columns=df.columns)

    # Guardamos el Scaler como artefacto
    with open("../artifacts/mm_scaler.pkl", "wb") as f:
        pickle.dump(mm_scaler, f)

    # Guardamos dataset procesado
    df_scaled.to_csv('../data/processed/features_for_model.csv', index=False)


