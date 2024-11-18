import pandas as pd
import numpy as np
import pickle

def create_model_features():
    df = pd.read_csv('../data/processed/df_test.csv')

    # Eliminamos variables con número significativo de faltantes
    df.drop(['SkinThickness', 'Insulin'], axis=1, inplace=True)

    # Imputación de variables
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

    # Guardamos dataset procesado
    df.to_csv('../data/processed/features_for_model.csv', index=False)

