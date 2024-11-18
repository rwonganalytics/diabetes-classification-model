"""
modulo para producción del modelo, genera predicciones con modelos entrenados
"""
import pickle
import pandas as pd

def model_predicts():
    """
    funcion para generar predicciones con data nueva
    """
    data_test = pd.read_csv("../data/processed/df_test.csv")

    # Eliminamos columnas
    data_test.drop(['SkinThickness', 'Insulin'], axis=1, inplace=True)

    # Imputación de datos
    with open("../artifacts/feature_eng_configs.pkl","rb") as f:
        feature_eng_configs = pickle.load(f)

    data_test["Glucose"] = data_test["Glucose"].replace(0,
                                feature_eng_configs["Glucose_imputed_value"]).astype(int)
    data_test["BloodPressure"] = data_test["BloodPressure"].replace(0,
                                feature_eng_configs["BloodPressure_imputed_value"]).astype(int)
    data_test["BMI"] = data_test["BMI"].replace(0,
                                feature_eng_configs["BMI_imputed_value"]).astype(int)

    # Estandarización con artefacto guardado
    with open("../artifacts/mm_scaler.pkl", "rb") as f:
        mm_scaler = pickle.load(f)

    X_data_test_std = mm_scaler.transform(data_test)

    # gradient boost
    with open("../models/gb_model.pkl", "rb") as f:
        modelo_gb = pickle.load(f)
    data_test_predicts_gb = modelo_gb.predict(X_data_test_std)
    # knn
    with open("../models/knn_model.pkl", "rb") as f:
        modelo_knn = pickle.load(f)
    data_test_predicts_knn = modelo_knn.predict(X_data_test_std)
    # logistic regression
    with open("../models/logistic_regression_model.pkl", "rb") as f:
        modelo_lr = pickle.load(f)
    data_test_predicts_lr = modelo_lr.predict(X_data_test_std)
    # random forest
    with open("../models/random_forest_model.pkl", "rb") as f:
        modelo_rf = pickle.load(f)
    data_test_predicts_rf = modelo_rf.predict(X_data_test_std)
    # svc
    with open("../models/svc_model.pkl", "rb") as f:
        modelo_svc = pickle.load(f)
    data_test_predicts_svc = modelo_svc.predict(X_data_test_std)
    predicts = [data_test_predicts_gb, data_test_predicts_knn, data_test_predicts_lr,
                data_test_predicts_rf, data_test_predicts_svc]
    return predicts
