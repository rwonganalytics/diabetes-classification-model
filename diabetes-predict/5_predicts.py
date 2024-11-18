# %% [markdown]
# # Predicts

# %%
import pandas as pd
import numpy as np
import pickle

# %%
data_test = pd.read_csv("../data/processed/df_test.csv")
data_test.head()

# %% [markdown]
# Verificamos que no haya nulos

# %%
data_test.isnull().mean()

# %% [markdown]
# Eliminamos columnas

# %%
data_test.drop(['SkinThickness', 'Insulin'], axis=1, inplace=True)

# %% [markdown]
# Imputación de datos

# %%
with open("../artifacts/feature_eng_configs.pkl","rb") as f:
    feature_eng_configs = pickle.load(f)

feature_eng_configs

# %%
data_test["Glucose"] = data_test["Glucose"].replace(0, feature_eng_configs["Glucose_imputed_value"]).astype(int)
data_test["BloodPressure"] = data_test["BloodPressure"].replace(0, feature_eng_configs["BloodPressure_imputed_value"]).astype(int)
data_test["BMI"] = data_test["BMI"].replace(0, feature_eng_configs["BMI_imputed_value"]).astype(int)

data_test.head(10)

# %% [markdown]
# Estandarización con artefacto guardado

# %%
with open("../artifacts/mm_scaler.pkl", "rb") as f:
    mm_scaler = pickle.load(f)

mm_scaler

# %%
X_data_test_std = mm_scaler.transform(data_test)

# %% [markdown]
# Cargamos modelos ya entrenados

# %%
with open("../models/gb_model.pkl", "rb") as f:
    modelo_gb = pickle.load(f)

modelo_gb

# %%
data_test_predicts_gb = modelo_gb.predict(X_data_test_std)
data_test_predicts_gb

# %%
with open("../models/knn_model.pkl", "rb") as f:
    modelo_knn = pickle.load(f)

modelo_knn

# %%
data_test_predicts_knn = modelo_knn.predict(X_data_test_std)
data_test_predicts_knn

# %%
with open("../models/logistic_regression_model.pkl", "rb") as f:
    modelo_lr = pickle.load(f)

modelo_lr

# %%
data_test_predicts_lr = modelo_lr.predict(X_data_test_std)
data_test_predicts_lr

# %%
with open("../models/random_forest_model.pkl", "rb") as f:
    modelo_rf = pickle.load(f)

modelo_rf

# %%
data_test_predicts_rf = modelo_rf.predict(X_data_test_std)
data_test_predicts_rf

# %%
with open("../models/svc_model.pkl", "rb") as f:
    modelo_svc = pickle.load(f)

modelo_svc

# %%
data_test_predicts_svc = modelo_rf.predict(X_data_test_std)
data_test_predicts_svc


