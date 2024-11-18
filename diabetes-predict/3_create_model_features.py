# %% [markdown]
# # Ingeniería de Características

# %%
import pandas as pd
import numpy as np
import pickle

# %% [markdown]
# ## 1. Cargamos dataset

# %%
df = pd.read_csv('../data/processed/df_test.csv')
df.head(10)

# %% [markdown]
# ## 2. Eliminamos variables con muchos faltantes

# %%
df.drop(['SkinThickness', 'Insulin'], axis=1, inplace=True)
df.head()

# %% [markdown]
# ## 3. Ingeniería de características

# %% [markdown]
# ### 3.1 Imputación de variables

# %%
proporcion_ceros = (df == 0).mean()
proporcion_ceros

# %%
cols_imputacion = ["Glucose", "BloodPressure", "BMI"]

feature_eng_configs = {}

for col in cols_imputacion:
    media = int(df[col].mean())
    imputed_key = f"{col}_imputed_value"  
    feature_eng_configs[imputed_key] = media
    df[col] = df[col].replace(0, media).astype(int)

# %% [markdown]
# ### Guardamos valores imputados como artefacto

# %%
with open("../artifacts/feature_eng_configs.pkl", "wb") as f:
    pickle.dump(feature_eng_configs, f)

# %% [markdown]
# ## 5. Guardamos dataset procesado

# %%
df.to_csv('../data/processed/features_for_model.csv', index=False)

# %%
df


