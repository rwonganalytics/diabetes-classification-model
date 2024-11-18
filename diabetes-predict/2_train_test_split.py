# %% [markdown]
# # Train - Test Split

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 1. Carga del dataset original

# %%
df = pd.read_csv('../data/raw/diabetes.csv')
df.head(10)

# %% [markdown]
# ## 2. División en train y test

# %%
X = df.drop('Outcome', axis=1)  
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
df_train = pd.concat([X_train, y_train], axis=1)
df_train.head(10)

# %% [markdown]
# ## 3. Guardamos cada dataset (train y test)

# %%
df_train.to_csv('../data/processed/df_train.csv', index=False)

# %%
X_test.to_csv('../data/processed/df_test.csv', index=False)


