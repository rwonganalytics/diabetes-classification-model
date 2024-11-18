# %% [markdown]
# # Creación de modelos

# %%
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
# modelos
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# %% [markdown]
# ### Carga del dataset

# %%
dataset = pd.read_csv("../data/processed/features_for_model.csv")
dataset.head()

# %% [markdown]
# ### Dividir en train y test

# %%
# Dividir el dataset en X (características) y y (target)
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Dividir los datos en conjunto de entrenamiento y prueba (con los datos ya procesados)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificación
print(X_train.shape, X_test.shape)

# %% [markdown]
# ### Estandarización de variables

# %%
mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)

# %%
df_scaled = pd.DataFrame(mm_scaler.transform(X_train), columns=X_train.columns)
df_scaled.head()

# %% [markdown]
# #### Guardamos el Scaler como artefacto

# %%
with open("../artifacts/mm_scaler.pkl", "wb") as f:
    pickle.dump(mm_scaler, f)

# %% [markdown]
# ## Modelos de predicción

# %% [markdown]
# ### 1. Gradient Boost

# %%
# Crear el modelo
model_gb = GradientBoostingClassifier(random_state=42)

# Entrenar el modelo
model_gb.fit(X_train, y_train)

# Realizar predicciones
y_pred_gb = model_gb.predict(X_test)

# Evaluar el modelo
print("Gradient Boosting - Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Gradient Boosting - Classification Report:\n", classification_report(y_test, y_pred_gb))
print("Gradient Boosting - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))

# %%
# Probar diferentes configuraciones para Gradient Boosting Classifier
model_gb_1 = GradientBoostingClassifier(n_estimators=50, random_state=42)
model_gb_1.fit(X_train, y_train)
y_pred_gb_1 = model_gb_1.predict(X_test)
print("GB Config 1 - Accuracy:", accuracy_score(y_test, y_pred_gb_1))

model_gb_2 = GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, random_state=42)
model_gb_2.fit(X_train, y_train)
y_pred_gb_2 = model_gb_2.predict(X_test)
print("GB Config 2 - Accuracy:", accuracy_score(y_test, y_pred_gb_2))

# %%
# Guardar el modelo entrenado
with open('../models/gb_model.pkl', 'wb') as f:
    pickle.dump(model_gb_2, f)

# %% [markdown]
# ## 2. KN Neighbors

# %%
# Crear el modelo
model_knn = KNeighborsClassifier()

# Entrenar el modelo
model_knn.fit(X_train, y_train)

# Realizar predicciones
y_pred_knn = model_knn.predict(X_test)

# Evaluar el modelo
print("KNeighbors Classifier - Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNeighbors Classifier - Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNeighbors Classifier - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# %%
# Probar diferentes configuraciones para KNeighbors Classifier
model_knn_1 = KNeighborsClassifier(n_neighbors=5)
model_knn_1.fit(X_train, y_train)
y_pred_knn_1 = model_knn_1.predict(X_test)
print("KNN Config 1 - Accuracy:", accuracy_score(y_test, y_pred_knn_1))

model_knn_2 = KNeighborsClassifier(n_neighbors=2)
model_knn_2.fit(X_train, y_train)
y_pred_knn_2 = model_knn_2.predict(X_test)
print("KNN Config 2 - Accuracy:", accuracy_score(y_test, y_pred_knn_2))


# %%
with open('../models/knn_model.pkl', 'wb') as f:
    pickle.dump(model_knn, f)

# %% [markdown]
# ### 3. Logistic Regression

# %%
# Crear el modelo
model_lr = LogisticRegression(random_state=42)

# Entrenar el modelo
model_lr.fit(X_train, y_train)

# Realizar predicciones
y_pred_lr = model_lr.predict(X_test)

# Evaluar el modelo
print("Logistic Regression - Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression - Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# %%
# Probar diferentes configuraciones para Logistic Regression
model_lr_1 = LogisticRegression(random_state=42, max_iter=200)
model_lr_1.fit(X_train, y_train)
y_pred_lr_1 = model_lr_1.predict(X_test)
print("Logistic Regression Config 1 - Accuracy:", accuracy_score(y_test, y_pred_lr_1))

model_lr_2 = LogisticRegression(random_state=42, solver='liblinear')
model_lr_2.fit(X_train, y_train)
y_pred_lr_2 = model_lr_2.predict(X_test)
print("Logistic Regression Config 2 - Accuracy:", accuracy_score(y_test, y_pred_lr_2))

# %%
# Guardar el modelo entrenado
with open('../models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model_lr, f)

# %% [markdown]
# ### 4. Random Forest

# %%
# Crear el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
# Configuración 1: Sin cambios adicionales
model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)
print("Config 1 - Accuracy:", accuracy_score(y_test, y_pred_1))

# Configuración 2: Aumentar los estimadores
model_2 = RandomForestClassifier(n_estimators=200, random_state=42)
model_2.fit(X_train, y_train)
y_pred_2 = model_2.predict(X_test)
print("Config 2 - Accuracy:", accuracy_score(y_test, y_pred_2))

# Configuración 3: Cambiar la profundidad máxima de los árboles
model_3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_3.fit(X_train, y_train)
y_pred_3 = model_3.predict(X_test)
print("Config 3 - Accuracy:", accuracy_score(y_test, y_pred_3))


# %%
# Guardar el modelo entrenado
with open('../models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model_1, f)

# %% [markdown]
# ### 5. SVC

# %%
# Crea el modelo
model_svc = SVC(random_state=42)

# Entrenar el modelo
model_svc.fit(X_train, y_train)

# Realiza predicciones
y_pred_svc = model_svc.predict(X_test)

# Evaluar el modelo
print("Support Vector Classifier - Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Support Vector Classifier - Classification Report:\n", classification_report(y_test, y_pred_svc))
print("Support Vector Classifier - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))

# %%
# Probar diferetnes configuraciones pra Support Vector Classifier
model_svc_1 = SVC(kernel='linear', random_state=42)
model_svc_1.fit(X_train, y_train)
y_pred_svc_1 = model_svc_1.predict(X_test)
print("SVC Config 1 - Accuracy:", accuracy_score(y_test, y_pred_svc_1))

model_svc_2 = SVC(kernel='rbf', random_state=42)
model_svc_2.fit(X_train, y_train)
y_pred_svc_2 = model_svc_2.predict(X_test)
print("SVC Config 2 - Accuracy:", accuracy_score(y_test, y_pred_svc_2))


# %%
# Guardar el modelo entrenado
with open('../models/svc_model.pkl', 'wb') as f:
    pickle.dump(model_svc_1, f)


