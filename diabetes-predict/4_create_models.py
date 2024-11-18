
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def model_creation():
    dataset = pd.read_csv("../data/processed/features_for_model.csv")

    # Dividir el dataset en X (características) y y (target)
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']

    # Dividir los datos en conjunto de entrenamiento y prueba (con los datos ya procesados)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarización de variables
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X_train)

    X_train = pd.DataFrame(mm_scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(mm_scaler.transform(X_test), columns=X_test.columns)

    # Guardamos el Scaler como artefacto
    with open("../artifacts/mm_scaler.pkl", "wb") as f:
        pickle.dump(mm_scaler, f)

    # Modelos de predicción
    # 1.Gradient Boost
    model_gb = GradientBoostingClassifier(random_state=42)
    model_gb.fit(X_train, y_train)
    y_pred_gb = model_gb.predict(X_test)

    model_gb_1 = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model_gb_1.fit(X_train, y_train)
    y_pred_gb_1 = model_gb_1.predict(X_test)

    model_gb_2 = GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, random_state=42)
    model_gb_2.fit(X_train, y_train)
    y_pred_gb_2 = model_gb_2.predict(X_test)

    with open('../models/gb_model.pkl', 'wb') as f:
        pickle.dump(model_gb_2, f)

    # 2. KN Neighbors
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)

    model_knn_1 = KNeighborsClassifier(n_neighbors=5)
    model_knn_1.fit(X_train, y_train)
    y_pred_knn_1 = model_knn_1.predict(X_test)

    model_knn_2 = KNeighborsClassifier(n_neighbors=2)
    model_knn_2.fit(X_train, y_train)
    y_pred_knn_2 = model_knn_2.predict(X_test)

    with open('../models/knn_model.pkl', 'wb') as f:
        pickle.dump(model_knn, f)

    # 3. Logistic Regression
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    model_lr_1 = LogisticRegression(random_state=42, max_iter=200)
    model_lr_1.fit(X_train, y_train)
    y_pred_lr_1 = model_lr_1.predict(X_test)

    model_lr_2 = LogisticRegression(random_state=42, solver='liblinear')
    model_lr_2.fit(X_train, y_train)
    y_pred_lr_2 = model_lr_2.predict(X_test)

    with open('../models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model_lr, f)

    # 4. Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1.fit(X_train, y_train)
    y_pred_1 = model_1.predict(X_test)

    model_2 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_2.fit(X_train, y_train)
    y_pred_2 = model_2.predict(X_test)

    model_3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_3.fit(X_train, y_train)
    y_pred_3 = model_3.predict(X_test)

    with open('../models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_1, f)

    # 5. SVC
    model_svc = SVC(random_state=42)
    model_svc.fit(X_train, y_train)
    y_pred_svc = model_svc.predict(X_test)

    model_svc_1 = SVC(kernel='linear', random_state=42)
    model_svc_1.fit(X_train, y_train)
    y_pred_svc_1 = model_svc_1.predict(X_test)

    model_svc_2 = SVC(kernel='rbf', random_state=42)
    model_svc_2.fit(X_train, y_train)
    y_pred_svc_2 = model_svc_2.predict(X_test)

    with open('../models/svc_model.pkl', 'wb') as f:
        pickle.dump(model_svc_1, f)


