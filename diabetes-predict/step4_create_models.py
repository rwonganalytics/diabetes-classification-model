"""
modulo para creación de modelos de clasificación (5 distintos con 3 config)
"""
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def model_creation():
    """
    funcion que divide en train/test para entrenar, estandariza data y crea modelos
    """
    dataset = pd.read_csv("../data/processed/features_for_model.csv")

    # Dividir el dataset en X (características) y y (target)
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']

    # Dividir los datos en conjunto de entrenamiento y prueba (con los datos ya procesados)
    X_train, X_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

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

    model_gb_1 = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model_gb_1.fit(X_train, y_train)

    model_gb_2 = GradientBoostingClassifier(n_estimators=25, learning_rate=0.05, random_state=42)
    model_gb_2.fit(X_train, y_train)

    with open('../models/gb_model.pkl', 'wb') as f:
        pickle.dump(model_gb_2, f)

    # 2. KN Neighbors
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train, y_train)

    model_knn_1 = KNeighborsClassifier(n_neighbors=5)
    model_knn_1.fit(X_train, y_train)

    model_knn_2 = KNeighborsClassifier(n_neighbors=2)
    model_knn_2.fit(X_train, y_train)

    with open('../models/knn_model.pkl', 'wb') as f:
        pickle.dump(model_knn, f)

    # 3. Logistic Regression
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train, y_train)

    model_lr_1 = LogisticRegression(random_state=42, max_iter=200)
    model_lr_1.fit(X_train, y_train)

    model_lr_2 = LogisticRegression(random_state=42, solver='liblinear')
    model_lr_2.fit(X_train, y_train)

    with open('../models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model_lr, f)

    # 4. Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1.fit(X_train, y_train)

    model_2 = RandomForestClassifier(n_estimators=200, random_state=42)
    model_2.fit(X_train, y_train)

    model_3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_3.fit(X_train, y_train)

    with open('../models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_1, f)

    # 5. SVC
    model_svc = SVC(random_state=42)
    model_svc.fit(X_train, y_train)

    model_svc_1 = SVC(kernel='linear', random_state=42)
    model_svc_1.fit(X_train, y_train)

    model_svc_2 = SVC(kernel='rbf', random_state=42)
    model_svc_2.fit(X_train, y_train)

    with open('../models/svc_model.pkl', 'wb') as f:
        pickle.dump(model_svc_1, f)
