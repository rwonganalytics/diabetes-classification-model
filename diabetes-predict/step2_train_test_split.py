"""
módulo para apartar una parte de la data original para hacer predicciones
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def dataset_split():
    """
    función para separar dataset en train y test
    """
    df = pd.read_csv('../data/raw/diabetes.csv')

    # División en train y test
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)

    # Guardamos cada dataset (train y test)
    df_train.to_csv('../data/processed/df_train.csv', index=False)

    X_test.to_csv('../data/processed/df_test.csv', index=False)
