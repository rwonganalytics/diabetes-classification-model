"""
Este modulo carga la data cruda de diabetes
"""
import pandas as pd

def load_data():
    """
    función que lee un .csv y devulve dataframe con dataset de diabetes
    """
    df = pd.read_csv('../data/raw/diabetes.csv')
    return df
