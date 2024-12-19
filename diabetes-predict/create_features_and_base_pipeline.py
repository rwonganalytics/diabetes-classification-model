"""
Este script define el pipeline base para el modelo de ML
"""

import pickle
import os
import configparser

from feature_engine.imputation import MeanMedianImputer
from feature_engine.selection import DropFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def create_features_and_base_pipeline():
    """
    Esta función define el pipeline base para el modelo de ML.
    """    
    
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  
    # leemos del configparser.
    config = configparser.ConfigParser()
    config.read(os.path.join(project_path, "pipeline.cfg"))

    # Creación del pipeline
    diabetes_predict_pipeline = Pipeline([
        # eliminamos variables que no usaremos.
        ('delete_features',
         DropFeatures(features_to_drop=config.get('GENERAL', 'vars_to_drop').split(','))),

        # Imputación de variables continuas
        ('continuous_var_mean_imputation',
         MeanMedianImputer(imputation_method='mean',
                           variables=config.get('CONTINUES', 'vars_to_impute').split(','))),

        # Estandarización de variables
        ("feature_scaling", MinMaxScaler())
    ])

    # Guardamos el pipeline
    with open(os.path.join(project_path, 'artifacts', 'pipeline.pkl'), 'wb') as f:
        pickle.dump(diabetes_predict_pipeline, f)

create_features_and_base_pipeline()
