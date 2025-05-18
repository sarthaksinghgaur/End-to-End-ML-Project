import os
import sys
import dataclasses

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifact', "preprocessor.pkl")

class FeatureEngineering(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['bmi_category'] = X['bmi'].apply(self.bmi_category)
        X['age_category'] = X['age'].apply(self.age_category)
        return X

    def bmi_category(self, bmi):
        if bmi < 18.5: return 'underweight'
        elif bmi < 25: return 'normal'
        elif bmi < 30: return 'overweight'
        else: return 'obese'
    
    def age_category(self, bmi):
        if bmi < 18: return 'child'
        elif bmi < 40: return 'yound_adult'
        elif bmi < 60: return 'middle_aged'
        else: return 'senior'

class DataTransformation:
    def __init__(self):
        self.Data_Transformation_Config = DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        try:
            cat_cols = ['gender', 
                        'smoking_history',
                        'bmi_category',
                        'age_category']
            
            num_cols = ['age',
                        'bmi',
                        'HbA1c_level',
                        'blood_glucose_level']
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(missing_values='No Info', strategy='most_frequent')),
                    ("onehot", OneHotEncoder(sparse_output=False)),
                    ("scalar", StandardScaler())
                ]
            )
            num_pipeline = Pipeline([
                ("scalar", StandardScaler())
            ])

            transformer = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            preprocessor = Pipeline(steps=[
                                ("feature_engineering", FeatureEngineering()),
                                ("preprocessing", transformer)
                            ])

            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining prepreocessing object")

            preprocessor = self.get_data_tranformer_object()

            target_column_name = 'diabetes'

            input_feautre_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feautre_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying feature engineering object on training dataframe and testing dataframe.")

            logging.info("Applying preprocession object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor.fit_transform(input_feautre_train_df)
            input_feature_test_arr = preprocessor.transform(input_feautre_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.Data_Transformation_Config.preprocessor_file_path,
                obj = preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.Data_Transformation_Config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

