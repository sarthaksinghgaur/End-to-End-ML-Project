import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            para = param[model_name]

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                scoring='f1',
                cv=skf,
                verbose=1,
                n_jobs=-1
            )
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)

            report[model_name] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            }

            trained_models[model_name] = model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)