import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            smote = SMOTE(random_state=88)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(eval_metric='logloss')),
                ('cat', CatBoostClassifier(verbose=0))
            ]
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
                "Voting Classifier": VotingClassifier(estimators=base_estimators, voting='soft'),
                "Stacking Classifier": StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(),
                    passthrough=True
                )
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': [None, 'balanced']
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'max_depth': [3, 5],
                },
                "Logistic Regression": {
                    'C': [0.01, 0.1, 1.0, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'class_weight': [None, 'balanced']
                },
                "XGBoost": {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                "Voting Classifier": {},
                "Stacking Classifier": {}
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            # Extract best model based on f1_score
            best_model_name = max(model_report, key=lambda k: model_report[k]['f1_score'])
            best_model_score = model_report[best_model_name]['f1_score']

            best_model = trained_models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_probs = best_model.predict_proba(X_test)[:, 1]
            predicted = best_model.predict(X_test)

            f1score = f1_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            roc_auc = roc_auc_score(y_test, y_probs)

            logging.info(f"F1 Score: {f1score}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"ROC AUC: {roc_auc}")

            return f1score
            
        except Exception as e:
            raise CustomException(e,sys)
