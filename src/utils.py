import os
import sys

import dill
import numpy as np
from sklearn.metrics import (
    accuracy_score,                                                                      # % of correct prediction
    classification_report,                                                               #shows precision,recall,f1-score together
    confusion_matrix,                                                                    #shows prediction vs actual values in matrix form
    f1_score,                                                                            #balance of precision and recall
    precision_score,                                                                     #predicted + are correct
    recall_score,                                                                        #actual + are correctly identified
    roc_auc_score,                                                                       #measures overall model performance
    roc_curve,  
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    try:
        if params is None:
            params = {}

        model_report = {}

        for model_name, model in models.items():
            if model_name in params and params[model_name]:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    cv=3,
                    n_jobs=-1,
                    verbose=0,
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            try:
                model_report[model_name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0, average='weighted'),
                    'recall': recall_score(y_test, y_pred, zero_division=0, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0, average='weighted'),
                }
            except Exception:
                model_report[model_name] = {
                    'r2_score': r2_score(y_test, y_pred),
                }

        return model_report
    except Exception as e:
        raise CustomException(e, sys)

    


