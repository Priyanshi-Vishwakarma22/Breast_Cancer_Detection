import os
import sys

import dill
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """Save a Python object to disk with dill, creating directories as needed."""
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object from disk."""
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """
    Train and evaluate multiple classification models.
    Returns a report dict with metrics for each model.
    """
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
                    scoring='f1_weighted'  # ✅ imbalanced dataset ke liye
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            #  Classification metrics
            model_report[model_name] = {
                'accuracy':  round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(y_test, y_pred, zero_division=0, average='weighted'), 4),
                'recall':    round(recall_score(y_test, y_pred, zero_division=0, average='weighted'), 4),
                'f1_score':  round(f1_score(y_test, y_pred, zero_division=0, average='weighted'), 4),
            }

        return model_report

    except Exception as e:
        raise CustomException(e, sys)

    


