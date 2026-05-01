import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score


# ----------------------------
# SAVE OBJECT
# ----------------------------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Exception(f"Error saving object: {e}")


# ----------------------------
# LOAD OBJECT
# ----------------------------
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Exception(f"Error loading object: {e}")


# ----------------------------
# EVALUATE MODELS
# ----------------------------
def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}
    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        report[name] = {
            "accuracy": acc,
            "roc_auc": roc
        }

        if acc > best_score:
            best_score = acc
            best_model = model

    return report, best_model