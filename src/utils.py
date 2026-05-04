# import os
# import sys
# import dill

# from sklearn.metrics import (
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
# from sklearn.model_selection import GridSearchCV

# from src.exception import CustomException


# def save_object(file_path, obj):
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, 'wb') as f:
#             dill.dump(obj, f)
#     except Exception as e:
#         raise CustomException(e, sys)


# def load_object(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             return dill.load(f)
#     except Exception as e:
#         raise CustomException(e, sys)


# def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
#     try:
#         params = params or {}
#         report = {}
#         trained_models = {}  

#         for name, model in models.items():

#             if name in params and params[name]:
#                 gs = GridSearchCV(
#                     model,
#                     params[name],
#                     cv=3,
#                     n_jobs=-1,
#                     scoring='f1_weighted'
#                 )
#                 gs.fit(X_train, y_train)
#                 model = gs.best_estimator_ 
#             else:
#                 model.fit(X_train, y_train)

#             trained_models[name] = model 

#             y_train_pred = model.predict(X_train)
#             y_test_pred  = model.predict(X_test)

#             if hasattr(model, "predict_proba"):
#                 y_prob  = model.predict_proba(X_test)[:, 1]
#                 roc_auc = roc_auc_score(y_test, y_prob)
#             else:
#                 roc_auc = roc_auc_score(y_test, y_test_pred)

#             report[name] = {
#                 "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
#                 "test_accuracy":  round(accuracy_score(y_test, y_test_pred), 4),
#                 "precision":      round(precision_score(y_test, y_test_pred, average="weighted", zero_division=0), 4),
#                 "recall":         round(recall_score(y_test, y_test_pred, average="weighted", zero_division=0), 4),
#                 "f1_score":       round(f1_score(y_test, y_test_pred, average="weighted", zero_division=0), 4),
#                 "roc_auc":        round(roc_auc, 4),
#                 "report":         classification_report(y_test, y_test_pred, zero_division=0),
#                 "matrix":         confusion_matrix(y_test, y_test_pred),
#             }

#         return report, trained_models  

#     except Exception as e:
#         raise CustomException(e, sys)
    


