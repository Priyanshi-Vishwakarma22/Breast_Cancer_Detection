# import os
# import sys
# from dataclasses import dataclass

# import numpy as np
# from catboost import CatBoostClassifier
# from sklearn.ensemble import (
#     AdaBoostClassifier,
#     GradientBoostingClassifier,
#     RandomForestClassifier,
# )
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import evaluate_models, save_object


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info('Splitting training and test input data')

#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1],
#             )

#             logging.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

#             models = {
#                 'Logistic Regression': LogisticRegression(
#                     max_iter=1000, random_state=42, class_weight='balanced'
#                 ),
#                 'Decision Tree': DecisionTreeClassifier(
#                     random_state=42, class_weight='balanced'
#                 ),
#                 'Random Forest': RandomForestClassifier(
#                     n_estimators=300, max_depth=6, min_samples_split=5,
#                     random_state=42, class_weight='balanced'
#                 ),
#                 'Gradient Boosting': GradientBoostingClassifier(
#                     n_estimators=200, max_depth=4, random_state=42
#                 ),
#                 'AdaBoost': AdaBoostClassifier(
#                     n_estimators=200, random_state=42
#                 ),
#                 'KNN': KNeighborsClassifier(n_neighbors=5),
#                 'CatBoost': CatBoostClassifier(
#                     random_state=42, verbose=0, auto_class_weights='Balanced'
#                 ),
#                 'XGBoost': XGBClassifier(
#                     random_state=42, eval_metric='logloss'
#                 ),
#                 'SVM': SVC(
#                     class_weight='balanced', kernel='rbf',
#                     probability=True, random_state=42
#                 ),
#                 'Naive Bayes': GaussianNB(),
#             }

#             params = {
#                 'Logistic Regression': {
#                     'C': [0.1, 1.0, 10.0],
#                     'solver': ['lbfgs', 'liblinear']
#                 },
#                 'Decision Tree': {
#                     'max_depth': [3, 5, 7, None],
#                     'min_samples_split': [2, 5, 10]
#                 },
#                 'Random Forest': {
#                     'n_estimators': [100, 200, 300],
#                     'max_depth': [4, 6, None],
#                     'min_samples_split': [2, 5, 10]
#                 },
#                 'Gradient Boosting': {
#                     'n_estimators': [100, 200],
#                     'learning_rate': [0.05, 0.1, 0.2]
#                 },
#                 'AdaBoost': {
#                     'n_estimators': [100, 200],
#                     'learning_rate': [0.5, 1.0]
#                 },
#                 'KNN': {
#                     'n_neighbors': [3, 5, 7, 9]
#                 },
#                 'CatBoost': {
#                     'iterations': [100, 200],
#                     'learning_rate': [0.05, 0.1]
#                 },
#                 'XGBoost': {
#                     'n_estimators': [50, 100],
#                     'learning_rate': [0.05, 0.1],
#                     'max_depth': [3, 5]
#                 },
#                 'SVM': {
#                     'C': [0.1, 1.0, 10.0],
#                     'kernel': ['rbf', 'linear']
#                 },
#                 'Naive Bayes': {},
#             }

    
#             model_report, trained_models = evaluate_models(
#                 X_train=X_train,
#                 y_train=y_train,
#                 X_test=X_test,
#                 y_test=y_test,
#                 models=models,
#                 params=params
#             )

       
#             best_model_name = max(model_report, key=lambda x: model_report[x]['f1_score'])
#             best_model_score = model_report[best_model_name]['f1_score']

#             best_model = trained_models[best_model_name]

#             if best_model_score < 0.6:
#                 raise CustomException("No best model found with acceptable F1 score", sys)

#             logging.info(f"Best model: {best_model_name} | F1 Score: {best_model_score:.4f}")

          
#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model,
#             )

      
#             for name, metrics in model_report.items():
#                 logging.info(
#                     f"{name} | Acc: {metrics['test_accuracy']:.4f} | "
#                     f"F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}"
#                 )

#             return best_model_name, best_model, model_report

#         except Exception as e:
#             raise CustomException(e, sys)