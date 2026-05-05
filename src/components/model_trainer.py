import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        try:
            results = {}

            for name, model in models.items():
                print(f"\n{'='*50}\nTraining: {name}")
                logging.info(f"Training model: {name}")

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred  = model.predict(X_test)

                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc  = accuracy_score(y_test,  y_test_pred)
                precision = precision_score(y_test, y_test_pred, zero_division=0)
                recall    = recall_score(y_test,    y_test_pred, zero_division=0)
                f1        = f1_score(y_test,        y_test_pred, zero_division=0)

                # ROC-AUC — predict_proba se calculate karo
                roc_auc = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob  = model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_prob)
                    except:
                        pass
                elif hasattr(model, "decision_function"):
                    try:
                        y_scores = model.decision_function(X_test)
                        roc_auc  = roc_auc_score(y_test, y_scores)
                    except:
                        pass

                print(f"  Train Accuracy : {train_acc:.4f}")
                print(f"  Test Accuracy  : {test_acc:.4f}")
                print(f"  Precision      : {precision:.4f}")
                print(f"  Recall         : {recall:.4f}")
                print(f"  F1 Score       : {f1:.4f}")
                print(f"ROC-AUC : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC: N/A")

                # Overfitting check
                if train_acc - test_acc > 0.1:
                    print("  Warning: Possible Overfitting!")

                results[name] = {
                    "model":     model,
                    "train_acc": train_acc,
                    "test_acc":  test_acc,
                    "precision": precision,
                    "recall":    recall,
                    "f1":        f1,
                    "roc_auc":   roc_auc,
                }

                roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"

                logging.info(
                    f"{name} → test_acc={test_acc:.4f} | "
                    f"f1={f1:.4f} | roc_auc={roc_auc_str}"
                    )

            return results

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training.")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)
            X_test  = test_array[:, :-1]
            y_test  = test_array[:, -1].astype(int)

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight='balanced', random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200, random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "KNN":           KNeighborsClassifier(),
                "SVM":           SVC(probability=True, random_state=42),
                "XGBoost":       XGBClassifier(
                    eval_metric='logloss', random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost":      AdaBoostClassifier(random_state=42),
                "Naive Bayes":   GaussianNB(),
            }

            # Train & evaluate all models
            results = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            #  Best model
            best_model_name = max(results, key=lambda x: results[x]['f1'])
            best_model      = results[best_model_name]['model']
            best_f1         = results[best_model_name]['f1']
            best_acc        = results[best_model_name]['test_acc']

            if best_f1 < 0.60:
                raise CustomException("No model achieved acceptable F1 score (>60%).", sys)

            # Final report 
            y_pred = best_model.predict(X_test)

            print("\n" + "=" * 60)
            print(f"  BEST MODEL    : {best_model_name}")
            print(f"  TEST ACCURACY : {best_acc:.4f}")
            print(f"  F1 SCORE      : {best_f1:.4f}")
            print("=" * 60)
            print("\nClassification Report:\n")
            print(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

            # Save best model 
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"Best model saved → {self.config.trained_model_file_path}")

            return best_f1, best_model_name

        except Exception as e:
            raise CustomException(e, sys)