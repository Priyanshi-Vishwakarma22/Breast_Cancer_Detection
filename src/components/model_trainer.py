import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_models(self):
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        }

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = self.get_models()
            model_report = {}
            best_model = None
            best_model_name = None
            best_score = -1.0

            for name, model in models.items():
                logging.info(f'Training model: {name}')
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                current_f1 = f1_score(y_test, y_pred, zero_division=0)

                model_report[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': current_f1,
                }

                logging.info(f'{name} test F1: {current_f1:.4f}')

                if current_f1 > best_score:
                    best_score = current_f1
                    best_model = model
                    best_model_name = name

            if best_model is None:
                raise CustomException('No model was trained successfully.', sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info(f'Best model: {best_model_name} with F1 score {best_score:.4f}')
            return best_model_name, best_model, model_report

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    logging.info('Model trainer module loaded.')
