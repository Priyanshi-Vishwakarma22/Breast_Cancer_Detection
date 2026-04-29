import os
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import CustomException


class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training started")

            model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model accuracy: {accuracy}")

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)

            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)