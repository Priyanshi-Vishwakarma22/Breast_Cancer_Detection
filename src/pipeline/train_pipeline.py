import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # 1. Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # 2. Data Transformation
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
                train_path, test_path
            )

            # 3. Model Training
            trainer = ModelTrainer()
            accuracy = trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )

            logging.info("Training pipeline completed")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)