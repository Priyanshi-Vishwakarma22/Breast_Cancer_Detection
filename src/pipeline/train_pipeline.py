import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion        = DataIngestion()
        self.data_transformation   = DataTransformation()
        self.model_trainer         = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 60)

            #Step 1: Data Ingestion 
            logging.info("Step 1: Data Ingestion")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Train path: {train_path} | Test path: {test_path}")

            # Step 2: Data Transformation 
            logging.info("Step 2: Data Transformation")
            train_arr, test_arr, preprocessor_path = \
                self.data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info(f"Preprocessor saved at: {preprocessor_path}")

            # Step 3: Model Training 
            logging.info("Step 3: Model Training")
            best_f1, best_model_name = \
                self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE COMPLETED")
            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"F1 Score   : {best_f1:.4f}")
            logging.info("=" * 60)

            print(f"\n{'='*60}")
            print(f"  PIPELINE COMPLETE")
            print(f"  Best Model : {best_model_name}")
            print(f"  F1 Score   : {best_f1:.4f}")
            print(f"{'='*60}")

            return best_f1, best_model_name

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()