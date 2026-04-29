import sys
import os
import pickle
import pandas as pd
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = pickle.load(open(model_path, "rb"))
            preprocessor = pickle.load(open(preprocessor_path, "rb"))

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


# Helper class for input
class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        return pd.DataFrame([self.data])       