import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path        = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model        = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds       = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: int,
                 Gender: str,
                 Protein1: float,
                 Protein2: float,
                 Protein3: float,
                 Protein4: float,
                 Tumour_Stage: str,
                 Histology: str,
                 HER2_status: str,
                 Surgery_type: str,
                 Date_of_Surgery: str,
                 Date_of_Last_Visit: str):

        self.Age                = Age
        self.Gender             = Gender
        self.Protein1           = Protein1
        self.Protein2           = Protein2
        self.Protein3           = Protein3
        self.Protein4           = Protein4
        self.Tumour_Stage       = Tumour_Stage
        self.Histology          = Histology
        self.HER2_status        = HER2_status
        self.Surgery_type       = Surgery_type
        self.Date_of_Surgery    = Date_of_Surgery
        self.Date_of_Last_Visit = Date_of_Last_Visit

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age":                [self.Age],
                "Gender":             [self.Gender],
                "Protein1":           [self.Protein1],
                "Protein2":           [self.Protein2],
                "Protein3":           [self.Protein3],
                "Protein4":           [self.Protein4],
                "Tumour_Stage":       [self.Tumour_Stage],
                "Histology":          [self.Histology],
                "HER2 status":        [self.HER2_status],
                "Surgery_type":       [self.Surgery_type],
                "Date_of_Surgery":    [self.Date_of_Surgery],
                "Date_of_Last_Visit": [self.Date_of_Last_Visit],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

        