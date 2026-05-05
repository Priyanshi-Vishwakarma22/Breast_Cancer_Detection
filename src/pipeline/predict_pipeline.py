import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.threshold = 0.55
        
    def load_artifacts(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            
            return self.model, self.preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_with_proba(self, features):
        try:
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            
            proba = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(data_scaled)
            
            return preds, proba
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
                 Date_of_Surgery: str = None,
                 Date_of_Last_Visit: str = None):
        
        self.Age = Age
        self.Gender = Gender
        self.Protein1 = Protein1
        self.Protein2 = Protein2
        self.Protein3 = Protein3
        self.Protein4 = Protein4
        self.Tumour_Stage = Tumour_Stage
        self.Histology = Histology
        self.HER2_status = HER2_status
        self.Surgery_type = Surgery_type
        self.Date_of_Surgery = Date_of_Surgery
        self.Date_of_Last_Visit = Date_of_Last_Visit
    
    def _calculate_survived_days(self):
        try:
            if self.Date_of_Surgery and self.Date_of_Last_Visit:
                surgery_date = pd.to_datetime(self.Date_of_Surgery, format='%d-%b-%y', errors='coerce')
                last_visit_date = pd.to_datetime(self.Date_of_Last_Visit, format='%d-%b-%y', errors='coerce')
                
                if pd.notna(surgery_date) and pd.notna(last_visit_date):
                    survived_days = (last_visit_date - surgery_date).days
                    if survived_days > 0:
                        return survived_days
        except:
            pass
        return 365
    
    def _encode_tumour_stage(self):
        stage_map = {'I': 1, 'II': 2, 'III': 3}
        return stage_map.get(self.Tumour_Stage, 2)
    
    def get_data_as_dataframe(self):
        try:
            survived_days = self._calculate_survived_days()
            tumour_stage_encoded = self._encode_tumour_stage()
            
            valid_surgeries = ['Lumpectomy', 'Modified Radical Mastectomy', 'Other', 'Simple Mastectomy']
            surgery_type = self.Surgery_type if self.Surgery_type in valid_surgeries else 'Other'
            
            custom_data_input_dict = {
                "Age": [float(self.Age)],
                "Protein1": [float(self.Protein1)],
                "Protein2": [float(self.Protein2)],
                "Protein3": [float(self.Protein3)],
                "Protein4": [float(self.Protein4)],
                "Survived_days": [survived_days],
                "Tumour_Stage_Encoded": [tumour_stage_encoded],
                "Surgery_type": [surgery_type],
                "Histology": [self.Histology],
                "Gender": [self.Gender],
                "HER2 status": [self.HER2_status]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)