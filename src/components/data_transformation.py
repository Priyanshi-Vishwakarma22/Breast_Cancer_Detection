import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Age", "Protein1", "Protein2", "Protein3", "Protein4"]
            categorical_columns = [
                "Gender",
                "Tumour_Stage",
                "Histology",
                "ER_status",
                "PR_status",
                "Surgery_type"
                
            ]

            num_pipeline = pipeline(
                steps=[
                    ()
                ]
            )

            pass
        except:
            pass


