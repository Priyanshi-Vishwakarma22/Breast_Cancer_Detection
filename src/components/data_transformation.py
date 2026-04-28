import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'Age',
                'Protein1',
                'Protein2',
                'Protein3',
                'Protein4'
            ]
            categorical_columns = [
                'Gender',
                'Tumour_Stage',
                'Histology',
                'ER status',
                'PR status',
                'HER2 status',
                'Surgery_type'
            ]

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'Patient_Status'
            drop_columns = ['Date_of_Surgery', 'Date_of_Last_Visit']

            train_df = train_df.dropna(subset=[target_column])
            test_df = test_df.dropna(subset=[target_column])

            input_feature_train_df = train_df.drop(columns=drop_columns + [target_column], errors='ignore')
            input_feature_test_df = test_df.drop(columns=drop_columns + [target_column], errors='ignore')

            target_train_series = train_df[target_column].map({'Alive': 0, 'Dead': 1})
            target_test_series = test_df[target_column].map({'Alive': 0, 'Dead': 1})

            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(input_feature_train_df)

            input_train_arr = preprocessor.transform(input_feature_train_df)
            input_test_arr = preprocessor.transform(input_feature_test_df)

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)

            return (
                input_train_arr,
                input_test_arr,
                target_train_series.values,
                target_test_series.values,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


