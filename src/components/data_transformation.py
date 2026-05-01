import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

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
    
        # this function is responsible for data transformation
        
    
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
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))

            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        

        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed.')

            train_df.replace('', np.nan, inplace=True)
            test_df.replace('', np.nan, inplace=True)

            target_column = 'Patient_Status'
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

            feature_columns = numerical_columns + categorical_columns

            train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
            test_df = test_df.dropna(subset=[target_column]).reset_index(drop=True)

            target_mapping = {
                'Alive': 1,
                'Dead': 0,
            }

            train_df[target_column] = train_df[target_column].map(target_mapping)
            test_df[target_column] = test_df[target_column].map(target_mapping)

            train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
            test_df = test_df.dropna(subset=[target_column]).reset_index(drop=True)

            logging.info('Obtaining processing object.')
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column].astype(int)

            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column].astype(int)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df, dtype=np.float64)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df, dtype=np.float64)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

