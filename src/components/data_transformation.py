import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols, fmt='%d-%b-%y'):
        self.date_cols = date_cols
        self.fmt = fmt

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols = [c for c in self.date_cols if c in X_copy.columns]
        for col in cols:
            X_copy[col] = pd.to_datetime(X_copy[col], format=self.fmt, errors='coerce')

        if len(cols) == 2:
            X_copy['Survival_Days'] = (X_copy[cols[1]] - X_copy[cols[0]]).dt.days
            X_copy['Survival_Days'] = X_copy['Survival_Days'].fillna(0)

        return X_copy.drop(columns=cols, axis=1)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4', 'Survival_Days'
            ]
            categorical_columns = [
                'Gender', 'Tumour_Stage', 'Histology', 'HER2 status', 'Surgery_type'
            ]
            date_columns = ['Date_of_Surgery', 'Date_of_Last_Visit']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            #  first extract dates, then transform
            preprocessor = Pipeline(steps=[
                ('date_extractor', DateFeatureExtractor(date_cols=date_columns)),
                ('col_transformer', ColumnTransformer(transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]))
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed.')

            target_column_name = 'Patient_Status'

            
            train_df = train_df.dropna(subset=[target_column_name])
            test_df = test_df.dropna(subset=[target_column_name])

            
            train_df[target_column_name] = train_df[target_column_name].map({'Alive': 1, 'Dead': 0})
            test_df[target_column_name] = test_df[target_column_name].map({'Alive': 1, 'Dead': 0})

            
            cols_to_drop = ['ER status', 'PR status']

            input_feature_train_df = train_df.drop(columns=[target_column_name] + cols_to_drop)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name] + cols_to_drop)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing on train and test data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed.")

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