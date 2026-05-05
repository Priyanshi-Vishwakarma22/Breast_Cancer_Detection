import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

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
                'Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4',
                'Survived_days', 'Tumour_Stage_Encoded'
            ]

            categorical_columns = [
                'Surgery_type', 'Histology', 'Gender', 'HER2 status'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler',  StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])

            logging.info(f"Numerical columns  : {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            # ── 1. Cleaning ───────────────────────────────────────────────
            cols_to_drop = ['ER status', 'PR status']
            train_df = train_df.drop(columns=cols_to_drop)
            test_df  = test_df.drop(columns=cols_to_drop)
            logging.info(f"Dropped constant columns: {cols_to_drop}")

            # ── 2. Feature Engineering ────────────────────────────────────
            train_df['Date_of_Surgery']    = pd.to_datetime(train_df['Date_of_Surgery'],    format='%d-%b-%y', errors='coerce')
            train_df['Date_of_Last_Visit'] = pd.to_datetime(train_df['Date_of_Last_Visit'], format='%d-%b-%y', errors='coerce')
            test_df['Date_of_Surgery']     = pd.to_datetime(test_df['Date_of_Surgery'],     format='%d-%b-%y', errors='coerce')
            test_df['Date_of_Last_Visit']  = pd.to_datetime(test_df['Date_of_Last_Visit'],  format='%d-%b-%y', errors='coerce')

            train_df['Survived_days'] = (train_df['Date_of_Last_Visit'] - train_df['Date_of_Surgery']).dt.days
            test_df['Survived_days']  = (test_df['Date_of_Last_Visit']  - test_df['Date_of_Surgery']).dt.days

            # Sirf train ka median use karo — data leakage rokne ke liye
            survived_median = train_df['Survived_days'].median()
            train_df['Survived_days'] = train_df['Survived_days'].fillna(survived_median)
            test_df['Survived_days']  = test_df['Survived_days'].fillna(survived_median)

            train_df = train_df.drop(columns=['Date_of_Surgery', 'Date_of_Last_Visit'])
            test_df  = test_df.drop(columns=['Date_of_Surgery', 'Date_of_Last_Visit'])
            logging.info("Survived_days feature created. Date columns dropped.")

            # ── 3. Ordinal encode Tumour_Stage ────────────────────────────
            stage_map = {'I': 1, 'II': 2, 'III': 3}
            train_df['Tumour_Stage_Encoded'] = train_df['Tumour_Stage'].map(stage_map)
            test_df['Tumour_Stage_Encoded']  = test_df['Tumour_Stage'].map(stage_map)
            train_df = train_df.drop(columns=['Tumour_Stage'])
            test_df  = test_df.drop(columns=['Tumour_Stage'])
            logging.info("Tumour_Stage ordinal-encoded (I=1, II=2, III=3).")

            # ── 4. Encode target ──────────────────────────────────────────
            target_column = 'Patient_Status'
            label_map = {'Alive': 1, 'Dead': 0}

            y_train = train_df[target_column].map(label_map)
            y_test  = test_df[target_column].map(label_map)

            X_train = train_df.drop(columns=[target_column])
            X_test  = test_df.drop(columns=[target_column])
            logging.info("Target encoded: Alive=1, Dead=0.")

            # ── 5. Fit & Transform ────────────────────────────────────────
            preprocessor_obj = self.get_data_transformer_object()
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            X_test_arr  = preprocessor_obj.transform(X_test)
            logging.info("Preprocessor fitted on train and applied on test.")

            # ── 6. SMOTE — sirf train pe ──────────────────────────────────
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_arr, y_train)
            logging.info(
                f"SMOTE applied. Balanced train shape: {X_train_balanced.shape} | "
                f"Class counts: {np.bincount(y_train_balanced)}"
            )

            # ── 7. Stack features + target ────────────────────────────────
            train_arr = np.c_[X_train_balanced, np.array(y_train_balanced)]
            test_arr  = np.c_[X_test_arr,       np.array(y_test)]

            # ── 8. Save preprocessor ──────────────────────────────────────
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )
            logging.info(f"Preprocessor saved → {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)