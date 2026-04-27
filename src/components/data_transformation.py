import sys
import os
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self, X):

        numerical_columns = X.select_dtypes(exclude="object").columns
        categorical_columns = X.select_dtypes(include="object").columns

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            target_column = "Patient_Status"

            # ✅ Remove NaN in target
            train_df = train_df.dropna(subset=[target_column])
            test_df = test_df.dropna(subset=[target_column])

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # ✅ Encode target
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            preprocessor = self.get_data_transformer_object(X_train)

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)