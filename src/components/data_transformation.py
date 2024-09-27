import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Remove 'Close' from numerical_columns, as it is the target variable
            numerical_columns = ["Open", "High", "Low", "Adj Close", "Volume"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scalar", StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Step 1: Read train and test CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Step 2: Print the columns to verify their existence and structure
            print("Train DataFrame Columns:", train_df.columns)
            print("Test DataFrame Columns:", test_df.columns)

            # Step 3: Strip any extra spaces from the column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            # Step 4: Verify if all required columns exist
            numerical_columns = ["Open", "High", "Low", "Adj Close", "Volume"]
            missing_cols_train = [col for col in numerical_columns if col not in train_df.columns]
            missing_cols_test = [col for col in numerical_columns if col not in test_df.columns]

            if missing_cols_train:
                raise ValueError(f"Missing columns in the train set: {missing_cols_train}")
            if missing_cols_test:
                raise ValueError(f"Missing columns in the test set: {missing_cols_test}")

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Step 5: Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Step 6: Prepare input features and target variables for both train and test datasets
            target_column_name = "Close"

            # Drop 'Close' from features since it's the target variable
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Step 7: Apply the preprocessor and transform the input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed input features with the target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Step 8: Save the preprocessor object to a file
            logging.info("Saving preprocessing object.")
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


if __name__ == "__main__":
    # Assuming DataIngestion is defined elsewhere and imported properly
    from src.components.data_ingestion import DataIngestion  # Ensure this class exists

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
