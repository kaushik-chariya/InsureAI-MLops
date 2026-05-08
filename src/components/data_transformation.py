import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import (
    TARGET_COLUMN,
    SCHEMA_FILE_PATH
)

from src.entity.config_entity import DataTransformationConfig

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)

from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


class DataTransformation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):

        try:

            self.data_ingestion_artifact = data_ingestion_artifact

            self.data_transformation_config = (
                data_transformation_config
            )

            self.data_validation_artifact = (
                data_validation_artifact
            )

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:

        try:

            df = pd.read_csv(file_path)

            # Remove unwanted column if exists
            if "Unnamed: 0" in df.columns:
                df.drop(
                    columns=["Unnamed: 0"],
                    inplace=True
                )

            return df

        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:

        """
        Creates preprocessing pipeline
        """

        logging.info(
            "Entered get_data_transformer_object method"
        )

        try:

            numeric_transformer = StandardScaler()

            min_max_scaler = MinMaxScaler()

            logging.info(
                "Transformers initialized"
            )

            num_features = self._schema_config[
                'num_features'
            ]

            mm_columns = self._schema_config[
                'mm_columns'
            ]

            logging.info(
                "Schema columns loaded"
            )

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "StandardScaler",
                        numeric_transformer,
                        num_features
                    ),

                    (
                        "MinMaxScaler",
                        min_max_scaler,
                        mm_columns
                    )

                ],

                remainder='passthrough'
            )

            final_pipeline = Pipeline(

                steps=[
                    ("Preprocessor", preprocessor)
                ]

            )

            logging.info(
                "Preprocessing pipeline created"
            )

            return final_pipeline

        except Exception as e:

            raise MyException(e, sys) from e

    def _map_gender_column(self, df):

        """
        Convert Gender to binary
        """

        logging.info(
            "Mapping Gender column"
        )

        df['Gender'] = df['Gender'].map(
            {
                'Female': 0,
                'Male': 1
            }
        ).astype(int)

        return df

    def _create_dummy_columns(self, df):

        """
        Create dummy variables
        """

        logging.info(
            "Creating dummy variables"
        )

        df = pd.get_dummies(
            df,
            drop_first=True
        )

        return df

    def _rename_columns(self, df):

        """
        Rename transformed columns
        """

        logging.info(
            "Renaming columns"
        )

        df = df.rename(

            columns={

                "Vehicle_Age_< 1 Year":
                "Vehicle_Age_lt_1_Year",

                "Vehicle_Age_> 2 Years":
                "Vehicle_Age_gt_2_Years"

            }

        )

        dummy_cols = [

            "Vehicle_Age_lt_1_Year",

            "Vehicle_Age_gt_2_Years",

            "Vehicle_Damage_Yes"

        ]

        for col in dummy_cols:

            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def _drop_id_column(self, df):

        """
        Drop columns from schema
        """

        logging.info(
            "Dropping columns"
        )

        drop_cols = self._schema_config[
            'drop_columns'
        ]

        if len(drop_cols) > 0:

            existing_cols = [

                col for col in drop_cols
                if col in df.columns

            ]

            if len(existing_cols) > 0:

                df = df.drop(
                    columns=existing_cols
                )

        return df

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:

        """
        Start transformation pipeline
        """

        try:

            logging.info(
                "Data Transformation Started !!!"
            )

            if not self.data_validation_artifact.validation_status:

                raise Exception(
                    self.data_validation_artifact.message
                )

            # Load train-test data

            train_df = self.read_data(

                file_path=self.data_ingestion_artifact.trained_file_path

            )

            test_df = self.read_data(

                file_path=self.data_ingestion_artifact.test_file_path

            )

            logging.info(
                "Train and test data loaded"
            )

            # Split input and target

            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[
                TARGET_COLUMN
            ]

            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[
                TARGET_COLUMN
            ]

            logging.info(
                "Input and target separated"
            )

            # Train transformations

            input_feature_train_df = (
                self._map_gender_column(
                    input_feature_train_df
                )
            )

            input_feature_train_df = (
                self._drop_id_column(
                    input_feature_train_df
                )
            )

            input_feature_train_df = (
                self._create_dummy_columns(
                    input_feature_train_df
                )
            )

            input_feature_train_df = (
                self._rename_columns(
                    input_feature_train_df
                )
            )

            # Test transformations

            input_feature_test_df = (
                self._map_gender_column(
                    input_feature_test_df
                )
            )

            input_feature_test_df = (
                self._drop_id_column(
                    input_feature_test_df
                )
            )

            input_feature_test_df = (
                self._create_dummy_columns(
                    input_feature_test_df
                )
            )

            input_feature_test_df = (
                self._rename_columns(
                    input_feature_test_df
                )
            )

            logging.info(
                "Custom transformations completed"
            )

            # Preprocessor

            preprocessor = (
                self.get_data_transformer_object()
            )

            logging.info(
                "Applying preprocessing"
            )

            input_feature_train_arr = (
                preprocessor.fit_transform(
                    input_feature_train_df
                )
            )

            input_feature_test_arr = (
                preprocessor.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Preprocessing completed"
            )

            # Handle imbalance

            smt = SMOTEENN(
                sampling_strategy="minority"
            )

            input_feature_train_final, \
            target_feature_train_final = (

                smt.fit_resample(

                    input_feature_train_arr,
                    target_feature_train_df

                )

            )

            input_feature_test_final, \
            target_feature_test_final = (

                smt.fit_resample(

                    input_feature_test_arr,
                    target_feature_test_df

                )

            )

            logging.info(
                "SMOTEENN applied"
            )

            # Concatenate target

            train_arr = np.c_[

                input_feature_train_final,
                np.array(target_feature_train_final)

            ]

            test_arr = np.c_[

                input_feature_test_final,
                np.array(target_feature_test_final)

            ]

            logging.info(
                "Train-test arrays created"
            )

            # Save objects

            save_object(

                self.data_transformation_config.transformed_object_file_path,

                preprocessor

            )

            save_numpy_array_data(

                self.data_transformation_config.transformed_train_file_path,

                array=train_arr

            )

            save_numpy_array_data(

                self.data_transformation_config.transformed_test_file_path,

                array=test_arr

            )

            logging.info(
                "Transformation artifacts saved"
            )

            return DataTransformationArtifact(

                transformed_object_file_path=

                self.data_transformation_config.transformed_object_file_path,

                transformed_train_file_path=

                self.data_transformation_config.transformed_train_file_path,

                transformed_test_file_path=

                self.data_transformation_config.transformed_test_file_path

            )

        except Exception as e:

            raise MyException(e, sys) from e