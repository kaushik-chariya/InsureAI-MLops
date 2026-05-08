import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from src.exception import MyException
from src.constants import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

class Proj1Data:
    """
    A class to export PostgreSQL table as pandas DataFrame.
    """

    def __init__(self) -> None:

        try:

            DATABASE_URL = (
                f"postgresql://{POSTGRES_USER}:"
                f"{POSTGRES_PASSWORD}@"
                f"{POSTGRES_HOST}:"
                f"{POSTGRES_PORT}/"
                f"{POSTGRES_DB}"
            )

            self.engine = create_engine(DATABASE_URL)

        except Exception as e:
            raise MyException(e, sys)

    def export_table_as_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Export PostgreSQL table as DataFrame
        """

        try:

            query = f"SELECT * FROM {table_name}"

            print("Fetching data from PostgreSQL")

            df = pd.read_sql(query, self.engine)

            print(f"Data fetched with len: {len(df)}")

            if "id" in df.columns.to_list():
                df = df.drop(columns=["id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)