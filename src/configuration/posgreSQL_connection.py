import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.exception import MyException
from src.logger import logging

from src.constants import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD
)


class PostgreSQLClient:
    """
    PostgreSQLClient is responsible for establishing
    connection with PostgreSQL database.
    """

    engine = None
    SessionLocal = None

    def __init__(self) -> None:

        try:

            # Create connection only once
            if PostgreSQLClient.engine is None:

                DATABASE_URL = (
                    f"postgresql://{POSTGRES_USER}:"
                    f"{POSTGRES_PASSWORD}@"
                    f"{POSTGRES_HOST}:"
                    f"{POSTGRES_PORT}/"
                    f"{POSTGRES_DB}"
                )

                # Create SQLAlchemy engine
                PostgreSQLClient.engine = create_engine(DATABASE_URL)

                # Session maker
                PostgreSQLClient.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=PostgreSQLClient.engine
                )

                logging.info("PostgreSQL connection successful")

            self.engine = PostgreSQLClient.engine
            self.session = PostgreSQLClient.SessionLocal()

        except Exception as e:
            raise MyException(e, sys)