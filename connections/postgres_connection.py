import psycopg2
import os


DB_NAME = os.getenv("DB_NAME", "sentiment_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "d7fiwxxmlx")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")


class PostgresConnection:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = None
            cls._instance.cursor = None
            print("creating new postgres instance")
        else:
            print("existing postgre instance already exists")
        return cls._instance

    def __init__(self):
        pass

    def connect(self):
        if self._instance.connection is None or self._instance.connection.closed:
            try:
                self._instance.connection = psycopg2.connect(
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT,
                )
                self._instance.cursor = self._instance.connection.cursor()
                print("connection to PostgreSQL DB successful")
            except psycopg2.Error as e:
                print(f"connection to PostgreSQL DB failed: {e}")
                self._instance.connection = None
                self._instance.cursor = None
        else:
            print("connection already active")
        return self._instance.connection

    def get_cursor(self):
        if self._instance.connection is None or self._instance.connection.closed:
            self.connect()
        return self._instance.cursor

    def is_connected(self):
        return (
            self._instance.connection is not None
            and not self._instance.connection.closed
        )

    def close(self):
        if self._instance.cursor and not self._instance.cursor.closed:
            self._instance.cursor.close()
            print("cursor closed")
        if self._instance.connection and self._instance.connection.closed:
            self._instance.connection.close()
            print("connection to postgres DB closed")
        self._instance.connection = None
        self._instance.cursor = None
