import psycopg2
import pandas as pd


class PostgreDataFetch:
    def __init__(self):
        pass

    def fetch_data_from_db(self, cursor):
        """
        Fetches the entire movies dataset from postgres
        """
        try:
            query = f"""
                SELECT review, sentiment
                FROM movie_reviews;
            """
            cursor.execute(query)
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=["review", "sentiment"])
            return df
        except Exception as e:
            print(f"Error fetching data from database: {e}")
            return pd.DataFrame()
        finally:
            if cursor:
                cursor.close()
