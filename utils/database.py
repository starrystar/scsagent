import pymysql
import json
from scsagent.config.env import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, ENCODING

class Database:
    """
    Example
        from util import Database
        db = Database()
        tool = 'SingleR'
        res = db.execute_query(f"select code from tool where name='{tool}'")
        res[0][0] # 结果的第0行第0列

        db.execute_update("UPDATE your_table SET column = %s WHERE condition", ("new_value",))
        db.disconnect()

    """

    def __init__(self):
        self._connection = None
        self.connect()

    def connect(self):
        self._connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def execute_query(self, query, params=None):
        with self._connection.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def execute_update(self, query, params=None):
        with self._connection.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self._connection.commit()
            return cursor.rowcount
