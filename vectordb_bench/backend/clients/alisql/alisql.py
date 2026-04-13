import logging
from contextlib import contextmanager

import mysql.connector as mysql
import numpy as np
from mysql.connector.cursor import MySQLCursorPrepared

from ..api import VectorDB
from .config import AliSQLConfigDict, AliSQLIndexConfig

log = logging.getLogger(__name__)


class _NoResetPreparedCursor(MySQLCursorPrepared):
    """MySQLCursorPrepared that skips the unnecessary COM_STMT_RESET.

    mysql-connector-python sends COM_STMT_RESET before every COM_STMT_EXECUTE,
    adding a full network round-trip per query. This is safe to skip when
    results are always fully consumed via fetchall().
    """

    def execute(self, operation, params=None, **kwargs):
        conn = self._connection
        real_reset = conn.cmd_stmt_reset
        conn.cmd_stmt_reset = lambda *a, **kw: None
        try:
            return super().execute(operation, params, **kwargs)
        finally:
            conn.cmd_stmt_reset = real_reset


class AliSQL(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: AliSQLConfigDict,
        db_case_config: AliSQLIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "AliSQL"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        # construct basic units
        self.conn, self.cursor = self._create_connection()

        if drop_old:
            self._drop_db()
            self._create_db_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    def _create_connection(self, use_prepared=False):
        conn = mysql.connect(
            host=self.db_config["host"],
            user=self.db_config["user"],
            port=self.db_config["port"],
            password=self.db_config["password"],
            ssl_disabled=True,
        )
        if use_prepared:
            cursor = conn.cursor(cursor_class=_NoResetPreparedCursor)
        else:
            cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _drop_db(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f'{self.name} client drop db : {self.db_config["database"]}')

        self.cursor.execute(f'DROP DATABASE IF EXISTS {self.db_config["database"]}')
        self.cursor.execute("COMMIT")

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            log.info(f'{self.name} client create database : {self.db_config["database"]}')
            self.cursor.execute(f'CREATE DATABASE {self.db_config["database"]}')

            log.info(f"{self.name} client create table : {self.table_name}")
            self.cursor.execute(f'USE {self.db_config["database"]}')

            self.cursor.execute(
                f"""
              CREATE TABLE {self.table_name} (
                id INT PRIMARY KEY,
                v VECTOR({self.dim}) NOT NULL
              )
            """
            )
            self.cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise e from None

    @contextmanager
    def init(self):
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn, self.cursor = self._create_connection(use_prepared=True)

        index_param = self.case_config.index_param()
        search_param = self.case_config.search_param()

        # Use a plain cursor for SET/COMMIT statements
        plain_cursor = self.conn.cursor()
        plain_cursor.execute("SET sql_mode = ''")

        if index_param["index_type"] == "HNSW":
            if search_param["ef_search"] is not None:
                plain_cursor.execute(f"SET SESSION vidx_hnsw_ef_search = {search_param['ef_search']}")
            plain_cursor.execute("COMMIT")
        plain_cursor.close()

        self.insert_sql = (
            f'INSERT INTO {self.db_config["database"]}.{self.table_name} (id, v) VALUES (%s, %s)'  # noqa: S608
        )
        self.select_sql = (
            f'SELECT id FROM {self.db_config["database"]}.{self.table_name} '  # noqa: S608
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %s"
        )
        self.select_sql_with_filter = (
            f'SELECT id FROM {self.db_config["database"]}.{self.table_name} WHERE id >= %s '  # noqa: S608
            f"ORDER by vec_distance_{search_param['metric_type']}(v, %s) LIMIT %s"
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            index_options = f"DISTANCE={index_param['metric_type']}"
            if index_param["index_type"] == "HNSW" and index_param["M"] is not None:
                index_options += f" M={index_param['M']}"

            self.cursor.execute(
                f"""
              ALTER TABLE {self.db_config["database"]}.{self.table_name}
              ADD VECTOR KEY v(v) {index_options}
            """
            )
            self.cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create index: {self.table_name} error: {e}")
            raise e from None

    @staticmethod
    def vector_to_hex(v):  # noqa: ANN001
        return np.array(v, "float32").tobytes()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            batch_data = []
            for i, row in enumerate(metadata_arr):
                batch_data.append((int(row), self.vector_to_hex(embeddings_arr[i])))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.cursor.execute("COMMIT")

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        search_param = self.case_config.search_param()  # noqa: F841

        try:
            if filters:
                self.cursor.execute(self.select_sql_with_filter, (filters.get("id"), self.vector_to_hex(query), k))
            else:
                self.cursor.execute(self.select_sql, (self.vector_to_hex(query), k))
            return [row[0] for row in self.cursor.fetchall()]

        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
