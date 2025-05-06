from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import (
    Any,
    Generic,
)

from .abstract import DataProvider, T


class DatabaseAdapter(ABC, Generic[T]):
    """Abstract base class for database adapter implementations.

    This class defines the interface for adapters that connect to different database
    systems. Concrete implementations of this class handle the specific details of
    connecting to databases, executing queries, and transforming results into a
    consistent format for the data processing pipeline.

    Example:
        ```python
        import sqlite3


        class SQLiteAdapter(DatabaseAdapter[dict[str, Any]]):
            def connect(self, connection_params: dict[str, Any]) -> sqlite3.Connection:
                connection = sqlite3.connect(connection_params["database"])
                connection.row_factory = sqlite3.Row
                return connection

            def execute_query(
                self, connection: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()
            ) -> sqlite3.Cursor:
                cursor = connection.cursor()
                cursor.execute(query, params)
                return cursor

            def fetch_data(self, cursor: sqlite3.Cursor) -> Iterator[dict[str, Any]]:
                for row in cursor:
                    yield dict(row)

            def close_cursor(self, cursor: sqlite3.Cursor) -> None:
                cursor.close()

            def close_connection(self, connection: sqlite3.Connection) -> None:
                connection.close()


        # Usage:
        adapter = SQLiteAdapter()
        provider = DataBaseDataProvider(
            connection_params={"database": "time_series.db"},
            query="SELECT timestamp, value FROM measurements WHERE sensor_id = ?",
            adapter=adapter,
            params=(42,),
        )
        ```
    """

    @abstractmethod
    def connect(self, connection_params: dict[str, Any]) -> Any:
        """Establish a connection to the database.

        :param connection_params: Dictionary containing connection parameters
                                 (e.g., host, port, username, password, etc.)
        :return: Database connection object
        :raises Exception: If connection fails
        """
        pass

    @abstractmethod
    def execute_query(self, connection: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute a query on the given database connection.

        :param connection: Database connection object
        :param query: SQL query string
        :param params: Query parameters, defaults to ()
        :return: Query result cursor or similar object
        :raises Exception: If query execution fails
        """
        pass

    @abstractmethod
    def fetch_data(self, cursor: Any) -> Iterator[T]:
        """Extract and transform data from the query result.

        :param cursor: Query result cursor
        :return: Iterator yielding data items of type T
        """
        pass

    @abstractmethod
    def close_cursor(self, cursor: Any) -> None:
        """Close the query cursor.

        :param cursor: Query result cursor
        """
        pass

    @abstractmethod
    def close_connection(self, connection: Any) -> None:
        """Close the database connection.

        :param connection: Database connection object
        """
        pass


class DataBaseDataProvider(DataProvider[T], Generic[T]):
    """A data provider that sources time series data from a database.

    This class provides a way to query time series data from any database system
    using adapters that implement the DatabaseAdapter interface. It handles the
    connection lifecycle and streaming of data from database queries.

    :param connection_params: Dictionary containing connection parameters
    :param query: SQL query string to execute
    :param adapter: Database adapter implementation
    :param params: Query parameters, defaults to ()

    Example:
        ```python
        # Using the SQLiteAdapter from the example above
        import sqlite3

        class SQLiteAdapter(DatabaseAdapter[dict[str, Any]]):
            # ... adapter implementation as shown above ...

        # Create a data provider for SQLite database
        provider = DataBaseDataProvider(
            connection_params={"database": "sensors.db"},
            query="SELECT timestamp, value FROM temperature WHERE location_id = ? AND timestamp > ?",
            adapter=SQLiteAdapter(),
            params=("zone-1", "2023-01-01")
        )

        # Use the provider in a processing pipeline
        for record in provider:
            print(f"Time: {record['timestamp']}, Value: {record['value']}")

        # Or connect to a processing pipeline
        pipeline = provider | WindowHandler(60) | AverageHandler()
        ```
    """

    def __init__(
        self,
        connection_params: dict[str, Any],
        query: str,
        adapter: DatabaseAdapter[T],
        params: tuple[Any, ...] = (),
    ) -> None:
        """Initialize a database data provider.

        :param connection_params: Dictionary containing connection parameters
        :param query: SQL query string to execute
        :param adapter: Database adapter implementation
        :param params: Query parameters, defaults to ()
        """
        super().__init__()
        self._connection_params = connection_params
        self._query = query
        self._params = params
        self._adapter = adapter

    @contextmanager
    def _connection_context(self) -> Any:
        """Context manager for database connection lifecycle.

        This method handles the proper setup and teardown of database connections,
        ensuring that resources are properly released even in the case of errors.

        :return: Database cursor or similar query result object
        :raises Exception: If database operations fail
        """
        connection = None
        cursor = None
        try:
            connection = self._adapter.connect(self._connection_params)
            cursor = self._adapter.execute_query(connection, self._query, self._params)
            yield cursor
        finally:
            if cursor is not None:
                self._adapter.close_cursor(cursor)
            if connection is not None:
                self._adapter.close_connection(connection)

    def __iter__(self) -> Iterator[T]:
        """Create an iterator over the query results from the database.

        This method executes the query and yields data items by delegating
        to the adapter's fetch_data method.

        :return: An iterator yielding data items from the database query
        :raises Exception: If database operations fail
        """
        with self._connection_context() as cursor:
            yield from self._adapter.fetch_data(cursor)
