from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from pysatl_tsp.core.data_providers import DatabaseAdapter, DataBaseDataProvider  # скорректируйте импорт!

QUERY: str = "SELECT * FROM table WHERE foo=?"
PARAMS: tuple[str, ...] = ("bar",)
CONNECTION_PARAMS: dict[str, Any] = {"database": "testdb.sqlite"}


class DummyAdapter(DatabaseAdapter[dict[str, Any]]):
    def connect(self, connection_params: dict[str, Any]) -> Any:
        pass

    def execute_query(self, connection: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
        pass

    def fetch_data(self, cursor: Any) -> Iterator[dict[str, Any]]:
        return iter([])

    def close_cursor(self, cursor: Any) -> None:
        pass

    def close_connection(self, connection: Any) -> None:
        pass


@pytest.fixture
def adapter(mocker: MockerFixture) -> Mock:
    return mocker.create_autospec(DummyAdapter, instance=True)


@pytest.fixture
def provider(adapter: DummyAdapter) -> DataBaseDataProvider[dict[str, Any]]:
    return DataBaseDataProvider(
        connection_params=CONNECTION_PARAMS,
        query=QUERY,
        adapter=adapter,
        params=PARAMS,
    )


def test_iter__normal_flow(
    mocker: MockerFixture, provider: DataBaseDataProvider[dict[str, Any]], adapter: DummyAdapter
) -> None:
    connection: Mock = mocker.Mock()
    cursor: Mock = mocker.Mock()
    adapter.connect.return_value = connection  # type: ignore[attr-defined]
    adapter.execute_query.return_value = cursor  # type: ignore[attr-defined]
    data_iter: Iterator[dict[str, Any]] = iter([{"a": 1}, {"a": 2}])
    adapter.fetch_data.return_value = data_iter  # type: ignore[attr-defined]

    result: list[dict[str, Any]] = list(provider)

    assert result == [{"a": 1}, {"a": 2}]
    adapter.connect.assert_called_once_with(CONNECTION_PARAMS)  # type: ignore[attr-defined]
    adapter.execute_query.assert_called_once_with(connection, QUERY, PARAMS)  # type: ignore[attr-defined]
    adapter.fetch_data.assert_called_once_with(cursor)  # type: ignore[attr-defined]
    adapter.close_cursor.assert_called_once_with(cursor)  # type: ignore[attr-defined]
    adapter.close_connection.assert_called_once_with(connection)  # type: ignore[attr-defined]


def test_connection_context_cleanup_on_exception(
    mocker: MockerFixture, provider: DataBaseDataProvider[dict[str, Any]], adapter: DummyAdapter
) -> None:
    connection: Mock = mocker.Mock()
    cursor: Mock = mocker.Mock()
    adapter.connect.return_value = connection  # type: ignore[attr-defined]
    adapter.execute_query.return_value = cursor  # type: ignore[attr-defined]
    adapter.fetch_data.side_effect = Exception("fail!!!")  # type: ignore[attr-defined]

    with pytest.raises(Exception, match="fail!!!"):
        list(provider)
    adapter.close_cursor.assert_called_once_with(cursor)  # type: ignore[attr-defined]
    adapter.close_connection.assert_called_once_with(connection)  # type: ignore[attr-defined]


def test_context_manager_closes_even_if_execute_query_fails(
    mocker: MockerFixture, provider: DataBaseDataProvider[dict[str, Any]], adapter: DummyAdapter
) -> None:
    connection: Mock = mocker.Mock()
    adapter.connect.return_value = connection  # type: ignore[attr-defined]
    adapter.execute_query.side_effect = Exception("query fail")  # type: ignore[attr-defined]

    with pytest.raises(Exception, match="query fail"):
        list(provider)

    adapter.close_cursor.assert_not_called()  # type: ignore[attr-defined]
    adapter.close_connection.assert_called_once_with(connection)  # type: ignore[attr-defined]
