from tempfile import NamedTemporaryFile
from typing import Any, TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysatl_tsp.core.data_providers import DataProvider, FileDataProvider, SimpleDataProvider

T = TypeVar("T")


class TestListDataProvider:
    @given(st.lists(st.integers()))
    def test_iter_returns_same_elements(self, data: list[int]) -> None:
        provider = SimpleDataProvider(data)
        assert list(provider) == data

    @given(st.lists(st.text()))
    def test_text_data(self, data: list[str]) -> None:
        provider = SimpleDataProvider(data)
        assert list(provider) == data

    def test_empty_data(self) -> None:
        provider: SimpleDataProvider[Any] = SimpleDataProvider([])
        assert list(provider) == []

    def test_inheritance(self) -> None:
        assert issubclass(SimpleDataProvider, DataProvider)


class TestFileDataProvider:
    @given(st.lists(st.integers()))
    def test_numeric_data_with_conversion(self, data: list[int]) -> None:
        with NamedTemporaryFile(mode="w") as tmp:
            tmp.write("\n".join(map(str, data)))
            tmp.flush()

            provider = FileDataProvider(tmp.name, lambda x: int(x))

            assert list(provider) == data

    @given(
        st.lists(st.text(alphabet=st.characters(codec="utf-8", exclude_characters="\x00\r\n"), min_size=1), min_size=1)
    )
    def test_text_data_without_conversion(self, data: list[str]) -> None:
        with NamedTemporaryFile(mode="w") as tmp:
            tmp.write("\n".join(data))
            tmp.flush()

            provider = FileDataProvider(tmp.name, lambda x: x)
            result = list(iter(provider))

            print(result, data)
            assert len(result) == len(data)
            assert all(result[i].strip() == data[i].strip() for i in range(len(result)))

    def test_invalid_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            provider = FileDataProvider("nonexistent.file", lambda x: x)
            next(iter(provider))

    def test_handler_exception_propagation(self) -> None:
        def faulty_handler(s: str) -> int:
            return int(s)

        with NamedTemporaryFile(mode="w") as tmp:
            tmp.write("invalid_number")
            tmp.flush()

            provider = FileDataProvider(tmp.name, faulty_handler)
            iterator = iter(provider)

            with pytest.raises(ValueError):
                next(iterator)

    def test_inheritance(self) -> None:
        assert issubclass(FileDataProvider, DataProvider)


class TestDataProvider:
    def test_generic_type(self) -> None:
        provider = SimpleDataProvider([1, 2, 3])
        assert isinstance(provider, DataProvider)
        assert all(isinstance(x, int) for x in provider)
