import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysatl_tsp.core.data_providers import WebSocketDataProvider


@pytest.fixture
def provider() -> WebSocketDataProvider:
    return WebSocketDataProvider("ws://test", {"action": "subscribe"})


@pytest.mark.asyncio
async def test_receiver_puts_json(provider: WebSocketDataProvider) -> None:
    fake_msgs = [
        json.dumps({"foo": "bar"}),
        json.dumps({"baz": 42}),
        "not a json",
    ]

    fake_ws = AsyncMock()
    fake_ws.__aenter__.return_value = fake_ws
    fake_ws.__aiter__.return_value = (m for m in fake_msgs)
    fake_ws.send = AsyncMock()
    with patch("websockets.connect", return_value=fake_ws):
        await provider._receiver()

    assert provider._iterator_queue.qsize() == len(fake_msgs)
    assert provider._iterator_queue.get() == fake_msgs[0]
    assert provider._iterator_queue.get() == fake_msgs[1]
    assert provider._iterator_queue.get() == fake_msgs[2]

    fake_ws.send.assert_called_once_with(json.dumps({"action": "subscribe"}))


def test_iter_yields_items_from_queue(provider: WebSocketDataProvider) -> None:
    provider._iterator_queue.put("one")
    provider._iterator_queue.put("two")

    with patch.object(provider, "_thread", None):
        it = iter(provider)
        value1 = next(it)
        value2 = next(it)
        provider._stop_event.set()
        rest = list(it)
    assert [value1, value2] == ["one", "two"]
    assert rest == []


def test_close_stops_thread(provider: WebSocketDataProvider) -> None:
    mock_thread = MagicMock()
    provider._thread = mock_thread
    provider.close()
    mock_thread.join.assert_called_once()


@pytest.mark.asyncio
async def test_receiver_swallows_exceptions(provider: WebSocketDataProvider) -> None:
    with patch("websockets.connect", side_effect=Exception("fail")):
        await provider._receiver()
