import asyncio
import json
import queue
import threading
from collections.abc import Iterator
from typing import (
    Any,
)

import websockets

from .abstract import DataProvider


class WebSocketDataProvider(DataProvider[str]):
    """A data provider that streams time series data from a WebSocket connection.

    This class establishes a WebSocket connection to a specified URI and streams
    received messages into the processing pipeline. It handles the WebSocket connection
    in a separate thread to avoid blocking the main processing flow and provides
    a clean streaming interface through the standard iterator protocol.

    :param uri: WebSocket endpoint URI
    :param subscribe_message: Optional message to send after connection to subscribe to specific data streams

    Example:
        ```python
        # Example: Connecting to Bybit WebSocket API for Bitcoin price data

        bybit_provider = WebSocketDataProvider(
            uri="wss://stream.bybit.com/v5/public/spot",
            subscribe_message={"op": "subscribe", "args": ["tickers.BTCUSDT"]},
        )

        try:
            for message in bybit_provider:
                data = json.loads(message)
                if "data" in data and data.get("topic") == "tickers.BTCUSDT":
                    price_data = data["data"]
                    print(f"BTC/USDT: {price_data['lastPrice']} (Time: {price_data['timestamp']})")
        except KeyboardInterrupt:
            bybit_provider.close()
        ```
    """

    def __init__(self, uri: str, subscribe_message: dict[str, Any] | None = None) -> None:
        """Initialize a WebSocket data provider.

        :param uri: WebSocket endpoint URI
        :param subscribe_message: Optional message to send after connection to subscribe to specific data streams
        """
        super().__init__()
        self._uri = uri
        self._subscribe_message = subscribe_message
        self._iterator_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __iter__(self) -> Iterator[str]:
        """Create an iterator over the messages received from the WebSocket.

        This method starts a background thread to handle the WebSocket connection
        if it's not already running, and yields messages as they are received.

        :return: An iterator yielding message strings from the WebSocket
        """
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._thread_main, daemon=True)
            self._thread.start()
        while not self._stop_event.is_set():
            try:
                item = self._iterator_queue.get(timeout=1)
                yield item
            except queue.Empty:
                continue

    def _thread_main(self) -> None:
        """Entry point for the background thread.

        This method runs the asyncio event loop that handles the WebSocket connection.
        """
        asyncio.run(self._receiver())

    async def _receiver(self) -> None:
        """Asynchronous method to handle WebSocket communication.

        This method establishes the WebSocket connection, sends the subscription message
        if provided, and places received messages into the queue for the iterator.
        """
        try:
            async with websockets.connect(self._uri) as ws:
                if self._subscribe_message is not None:
                    await ws.send(json.dumps(self._subscribe_message))
                async for msg in ws:
                    try:
                        if msg is not None:
                            self._iterator_queue.put(str(msg))
                    except Exception:
                        continue
        except Exception:
            pass

    def close(self) -> None:
        """Close the WebSocket connection and stop the background thread.

        This method should be called when the provider is no longer needed
        to release resources properly.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
