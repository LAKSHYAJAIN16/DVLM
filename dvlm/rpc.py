"""Minimal multiplexed async RPC over persistent TCP connections."""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Awaitable, Callable

from .wire import read_frame, write_frame

log = logging.getLogger(__name__)

Handler = Callable[..., Awaitable[Any]]


class RemoteError(Exception):
    """The remote handler raised; the peer is alive but rejected the request."""


class RpcServer:
    def __init__(self, handlers: dict[str, Handler], host: str = "127.0.0.1", port: int = 0):
        self.handlers = handlers
        self.host, self.port = host, port
        self._server: asyncio.base_events.Server | None = None
        self._conns: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            for writer in list(self._conns):
                writer.close()
            await self._server.wait_closed()
            self._server = None

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._conns.add(writer)
        lock = asyncio.Lock()
        tasks: set[asyncio.Task] = set()
        try:
            while True:
                msg = await read_frame(reader)
                task = asyncio.create_task(self._dispatch(msg, writer, lock))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            self._conns.discard(writer)
            writer.close()

    async def _dispatch(self, msg: dict, writer: asyncio.StreamWriter, lock: asyncio.Lock) -> None:
        req_id = msg.get("id")
        try:
            handler = self.handlers[msg["method"]]
            reply = {"id": req_id, "result": await handler(**msg.get("params", {}))}
        except Exception as e:  # report every handler failure to the caller
            log.debug("rpc %s failed", msg.get("method"), exc_info=True)
            reply = {"id": req_id, "error": f"{type(e).__name__}: {e}"}
        try:
            async with lock:
                await write_frame(writer, reply)
        except ConnectionError:
            pass


class RpcClient:
    def __init__(self, address: str):
        self.address = address
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count()
        self._lock = asyncio.Lock()
        self._recv_task: asyncio.Task | None = None

    @property
    def closed(self) -> bool:
        return self._writer is None or self._writer.is_closing()

    async def connect(self, timeout: float = 10.0) -> None:
        host, port = self.address.rsplit(":", 1)
        self._reader, self._writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), timeout)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            while True:
                msg = await read_frame(self._reader)
                fut = self._pending.pop(msg["id"], None)
                if fut is None or fut.done():
                    continue
                if "error" in msg:
                    fut.set_exception(RemoteError(msg["error"]))
                else:
                    fut.set_result(msg["result"])
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as e:
            self._fail_all(ConnectionError(f"lost connection to {self.address}: {e!r}"))
        finally:
            if self._writer is not None:
                self._writer.close()

    def _fail_all(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    async def call(self, method: str, timeout: float | None = 60.0, **params) -> Any:
        if self.closed:
            raise ConnectionError(f"not connected to {self.address}")
        req_id = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            async with self._lock:
                await write_frame(self._writer, {"id": req_id, "method": method, "params": params})
        except (ConnectionError, OSError) as e:
            self._pending.pop(req_id, None)
            raise ConnectionError(f"send to {self.address} failed: {e!r}") from e
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(req_id, None)

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self._writer is not None:
            self._writer.close()
        self._fail_all(ConnectionError("client closed"))


class ConnectionPool:
    """One persistent connection per peer address, reconnected on demand."""

    def __init__(self):
        self._clients: dict[str, RpcClient] = {}

    async def get(self, address: str) -> RpcClient:
        client = self._clients.get(address)
        if client is None or client.closed:
            client = RpcClient(address)
            try:
                await client.connect()
            except (OSError, asyncio.TimeoutError) as e:
                raise ConnectionError(f"cannot connect to {address}: {e!r}") from e
            self._clients[address] = client
        return client

    async def call(self, address: str, method: str, timeout: float | None = 60.0, **params) -> Any:
        return await (await self.get(address)).call(method, timeout=timeout, **params)

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
