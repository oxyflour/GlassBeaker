from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


class Timer:
    def __init__(self, interval: float, step_once) -> None:
        self.step_once = step_once
        self.interval = interval
        self.last_run = time.time()

    def step(self, now: float):
        while now - self.last_run > self.interval:
            try:
                self.step_once()
            except Exception:
                traceback.print_exc()
            self.last_run += self.interval


@dataclass(slots=True)
class SessionCall:
    fn: Callable[["Session"], Any]
    future: asyncio.Future[Any]
    world_token: object | None = None


class Session:
    def __init__(self, timeout=120) -> None:
        self.loop = self._bind_owner_loop()

        self.msgs: queue.Queue[dict] = queue.Queue(maxsize=64)
        self.calls: queue.Queue[SessionCall] = queue.Queue()
        self._deferred_calls: list[SessionCall] = []

        self.active = time.time()
        # If the session is idle for more than `timeout` seconds, it will be automatically closed
        self.timeout = timeout

        self.timers: list[Timer] = []
        self._world_owner: object | None = None
        self._running_calls = 0

        self.proc = threading.Thread(target=self.run, daemon=True)
        self.proc.start()

    async def run_sync(self, fn: Callable[["Session"], Any], world_token: object | None = None):
        future = asyncio.get_running_loop().create_future()
        self.calls.put_nowait(SessionCall(fn=fn, future=future, world_token=world_token))
        return await future

    async def call(self, method: str, *args):
        return await self.run_sync(lambda current: current.call_once(method, args))

    def schedule_on_owner_loop(self, coro) -> ConcurrentFuture[Any]:
        if self.loop.is_closed():
            coro.close()
            raise RuntimeError("session owner loop is closed")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    @asynccontextmanager
    async def reserve_world(self):
        token = object()
        await self.run_sync(lambda current: current._claim_world(token))
        try:
            yield token
        finally:
            await asyncio.shield(self.run_sync(lambda current: current._release_world(token), world_token=token))

    def proc_once(self):
        call = self._next_call()
        loop = call.future.get_loop()
        self._running_calls += 1
        self.active = time.time()
        try:
            ret = call.fn(self)
            loop.call_soon_threadsafe(self._set_future_result, call.future, ret)
        except Exception as err:
            loop.call_soon_threadsafe(self._set_future_exception, call.future, err)
        finally:
            self.active = time.time()
            self._running_calls -= 1

    def proc_calls(self):
        while True:
            try:
                self.proc_once()
            except queue.Empty:
                break
            except Exception:
                traceback.print_exc()

    def _next_call(self) -> SessionCall:
        if not self.world_owned():
            if self._deferred_calls:
                return self._deferred_calls.pop(0)
            return self.calls.get(False)

        for index, call in enumerate(self._deferred_calls):
            if call.world_token is self._world_owner:
                return self._deferred_calls.pop(index)

        while True:
            call = self.calls.get(False)
            if call.world_token is self._world_owner:
                return call
            self._deferred_calls.append(call)

    @staticmethod
    def _set_future_result(future: asyncio.Future[Any], result: Any) -> None:
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _set_future_exception(future: asyncio.Future[Any], err: Exception) -> None:
        if not future.done():
            future.set_exception(err)

    def _claim_world(self, token: object) -> None:
        if self._world_owner is not None:
            raise RuntimeError("world is already reserved")
        self._world_owner = token

    def _release_world(self, token: object) -> None:
        if self._world_owner is not token:
            raise RuntimeError("world reservation is not owned by caller")
        self._world_owner = None

    def world_owned(self) -> bool:
        return self._world_owner is not None

    @staticmethod
    def _bind_owner_loop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()

    def is_active(self):
        if self._running_calls > 0 or self.world_owned():
            return True
        return self.timeout <= 0 or time.time() - self.active < self.timeout

    def touch(self):
        self.active = time.time()

    def run(self):
        while self.is_active():
            now = time.time()
            for timer in self.timers:
                timer.step(now)
            self.proc_calls()
            try:
                if self.world_owned():
                    time.sleep(0.001)
                    continue
                self.step_once()
            except Exception:
                traceback.print_exc()
        self.destroy()

    async def stream(self):
        while self.is_active():
            msg = await asyncio.to_thread(lambda: self.msgs.get())
            yield f"data: {json.dumps(msg)}\n\n"
        msg = {"inactive": True}
        yield f"data: {json.dumps(msg)}\n\n"

    def destroy(self):
        pass

    def call_once(self, method: str, args: tuple) -> Any:
        # this runs in session thread
        # DO NOT run heavy computations here, or it will block the session
        return None

    def step_once(self):
        pass
