import asyncio
import json
import queue
import threading
import time
import traceback
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
            except:
                traceback.print_exc()
            self.last_run += self.interval

class Session:
    def __init__(self, timeout=120) -> None:
        self.loop = asyncio.get_event_loop()

        self.msgs: queue.Queue[dict] = queue.Queue(maxsize=64)
        self.calls: queue.Queue[tuple[str, tuple, asyncio.Future]] = queue.Queue()

        self.active = time.time()
        # If the session is idle for more than `timeout` seconds, it will be automatically closed
        self.timeout = timeout

        self.timers: list[Timer] = []

        self.proc = threading.Thread(target=self.run, daemon=True)
        self.proc.start()
    
    async def call(self, method: str, *args):
        res = asyncio.Future()
        self.calls.put_nowait((method, args, res))
        return await res

    def proc_call(self):
        method, args, res = self.calls.get(False)
        self.active = time.time()
        try:
            ret = self.call_once(method, args)
            self.loop.call_soon_threadsafe(res.set_result, ret)
        except Exception as err:
            self.loop.call_soon_threadsafe(res.set_exception, err)
    
    def is_active(self):
        return self.timeout <= 0 or time.time() - self.active < self.timeout
    
    def run(self):
        while self.is_active():
            now = time.time()
            for timer in self.timers:
                timer.step(now)
            try:
                self.proc_call()
            except queue.Empty:
                pass
            except:
                traceback.print_exc()
            try:
                self.step_once()
            except:
                traceback.print_exc()
        self.destroy()
    
    async def stream(self):
        while self.is_active():
            try:
                msg = self.msgs.get(False)
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
    
    def destroy(self):
        pass
    
    def call_once(self, method: str, args: tuple) -> Any:
        return None

    def step_once(self):
        pass
