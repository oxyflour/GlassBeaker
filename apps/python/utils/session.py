import asyncio
import json
import queue
import threading
import time
from typing import Any

class Session:
    def __init__(self, timeout=120) -> None:
        self.msgs: asyncio.Queue[dict] = asyncio.Queue()
        self.calls: queue.Queue[tuple[list, asyncio.Future]] = queue.Queue()

        self.active = time.time()
        # If the session is idle for more than `timeout` seconds, it will be automatically closed
        self.timeout = timeout

        self.proc = threading.Thread(target=self.run, daemon=True)
        self.proc.start()
    
    async def call(self, *args: list):
        res = asyncio.Future()
        self.calls.put_nowait((list(args), res))
        return await res

    def proc_call(self):
        try:
            while not self.calls.empty():
                args, res = self.calls.get_nowait()
                self.active = time.time()
                try:
                    ret = self.on_call(args)
                    res.set_result(ret)
                except Exception as err:
                    res.set_exception(err)
        except queue.Empty:
            pass
    
    def run(self):
        while time.time() - self.active < self.timeout:
            self.proc_call()
            try:
                self.step_once()
            except Exception as err:
                print('session error', err)
    
    async def stream(self):
        while time.time() - self.active < self.timeout:
            msg = await self.msgs.get()
            yield f"data: {json.dumps(msg)}\n\n"
    
    def on_call(self, args: list) -> Any:
        return None

    def step_once(self):
        pass
