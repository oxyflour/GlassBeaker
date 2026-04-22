import asyncio
import json
import queue
import threading
import time
import traceback
from typing import Any

class Session:
    def __init__(self, timeout=120) -> None:
        self.loop = asyncio.get_event_loop()

        self.msgs: queue.Queue[dict] = queue.Queue()
        self.calls: queue.Queue[tuple[str, tuple, asyncio.Future]] = queue.Queue()

        self.active = time.time()
        # If the session is idle for more than `timeout` seconds, it will be automatically closed
        self.timeout = timeout

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
            ret = self.on_call(method, args)
            self.loop.call_soon_threadsafe(res.set_result, ret)
        except Exception as err:
            self.loop.call_soon_threadsafe(res.set_exception, err)
    
    def run(self):
        while self.timeout <= 0 or time.time() - self.active < self.timeout:
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
    
    async def stream(self):
        while self.timeout <= 0 or time.time() - self.active < self.timeout:
            try:
                msg = self.msgs.get(False)
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                await asyncio.sleep(0.1)
    
    def on_call(self, method: str, args: tuple) -> Any:
        return None

    def step_once(self):
        pass
