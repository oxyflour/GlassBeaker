import pickle
import asyncio
import uuid
from fastapi import WebSocket


class BridgeUnavailable(RuntimeError):
    pass


class Bridge:
    def __init__(self) -> None:
        self.conns: set[WebSocket] = set()
        self.calls: dict[str, asyncio.Future] = {}
        self.subs: dict[str, set] = {}

    async def call(self, method, args):
        if not self.conns:
            raise BridgeUnavailable("no connections now")
        call = str(uuid.uuid4())
        payload = pickle.dumps([method, args, call])
        waiter = asyncio.get_running_loop().create_future()
        self.calls[call] = waiter
        try:
            while self.conns:
                ws = next(iter(self.conns))
                try:
                    await ws.send_bytes(payload)
                except Exception as exc:
                    self.conns.discard(ws)
                    if not self.conns:
                        raise BridgeUnavailable("no connections now") from exc
                    continue
                return await waiter
            raise BridgeUnavailable("no connections now")
        finally:
            self.calls.pop(call, None)
            if not waiter.done():
                waiter.cancel()
    
    def unsubscribe(self, topic: str, callback):
        subs = self.subs.get(topic)
        if subs and callback in subs:
            subs.remove(callback)
            if not subs:
                self.subs.pop(topic, None)
    
    async def subscribe(self, topic: str, type: str, callback):
        subs = self.subs.setdefault(topic, set())
        subs.add(callback)
        try:
            await self.call("subscribe", [topic, type])
        except Exception:
            self.unsubscribe(topic, callback)
            raise
    
    def reply(self, call: str, err, ret):
        if call.startswith('ros:'):
            topic = call[len('ros:'):]
            for callback in self.subs.get(topic) or []:
                callback(topic, ret)
        else:
            item = self.calls.get(call)
            if item is None or item.done():
                return
            if err:
                item.set_exception(err)
            else:
                item.set_result(ret)

bridge = Bridge()
