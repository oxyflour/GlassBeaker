import pickle
import asyncio
import uuid
from fastapi import WebSocket, WebSocketDisconnect

conns: set[WebSocket] = set()
calls: dict[str, asyncio.Future] = { }
subs: dict[str, list] = { }

class Bridge:
    async def call(self, method, args):
        if len(conns):
            ws = list(conns)[0]
            call = str(uuid.uuid4())
            calls[call] = asyncio.Future()
            await ws.send_bytes(pickle.dumps([method, args, call]))
            return await calls[call]
        else:
            raise Exception('no connections now')
    
    async def subscribe(self, topic: str, type: str, callback):
        if not topic in subs:
            subs[topic] = []
        subs[topic].append(callback)
        await self.call('subscribe', [topic, type])

bridge = Bridge()
