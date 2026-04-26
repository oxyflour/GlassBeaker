import pickle
import asyncio
import uuid
from fastapi import WebSocket

class Bridge:
    conns: set[WebSocket] = set()
    calls: dict[str, asyncio.Future] = { }
    subs: dict[str, set] = { }

    async def call(self, method, args):
        if len(self.conns):
            ws = list(self.conns)[0]
            call = str(uuid.uuid4())
            self.calls[call] = asyncio.Future()
            await ws.send_bytes(pickle.dumps([method, args, call]))
            return await self.calls[call]
        else:
            raise Exception('no connections now')
    
    def unsubscribe(self, topic: str, callback):
        subs = self.subs.get(topic)
        if subs and callback in subs:
            subs.remove(callback)
    
    async def subscribe(self, topic: str, type: str, callback):
        if not topic in self.subs:
            self.subs[topic] = set()
        self.subs[topic].add(callback)
        await self.call('subscribe', [topic, type])
    
    def reply(self, call: str, err, ret):
        if call in self.calls:
            item = self.calls[call]
            if err:
                item.set_exception(err)
            else:
                item.set_result(ret)
        elif 'topic' in ret:
            topic = ret['topic']
            for callback in self.subs.get(topic) or []:
                callback(topic, ret.get('msg'))
        elif call:
            print(f'ignored call {call}')

bridge = Bridge()
