import pickle
from fastapi import WebSocket, WebSocketDisconnect
from utils.ros_bridge import bridge

async def _ws_(ws: WebSocket) -> None:
    try:
        await ws.accept()
        bridge.conns.add(ws)
        while True:
            data = await ws.receive_bytes()
            call, err, ret = pickle.loads(data)
            bridge.reply(call, err, ret)
    except WebSocketDisconnect:
        pass
    finally:
        bridge.conns.discard(ws)
