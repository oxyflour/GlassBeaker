import pickle
from fastapi import WebSocket, WebSocketDisconnect
from utils.ros_bridge import conns, calls, subs

async def _ws_(ws: WebSocket) -> None:
    conns.add(ws)
    try:
        await ws.accept()
        while True:
            data = await ws.receive_bytes()
            call, err, ret = pickle.loads(data)
            if call in calls:
                if err:
                    calls[call].set_exception(err)
                else:
                    calls[call].set_result(ret)
            elif call:
                print(f'ignored call {call}')
            elif 'topic' in ret:
                topic = ret['topic']
                for callback in subs.get(topic) or []:
                    callback(topic, ret.get('msg'))
    except WebSocketDisconnect:
        pass
    finally:
        conns.remove(ws)
