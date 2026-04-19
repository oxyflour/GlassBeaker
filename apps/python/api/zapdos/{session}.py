from fastapi import Request
from fastapi.responses import StreamingResponse

from api.zapdos.sessions import get_session

async def call(req: Request):
    sess = req.path_params['session']
    args = await req.json()
    return await get_session(sess).call(*args)

async def start(req: Request):
    sess = req.path_params['session']
    return StreamingResponse(
        get_session(sess).stream(),
        media_type="text/event-stream")
