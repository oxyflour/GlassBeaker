from fastapi import Request
from api.zapdos.sessions import get_session
from fastapi.responses import StreamingResponse

async def _method_(req: Request):
    sess = req.path_params['session']
    method = req.path_params['method']
    if method == 'start':
        return StreamingResponse(
            get_session(sess).stream(),
            media_type="text/event-stream")
    else:
        args = await req.json()
        return await get_session(sess).call(method, *args)
