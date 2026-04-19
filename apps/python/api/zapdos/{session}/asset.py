from fastapi import Request, HTTPException, Request
from fastapi.responses import FileResponse

from api.zapdos.sessions import get_session

async def _name_(req: Request):
    sess = req.path_params['session']
    name = req.path_params['name']
    geom = get_session(sess).visuals.get(name)
    if geom is None:
        raise HTTPException(status_code=404, detail='Geom not found')
    if geom.abs_path is None:
        raise HTTPException(status_code=400, detail='Geom has no asset file')
    path = geom.abs_path
    return FileResponse(path, media_type='model/stl')
