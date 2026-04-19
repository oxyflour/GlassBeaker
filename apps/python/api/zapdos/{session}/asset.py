from fastapi import Request, HTTPException, Request
from fastapi.responses import FileResponse

from api.zapdos.sessions import get_session

async def _name_(req: Request):
    sess = req.path_params['session']
    name = req.path_params['name']
    try:
        path = get_session(sess).geoms[name].abs_path
    except ValueError as err:
        raise HTTPException(status_code=400, detail='Invalid geom query parameter') from err
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return FileResponse(path, media_type='model/stl')