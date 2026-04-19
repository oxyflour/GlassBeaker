from time import time

from fastapi import Request
import mujoco # type: ignore
from fastapi.responses import StreamingResponse

from utils.session import Session

def replace_package_uri(urdf: str) -> str:
    with open(urdf, 'r') as fn:
        xml = fn.read()
    xml = xml.replace('package://r1_v2_1_0/', '../')
    urdf = urdf + '.abs.urdf'
    with open(urdf, 'w') as fn:
        fn.write(xml)
    return urdf


class ZapdosSession(Session):
    def __init__(self) -> None:
        urdf = rf'tmp\URDF-galaxea-main\R1\urdf\r1_v2_1_0.urdf'
        urdf = replace_package_uri(urdf)
        self.model = mujoco.MjModel.from_xml_path(urdf) # type: ignore
        self.data = mujoco.MjData(self.model)           # type: ignore
        mujoco.mj_step(self.model, self.data)           # type: ignore

        self.last_sent = time()
        super().__init__()
    
    def get_visual(self) -> list[dict]:
        # TODO: return visual data, e.g. obj files, textures, etc.
        return [{
            'name': 'robot',
        }]
    
    def get_pose(self):
        # TODO: return the pose matrix of the robot to render in THREE.js
        return { }
    
    def on_call(self, args: list):
        cmd, *args = args
        if cmd == 'ping':
            return 'pong'
        elif cmd == 'get_visual':
            return self.get_visual()
        return super().on_call(args)
    
    def step_once(self):
        if not self.msgs.full() and time() - self.last_sent > 0.1:
            self.msgs.put_nowait({ 'pose': self.get_pose() })
            self.last_sent = time()

        mujoco.mj_step(self.model, self.data) # type: ignore
        return super().step_once()

sessions: dict[str, ZapdosSession] = { }

async def call(req: Request):
    sess = req.path_params['session']
    args = await req.json()
    if not sess in sessions:
        sessions[sess] = ZapdosSession()
    return await sessions[sess].call(*args)

async def start(req: Request):
    sess = req.path_params['session']
    if not sess in sessions:
        sessions[sess] = ZapdosSession()
    return StreamingResponse(
        sessions[sess].stream(),
        media_type="text/event-stream")