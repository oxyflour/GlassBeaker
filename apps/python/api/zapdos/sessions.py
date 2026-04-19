from pathlib import Path
import mujoco # type: ignore

from utils.session import Session


ROBOT_URDF = Path(r'tmp\URDF-galaxea-main\R1\urdf\r1_v2_1_0.urdf')


def create_abs_urdf(urdf: Path) -> Path:
    xml = urdf.read_text(encoding='utf-8')
    xml = xml.replace('package://r1_v2_1_0/', '../')
    resolved = urdf.with_name(urdf.name + '.abs.urdf')
    resolved.write_text(xml, encoding='utf-8')
    return resolved

class ZapdosGeometry:
    def __init__(self, name: str, body_id: int, body: str, color: list[float], abs_path: Path):
        self.name = name
        self.body_id = body_id
        self.body = body
        self.color = color
        self.abs_path = abs_path

class ZapdosSession(Session):
    def __init__(self, sess: str) -> None:
        self.sess = sess
        self.urdf = ROBOT_URDF.resolve()
        self.robot_root = self.urdf.parent.parent.resolve()
        abs_urdf = create_abs_urdf(self.urdf)

        self.model = mujoco.MjModel.from_xml_path(str(abs_urdf)) # type: ignore
        self.data = mujoco.MjData(self.model)                         # type: ignore
        mujoco.mj_step(self.model, self.data)                         # type: ignore

        self.geoms: dict[str, ZapdosGeometry] = { }
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH): # type: ignore
                mesh_id = int(self.model.geom_dataid[geom_id])
                rel_path = self.decode_path(int(self.model.mesh_pathadr[mesh_id]))
                body_id = int(self.model.geom_bodyid[geom_id])
                name = f'geom-{geom_id}' + Path(rel_path).suffix.lower()
                self.geoms[name] = ZapdosGeometry(
                    name=name,
                    body_id=body_id,
                    body=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world', # type: ignore
                    color=[float(value) for value in self.model.geom_rgba[geom_id]],
                    abs_path=(abs_urdf.parent / rel_path).resolve(),
                )

        super().__init__()

    def decode_path(self, start: int) -> str:
        end = self.model.paths.find(b'\x00', start)
        if end < 0:
            end = len(self.model.paths)
        return self.model.paths[start:end].decode('utf-8')

    def body_matrix(self, body_id: int) -> list[float]:
        xmat = self.data.xmat[body_id]
        xpos = self.data.xpos[body_id]
        return [
            float(xmat[0]), float(xmat[3]), float(xmat[6]), 0.0,
            float(xmat[1]), float(xmat[4]), float(xmat[7]), 0.0,
            float(xmat[2]), float(xmat[5]), float(xmat[8]), 0.0,
            float(xpos[0]), float(xpos[1]), float(xpos[2]), 1.0,
        ]

    def get_visual(self) -> list[dict]:
        poses = self.get_pose()
        return [{
            'name': name,
            'body': geom.body,
            'color': geom.color,
            'matrix': poses[name],
            'url': f'/python/zapdos/{self.sess}/asset/{name}',
        } for name, geom in self.geoms.items()]

    def get_pose(self) -> dict[str, list[float]]:
        # We serve the original STL files, whose vertices are already expressed in the body frame.
        return { name: self.body_matrix(geom.body_id) for name, geom in self.geoms.items() }

    def on_call(self, args: list):
        cmd, *args = args
        if cmd == 'ping':
            return 'pong'
        elif cmd == 'get_visual':
            return self.get_visual()
        elif cmd == 'get_pose':
            return self.get_pose()
        return super().on_call(args)
    
    def step_once(self):
        if not self.msgs.full():
            self.msgs.put_nowait({ 'pose': self.get_pose() })
        mujoco.mj_step(self.model, self.data) # type: ignore
        return super().step_once()

sessions: dict[str, ZapdosSession] = { }


def get_session(sess: str) -> ZapdosSession:
    if sess not in sessions:
        sessions[sess] = ZapdosSession(sess)
    return sessions[sess]
