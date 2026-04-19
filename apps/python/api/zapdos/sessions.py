from pathlib import Path
import mujoco # type: ignore
import numpy as np

from utils.session import Session


ROBOT_URDF = Path(r'tmp\URDF-galaxea-main\R1\urdf\r1_v2_1_0.urdf')


def create_abs_urdf(urdf: Path) -> Path:
    xml = urdf.read_text(encoding='utf-8')
    xml = xml.replace('package://r1_v2_1_0/', '../')
    resolved = urdf.with_name(urdf.name + '.abs.urdf')
    resolved.write_text(xml, encoding='utf-8')
    return resolved

def create_xml():
    urdf = ROBOT_URDF.resolve()
    abs_urdf = create_abs_urdf(urdf)
    urdf_model = mujoco.MjModel.from_xml_path(str(abs_urdf))           # type: ignore
    abs_xml = str(abs_urdf.with_suffix('.xml'))
    mujoco.mj_saveLastXML(abs_xml, urdf_model) # type: ignore
    xml_str = f'''
    <mujoco>
        <include file="{abs_xml}"/>
        <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        </worldbody>
    </mujoco>
    '''
    asset_root = abs_urdf.parent
    return xml_str, asset_root

class ZapdosGeometry:
    def __init__(self, name: str, geom_id: int, body: str, color: list[float], abs_path: Path):
        self.name = name
        self.geom_id = geom_id
        self.body = body
        self.color = color
        self.abs_path = abs_path

class ZapdosSession(Session):
    def __init__(self, sess: str) -> None:
        self.sess = sess

        xml_str, asset_root = create_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str) # type: ignore
        self.data = mujoco.MjData(self.model)                # type: ignore
        mujoco.mj_step(self.model, self.data)                # type: ignore

        self.geoms: dict[str, ZapdosGeometry] = { }
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH): # type: ignore
                mesh_id = int(self.model.geom_dataid[geom_id])
                rel_path = self.decode_path(int(self.model.mesh_pathadr[mesh_id]))
                body_id = int(self.model.geom_bodyid[geom_id])
                name = f'geom-{geom_id}' + Path(rel_path).suffix.lower()
                self.geoms[name] = ZapdosGeometry(
                    name=name,
                    geom_id=geom_id,
                    body=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or 'world', # type: ignore
                    color=[float(value) for value in self.model.geom_rgba[geom_id]],
                    abs_path=(asset_root / rel_path).resolve(),
                )

        super().__init__()

    def decode_path(self, start: int) -> str:
        end = self.model.paths.find(b'\x00', start)
        if end < 0:
            end = len(self.model.paths)
        return self.model.paths[start:end].decode('utf-8')

    def quat_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def pose_matrix(self, pos: np.ndarray, quat: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
        matrix = self.quat_matrix(quat)
        if scale is not None:
            scaled = np.eye(4)
            scaled[0, 0], scaled[1, 1], scaled[2, 2] = [float(v) for v in scale]
            matrix = matrix @ scaled
        matrix[:3, 3] = [float(v) for v in pos]
        return matrix

    def geom_source_matrix(self, geom_id: int) -> list[float]:
        mesh_id = int(self.model.geom_dataid[geom_id])
        geom_world = np.eye(4)
        geom_world[:3, :3] = np.array(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        geom_world[:3, 3] = np.array(self.data.geom_xpos[geom_id], dtype=float)
        mesh_local = self.pose_matrix(
            self.model.mesh_pos[mesh_id],
            self.model.mesh_quat[mesh_id],
            self.model.mesh_scale[mesh_id],
        )
        source_world = geom_world @ np.linalg.inv(mesh_local)
        return [float(v) for v in source_world.T.reshape(-1)]

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
        # We serve source STL files, so remove MuJoCo's internal mesh normalization transform.
        return { name: self.geom_source_matrix(geom.geom_id) for name, geom in self.geoms.items() }

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
