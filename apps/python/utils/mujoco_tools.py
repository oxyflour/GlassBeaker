import os
import re
import mujoco # type: ignore
import asyncio
import numpy as np
from pathlib import Path

def decode_mesh_path(model, mesh_id: int):
    start = int(model.mesh_pathadr[mesh_id])
    end = model.paths.find(b'\x00', start)
    if end < 0:
        end = len(model.paths)
    return Path(model.paths[start:end].decode('utf-8'))

def decode_texture_path(model, tex_id: int):
    start = int(model.tex_pathadr[tex_id])
    end = model.paths.find(b'\x00', start)
    if end < 0:
        end = len(model.paths)
    return Path(model.paths[start:end].decode('utf-8'))

def quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

def pose_matrix(pos: np.ndarray, quat: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    matrix = quat_matrix(quat)
    if scale is not None:
        scaled = np.eye(4)
        scaled[0, 0], scaled[1, 1], scaled[2, 2] = [float(v) for v in scale]
        matrix = matrix @ scaled
    matrix[:3, 3] = [float(v) for v in pos]
    return matrix

def flatten_matrix(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in matrix.T.reshape(-1)]

def geom_world_pose(data, geom_id: int) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.array(data.geom_xpos[geom_id], dtype=float)
    return matrix

def geom_size(model, geom_id: int, kind: str) -> list[float]:
    size = np.array(model.geom_size[geom_id], dtype=float)
    if kind == 'plane':
        return [float(max(size[0] * 2, 1e-3)), float(max(size[1] * 2, 1e-3))]
    if kind in {'box', 'ellipsoid'}:
        return [float(max(size[0] * 2, 1e-3)), float(max(size[1] * 2, 1e-3)), float(max(size[2] * 2, 1e-3))]
    if kind == 'sphere':
        return [float(max(size[0], 1e-3))]
    if kind in {'capsule', 'cylinder'}:
        return [float(max(size[0], 1e-3)), float(max(size[1] * 2, 1e-3))]
    raise ValueError(f'Unsupported primitive kind: {kind}')

async def fix_urdf_path(urdf: Path) -> Path:
    xml = urdf.read_text(encoding='utf-8')
    converted = urdf.with_suffix('.converted.v5.urdf')
    if not os.path.exists(converted):
        script = os.path.normpath(f"{__file__}/../../../../scripts/convert_usd.py")
        blender = os.environ.get('BLENDER_BINARY', rf"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
        cmd = [blender, '--background', '--python', script, '--', str(urdf.parent / 'usd')]
        print('RUN: ', ' '.join([f'"{item}"' for item in cmd]))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        xml = re.sub(r'filename="\./usd/(.*)\.usd"', r'filename="./usd/\1.stl"', xml)
        converted.write_text(xml, encoding='utf-8')
    return converted

async def create_xml(input: str):
    if input.endswith('.xml'):
        abs_xml = Path(input).resolve()

    elif input.endswith('.urdf'):
        abs_xml = Path(input).with_suffix('.xml').resolve()
        print(f'check {abs_xml} for {input}')
        if not abs_xml.exists():
            abs_urdf = await fix_urdf_path(Path(input))
            urdf_model = mujoco.MjModel.from_xml_path(str(abs_urdf))           # type: ignore
            mujoco.mj_saveLastXML(abs_xml, urdf_model) # type: ignore

    elif input.endswith('.usda'):
        abs_xml = Path(input).with_suffix('.xml').resolve()
        print(f'check {abs_xml} for {input}')
        if not abs_xml.exists():
            script = os.path.normpath(f"{__file__}/../../../../utils/usd_to_mjcf.py")
            cmd = ['python', '-u', script, input, abs_xml, '--model-name', 'r1pro']
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

    else:
        raise Exception(f'unknown input file format: {input}')

    xml_str = f'''
    <mujoco>
        <option timestep="0.001" />
        <include file="{abs_xml}"/>
        <worldbody>
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        </worldbody>
    </mujoco>
    '''
    out_xml = abs_xml.parent / 'out.xml'
    out_xml.write_text(xml_str)
    return out_xml
