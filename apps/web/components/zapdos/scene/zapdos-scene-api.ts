import { Object3D, Quaternion, Vector3 } from "three";

export interface BodyVisual {
  name: string;
  label: string;
  editable: boolean;
  selectable: boolean;
  movable: boolean;
  selectionBody: string | null;
  matrix: number[];
}

export interface MeshVisual {
  name: string;
  body: string | null;
  kind: string;
  color: [number, number, number, number] | null;
  matrix?: number[];
  localMatrix?: number[];
  size?: number[];
  mesh?: string;
  texture?: string;
}

export interface SceneVisuals {
  bodies: BodyVisual[];
  meshes: MeshVisual[];
}

export interface BodyPosePayload {
  pos: [number, number, number];
  quat: [number, number, number, number];
}

const worldPosition = new Vector3();
const worldQuaternion = new Quaternion();

export function createGetVisualRequest(): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  };
}

export function createSetBodyPoseRequest(body: string, pose: BodyPosePayload): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([body, pose.pos, pose.quat]),
  };
}

export function buildBodyPosePayload(object: Object3D): BodyPosePayload {
  object.updateWorldMatrix(true, false);
  object.getWorldPosition(worldPosition);
  object.getWorldQuaternion(worldQuaternion);
  return {
    pos: [worldPosition.x, worldPosition.y, worldPosition.z],
    quat: [worldQuaternion.w, worldQuaternion.x, worldQuaternion.y, worldQuaternion.z],
  };
}

export async function getSceneVisual(sess: string): Promise<SceneVisuals> {
  const response = await fetch(`/python/zapdos/${sess}/call/get_visual`, createGetVisualRequest());
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as SceneVisuals;
}

export async function setSceneBodyPose(sess: string, body: string, pose: BodyPosePayload): Promise<void> {
  const response = await fetch(
    `/python/zapdos/${sess}/call/set_body_pose`,
    createSetBodyPoseRequest(body, pose),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
}
