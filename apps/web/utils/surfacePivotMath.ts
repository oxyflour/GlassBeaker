import {
  Camera,
  Intersection,
  Matrix4,
  Mesh,
  Quaternion,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
} from "three";

export interface SurfacePivotRig {
  position: Vector3;
  quaternion: Quaternion;
  pivot: Vector3;
}

export interface SurfacePivotGesture {
  button: number;
  start: Vector2;
  pendingPivot: Vector3 | null;
}

export interface SurfacePivotUpdateConfig {
  rotateSpeed?: number;
  panSpeed?: number;
}

const dragThresholdPx = 5;

export function createRigState(position: Vector3, quaternion: Quaternion, pivot: Vector3): SurfacePivotRig {
  return {
    position: position.clone(),
    quaternion: quaternion.clone(),
    pivot: pivot.clone(),
  };
}

export function startGesture(button: number, start: Vector2, pendingPivot: Vector3 | null): SurfacePivotGesture {
  return {
    button,
    start: start.clone(),
    pendingPivot: pendingPivot?.clone() ?? null,
  };
}

export function finishGesture(rig: SurfacePivotRig, gesture: SurfacePivotGesture): SurfacePivotRig {
  if (gesture.button !== 0 || !gesture.pendingPivot) {
    return createRigState(rig.position, rig.quaternion, rig.pivot);
  }

  return {
    position: rig.position.clone(),
    quaternion: rig.quaternion.clone(),
    pivot: gesture.pendingPivot.clone(),
  };
}

export function isDragGesture(start: Vector2, current: Vector2, thresholdPx: number): boolean {
  return start.distanceTo(current) > thresholdPx;
}

export function updateGesture(
  rig: SurfacePivotRig,
  gesture: SurfacePivotGesture,
  point: Vector2,
  viewport: { width: number; height: number },
  _fov: number,
  config: SurfacePivotUpdateConfig,
): SurfacePivotRig {
  const committed = finishGesture(rig, gesture);
  if (!isDragGesture(gesture.start, point, dragThresholdPx)) {
    return committed;
  }

  const dx = point.x - gesture.start.x;
  const dy = point.y - gesture.start.y;

  if (gesture.button === 1) {
    const panSpeed = config.panSpeed ?? 1;
    const right = new Vector3(1, 0, 0).applyQuaternion(committed.quaternion).multiplyScalar((-dx / viewport.width) * panSpeed);
    const up = new Vector3(0, 1, 0).applyQuaternion(committed.quaternion).multiplyScalar((-dy / viewport.height) * panSpeed);
    const offset = right.add(up);
    return {
      position: committed.position.clone().add(offset),
      quaternion: committed.quaternion.clone(),
      pivot: committed.pivot.clone().add(offset),
    };
  }

  if (gesture.button === 0) {
    const rotateSpeed = config.rotateSpeed ?? 1;
    const yaw = (-dx / viewport.width) * Math.PI * rotateSpeed;
    const pitch = (-dy / viewport.height) * Math.PI * rotateSpeed;
    const offset = committed.position.clone().sub(committed.pivot);

    const yawQuat = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), yaw);
    offset.applyQuaternion(yawQuat);

    const right = new Vector3(1, 0, 0).applyQuaternion(committed.quaternion).applyQuaternion(yawQuat);
    const pitchQuat = new Quaternion().setFromAxisAngle(right.normalize(), pitch);
    offset.applyQuaternion(pitchQuat);

    const position = committed.pivot.clone().add(offset);
    const matrix = new Matrix4().lookAt(position, committed.pivot, new Vector3(0, 1, 0));
    const quaternion = new Quaternion().setFromRotationMatrix(matrix);

    return {
      position,
      quaternion,
      pivot: committed.pivot.clone(),
    };
  }

  return committed;
}

export function dollyRig(
  rig: SurfacePivotRig,
  deltaY: number,
  speed: number,
  minDistance: number,
  maxDistance: number,
): SurfacePivotRig {
  const forward = new Vector3(0, 0, -1).applyQuaternion(rig.quaternion).normalize();
  const amount = -deltaY * speed;
  const candidate = rig.position.clone().addScaledVector(forward, amount);
  const distance = candidate.distanceTo(rig.pivot);
  const clampedDistance = Math.min(Math.max(distance, minDistance), maxDistance);
  const position = rig.pivot.clone().sub(forward.multiplyScalar(clampedDistance));

  return {
    position,
    quaternion: rig.quaternion.clone(),
    pivot: rig.pivot.clone(),
  };
}

export function pickSurfacePoint(scene: Scene, camera: Camera, pointerNdc: Vector2): Intersection<Mesh> | null {
  const raycaster = new Raycaster();
  raycaster.setFromCamera(pointerNdc, camera);

  const meshes: Mesh[] = [];
  scene.traverseVisible(object => {
    if (object instanceof Mesh) {
      meshes.push(object);
    }
  });

  return raycaster.intersectObjects(meshes, false)[0] ?? null;
}
