import { Camera, Intersection, Mesh, Quaternion, Raycaster, Scene, Vector2, Vector3 } from "three";

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
  dragThresholdPx?: number;
}

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
  const thresholdPx = config.dragThresholdPx ?? 5;
  if (!isDragGesture(gesture.start, point, thresholdPx)) {
    return createRigState(rig.position, rig.quaternion, rig.pivot);
  }

  if (gesture.button === 1) {
    const panSpeed = config.panSpeed ?? 1;
    const committed = createRigState(rig.position, rig.quaternion, rig.pivot);
    const dx = point.x - gesture.start.x;
    const dy = point.y - gesture.start.y;
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
    const committed = finishGesture(rig, gesture);
    const rotateSpeed = config.rotateSpeed ?? 1;
    const dx = point.x - gesture.start.x;
    const dy = point.y - gesture.start.y;
    const yaw = (-dx / viewport.width) * Math.PI * rotateSpeed;
    const pitch = (-dy / viewport.height) * Math.PI * rotateSpeed;

    const orientation = committed.quaternion.clone();
    const offset = committed.position.clone().sub(committed.pivot);

    const yawAxis = new Vector3(0, 1, 0).applyQuaternion(orientation).normalize();
    const yawQuat = new Quaternion().setFromAxisAngle(yawAxis, yaw);
    offset.applyQuaternion(yawQuat);
    orientation.premultiply(yawQuat);

    const pitchAxis = new Vector3(1, 0, 0).applyQuaternion(orientation).normalize();
    const pitchQuat = new Quaternion().setFromAxisAngle(pitchAxis, pitch);
    offset.applyQuaternion(pitchQuat);
    orientation.premultiply(pitchQuat);

    return {
      position: committed.pivot.clone().add(offset),
      quaternion: orientation,
      pivot: committed.pivot.clone(),
    };
  }

  return createRigState(rig.position, rig.quaternion, rig.pivot);
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
  const relative = rig.position.clone().sub(rig.pivot);
  const parallel = relative.dot(forward);
  const lateral = relative.clone().addScaledVector(forward, -parallel);
  const lateralLengthSq = lateral.lengthSq();
  const desiredParallel = parallel + amount;
  const desiredDistance = Math.sqrt(lateralLengthSq + desiredParallel * desiredParallel);
  const sign = Math.sign(desiredParallel) || Math.sign(parallel) || 1;
  let clampedParallel = desiredParallel;

  if (desiredDistance > maxDistance) {
    clampedParallel = sign * Math.sqrt(Math.max(0, maxDistance * maxDistance - lateralLengthSq));
  } else if (desiredDistance < minDistance) {
    clampedParallel = sign * Math.sqrt(Math.max(0, minDistance * minDistance - lateralLengthSq));
  }

  return {
    position: rig.pivot.clone().add(lateral).addScaledVector(forward, clampedParallel),
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
