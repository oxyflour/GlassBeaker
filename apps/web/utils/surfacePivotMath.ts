import { Camera, Intersection, Mesh, Quaternion, Raycaster, Scene, Vector2, Vector3 } from "three";

export interface SurfacePivotRig {
  position: Vector3;
  quaternion: Quaternion;
  pivot: Vector3;
}

export interface SurfacePivotGesture {
  button: number;
  start: Vector2;
  last: Vector2;
  dragging: boolean;
  pendingPivot: Vector3 | null;
}

export interface SurfacePivotUpdate {
  rig: SurfacePivotRig;
  gesture: SurfacePivotGesture;
  changed: boolean;
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
    last: start.clone(),
    dragging: false,
    pendingPivot: pendingPivot?.clone() ?? null,
  };
}

export function finishGesture(rig: SurfacePivotRig, gesture: SurfacePivotGesture): SurfacePivotRig {
  if (gesture.button !== 0 || !gesture.pendingPivot) {
    return createRigState(rig.position, rig.quaternion, rig.pivot);
  }

  return createRigState(rig.position, rig.quaternion, gesture.pendingPivot);
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
): SurfacePivotUpdate {
  const thresholdPx = config.dragThresholdPx ?? 5;
  const crossedThreshold = !gesture.dragging && isDragGesture(gesture.start, point, thresholdPx);
  const active = gesture.dragging || crossedThreshold;
  const delta = active ? point.clone().sub(gesture.dragging ? gesture.last : gesture.start) : new Vector2();

  if (!active) {
    return {
      rig: createRigState(rig.position, rig.quaternion, rig.pivot),
      gesture: {
        ...gesture,
      },
      changed: false,
    };
  }

  const currentRig = gesture.button === 0 && crossedThreshold ? finishGesture(rig, gesture) : createRigState(rig.position, rig.quaternion, rig.pivot);
  const changed = delta.lengthSq() > 0;
  const nextGesture: SurfacePivotGesture = {
    ...gesture,
    last: point.clone(),
    dragging: true,
  };

  if (gesture.button === 1) {
    const panSpeed = config.panSpeed ?? 1;
    const offset = new Vector3()
      .addScaledVector(new Vector3(1, 0, 0).applyQuaternion(currentRig.quaternion), (-delta.x / viewport.width) * panSpeed)
      .addScaledVector(new Vector3(0, 1, 0).applyQuaternion(currentRig.quaternion), (-delta.y / viewport.height) * panSpeed);

    return {
      rig: {
        position: currentRig.position.clone().add(offset),
        quaternion: currentRig.quaternion.clone(),
        pivot: currentRig.pivot.clone().add(offset),
      },
      gesture: nextGesture,
      changed,
    };
  }

  if (gesture.button === 0) {
    const rotateSpeed = config.rotateSpeed ?? 1;
    const yaw = (-delta.x / viewport.width) * Math.PI * rotateSpeed;
    const pitch = (-delta.y / viewport.height) * Math.PI * rotateSpeed;

    const orientation = currentRig.quaternion.clone();
    const offset = currentRig.position.clone().sub(currentRig.pivot);

    const yawAxis = new Vector3(0, 1, 0).applyQuaternion(orientation).normalize();
    const yawQuat = new Quaternion().setFromAxisAngle(yawAxis, yaw);
    orientation.premultiply(yawQuat);
    offset.applyQuaternion(yawQuat);

    const pitchAxis = new Vector3(1, 0, 0).applyQuaternion(orientation).normalize();
    const pitchQuat = new Quaternion().setFromAxisAngle(pitchAxis, pitch);
    orientation.premultiply(pitchQuat);
    offset.applyQuaternion(pitchQuat);

    return {
      rig: {
        position: currentRig.pivot.clone().add(offset),
        quaternion: orientation,
        pivot: currentRig.pivot.clone(),
      },
      gesture: {
        ...nextGesture,
        pendingPivot: null,
      },
      changed,
    };
  }

  return {
    rig: createRigState(rig.position, rig.quaternion, rig.pivot),
    gesture: nextGesture,
    changed: false,
  };
}

export function dollyRig(
  rig: SurfacePivotRig,
  deltaY: number,
  speed: number,
  minDistance: number,
  maxDistance: number,
): SurfacePivotRig {
  const forward = new Vector3(0, 0, -1).applyQuaternion(rig.quaternion).normalize();
  const relative = rig.position.clone().sub(rig.pivot);
  const parallel = relative.dot(forward);
  const lateral = relative.clone().addScaledVector(forward, -parallel);
  const lateralLength = lateral.length();
  if (lateralLength > maxDistance) {
    return createRigState(rig.position, rig.quaternion, rig.pivot);
  }

  const amount = -deltaY * speed;
  const targetParallel = parallel + amount;
  const sign = Math.sign(parallel) || Math.sign(targetParallel) || Math.sign(amount) || 1;
  const minMagnitude = lateralLength >= minDistance ? 0 : Math.sqrt(Math.max(0, minDistance * minDistance - lateralLength * lateralLength));
  const maxMagnitude = Math.sqrt(Math.max(0, maxDistance * maxDistance - lateralLength * lateralLength));
  const signedMin = sign * minMagnitude;
  const signedMax = sign * maxMagnitude;
  const low = Math.min(signedMin, signedMax);
  const high = Math.max(signedMin, signedMax);
  const clampedParallel = Math.min(high, Math.max(low, targetParallel));
  const parallelOffset = forward.multiplyScalar(clampedParallel);
  return {
    position: rig.pivot.clone().add(lateral).add(parallelOffset),
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
