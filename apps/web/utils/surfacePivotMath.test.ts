import assert from "node:assert/strict";
import test from "node:test";
import { BoxGeometry, Mesh, MeshBasicMaterial, PerspectiveCamera, Quaternion, Scene, Vector2, Vector3 } from "three";

import {
  createRigState,
  dollyRig,
  finishGesture,
  isDragGesture,
  pickSurfacePoint,
  startGesture,
  updateGesture,
} from "./surfacePivotMath";

test("finishGesture commits a pending left-click pivot without moving the camera", () => {
  const rig = createRigState(new Vector3(1, 2, 3), new Quaternion(), new Vector3(0, 0, 0));
  const gesture = startGesture(0, new Vector2(10, 10), new Vector3(4, 5, 6));

  const next = finishGesture(rig, gesture);

  assert.deepEqual(next.position.toArray(), [1, 2, 3]);
  assert.deepEqual(next.quaternion.toArray(), [0, 0, 0, 1]);
  assert.deepEqual(next.pivot.toArray(), [4, 5, 6]);
});

test("isDragGesture stays false below threshold and true above it", () => {
  const start = new Vector2(10, 10);

  assert.equal(isDragGesture(start, new Vector2(13, 14), 5), false);
  assert.equal(isDragGesture(start, new Vector2(14, 14), 5), true);
});

test("updateGesture commits pending pivot before the first left-drag rotation", () => {
  const rig = createRigState(new Vector3(1, 0, 0), new Quaternion(), new Vector3(0, 0, 0));
  const gesture = startGesture(0, new Vector2(0, 0), new Vector3(0, 0, 0));
  const viewport = { width: 100, height: 100 };

  const next = updateGesture(rig, gesture, new Vector2(50, 0), viewport, 60, { rotateSpeed: 1 });

  assert.deepEqual(next.pivot.toArray(), [0, 0, 0]);
  assert.notDeepEqual(next.quaternion.toArray(), [0, 0, 0, 1]);
  assert.equal(next.position.length(), rig.position.length());
});

test("updateGesture does not commit a pending pivot below the drag threshold", () => {
  const rig = createRigState(new Vector3(1, 2, 3), new Quaternion(), new Vector3(0, 0, 0));
  const gesture = startGesture(0, new Vector2(0, 0), new Vector3(4, 5, 6));
  const viewport = { width: 100, height: 100 };

  const next = updateGesture(rig, gesture, new Vector2(3, 4), viewport, 60, { rotateSpeed: 1 });

  assert.deepEqual(next.pivot.toArray(), [0, 0, 0]);
  assert.deepEqual(next.position.toArray(), [1, 2, 3]);
  assert.deepEqual(next.quaternion.toArray(), [0, 0, 0, 1]);
});

test("updateGesture preserves roll instead of world-up locking during rotation", () => {
  const roll = new Quaternion().setFromAxisAngle(new Vector3(0, 0, -1), Math.PI / 2);
  const rig = createRigState(new Vector3(0, 0, 10), roll, new Vector3(0, 0, 0));
  const gesture = startGesture(0, new Vector2(0, 0), null);
  const viewport = { width: 100, height: 100 };

  const next = updateGesture(rig, gesture, new Vector2(40, 0), viewport, 60, { rotateSpeed: 1 });
  const up = new Vector3(0, 1, 0).applyQuaternion(next.quaternion);

  assert.ok(Math.abs(up.x) > 1e-6 || Math.abs(up.z) > 1e-6);
});

test("middle drag pans camera and pivot by the same world offset", () => {
  const rig = createRigState(new Vector3(1, 2, 3), new Quaternion(), new Vector3(4, 5, 6));
  const gesture = startGesture(1, new Vector2(0, 0), null);
  const viewport = { width: 100, height: 100 };

  const next = updateGesture(rig, gesture, new Vector2(10, 20), viewport, 60, { panSpeed: 1 });

  assert.deepEqual(next.position.toArray(), [0.9, 1.8, 3]);
  assert.deepEqual(next.pivot.toArray(), [3.9, 4.8, 6]);
  assert.deepEqual(next.quaternion.toArray(), [0, 0, 0, 1]);
});

test("dollyRig keeps the camera-to-pivot distance inside configured bounds", () => {
  const rig = createRigState(new Vector3(0, 0, 10), new Quaternion(), new Vector3(0, 0, 0));

  const closer = dollyRig(rig, 1000, 0.01, 2, 8);
  assert.ok(Math.abs(closer.position.distanceTo(closer.pivot) - 8) < 1e-9);

  const farther = dollyRig(closer, -1000, 0.01, 2, 8);
  assert.ok(Math.abs(farther.position.distanceTo(farther.pivot) - 2) < 1e-9);
});

test("dollyRig preserves lateral offset after a pivot-only commit", () => {
  const rig = createRigState(new Vector3(2, 0, 10), new Quaternion(), new Vector3(1, 0, 0));

  const next = dollyRig(rig, 3, 1, 2, 20);

  assert.equal(next.position.x, 2);
  assert.equal(next.pivot.x, 1);
  assert.ok(next.position.distanceTo(next.pivot) <= 20);
});

test("pickSurfacePoint ignores hidden meshes and returns the first visible mesh hit", () => {
  const scene = new Scene();
  const camera = new PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -1);
  camera.updateMatrixWorld(true);

  const hidden = new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial());
  hidden.visible = false;
  hidden.position.set(0, 0, -2);
  hidden.updateMatrixWorld(true);

  const visible = new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial());
  visible.position.set(0, 0, -4);
  visible.updateMatrixWorld(true);

  scene.add(hidden, visible);
  scene.updateMatrixWorld(true);

  const hit = pickSurfacePoint(scene, camera, new Vector2(0, 0));

  assert.ok(hit);
  assert.equal(hit.object, visible);
});
