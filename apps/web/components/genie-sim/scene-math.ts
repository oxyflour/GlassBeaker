import { Matrix4, Quaternion } from "three";

const BASIS = new Matrix4().set(
  0, -1, 0, 0,
  0, 0, 1, 0,
  -1, 0, 0, 0,
  0, 0, 0, 1,
);

const BASIS_INV = BASIS.clone().invert();

export function simPointToThree([x, y, z]: [number, number, number]) {
  return [-y, z, -x] as [number, number, number];
}

export function simQuaternionToThree([x, y, z, w]: [number, number, number, number]) {
  const simRotation = new Matrix4().makeRotationFromQuaternion(new Quaternion(x, y, z, w));
  const threeRotation = BASIS.clone().multiply(simRotation).multiply(BASIS_INV);
  const quaternion = new Quaternion().setFromRotationMatrix(threeRotation);
  return [quaternion.x, quaternion.y, quaternion.z, quaternion.w] as [number, number, number, number];
}
