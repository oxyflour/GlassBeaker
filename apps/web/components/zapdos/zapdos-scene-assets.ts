import {
  BoxGeometry,
  BufferGeometry,
  CapsuleGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PlaneGeometry,
  SphereGeometry,
  Texture,
  TextureLoader,
} from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

import type { MeshVisual } from "./zapdos-scene-api";

const stlLoader = new STLLoader();
const objLoader = new OBJLoader();
const textureLoader = new TextureLoader();
const materials: Record<string, MeshStandardMaterial> = {};
const matrix = new Matrix4();

export async function loadSceneGeometry(item: MeshVisual) {
  const { size = [1, 1, 1] } = item;
  if (item.kind === "mesh") {
    if (item.mesh?.endsWith(".stl")) return await stlLoader.loadAsync(item.mesh);
    if (item.mesh?.endsWith(".obj")) {
      const obj = await objLoader.loadAsync(item.mesh);
      let geometry: BufferGeometry | null = null;
      obj.traverse(child => {
        if (child instanceof Mesh && child.geometry) geometry = child.geometry as BufferGeometry;
      });
      if (geometry) return geometry;
    }
    throw new Error(`unknown mesh type ${item.mesh}`);
  }
  if (item.kind === "box") return new BoxGeometry(size[0], size[1], size[2]);
  if (item.kind === "capsule") {
    const geometry = new CapsuleGeometry(size[0], size[1], 12, 24);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  if (item.kind === "cylinder") {
    const geometry = new CylinderGeometry(size[0], size[0], size[1], 24);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  }
  if (item.kind === "ellipsoid") {
    const geometry = new SphereGeometry(1, 24, 16);
    geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2);
    return geometry;
  }
  if (item.kind === "plane") return new PlaneGeometry(size[0], size[1]);
  if (item.kind === "sphere") return new SphereGeometry(size[0], 24, 16);
  throw new Error(`Unsupported geometry type for ${item.name}`);
}

export async function loadSceneTexture(texture?: string) {
  return texture ? await textureLoader.loadAsync(texture) : undefined;
}

export function getSceneMaterial(item: MeshVisual, image?: Texture) {
  const [r, g, b, a] = item.color ?? [1, 1, 1, 1];
  const isPlane = item.name.endsWith(".plane");
  const key = `${item.color?.join(",") ?? "default"}:${isPlane}:${item.texture}`;
  return materials[key] || (materials[key] = new MeshStandardMaterial({
    color: new Color(r, g, b),
    opacity: a,
    roughness: isPlane ? 1.0 : 0.2,
    metalness: isPlane ? 0.0 : 0.3,
    transparent: a < 1,
    ...(image ? { map: image } : {}),
    ...(isPlane ? { side: DoubleSide } : {}),
  }));
}

export function applyObjectMatrix(object: Object3D, elements: number[]) {
  matrix.fromArray(elements);
  matrix.decompose(object.position, object.quaternion, object.scale);
  object.updateMatrix();
  object.updateMatrixWorld(true);
}
