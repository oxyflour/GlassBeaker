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
const meshes: Record<string, Promise<BufferGeometry>> = { }

async function loadObj(url: string) {
  const obj = await objLoader.loadAsync(url);
  let geometry: BufferGeometry | null = null;
  obj.traverse(child => {
    if (child instanceof Mesh && child.geometry) {
      geometry = child.geometry as BufferGeometry;
    }
  });
  if (geometry) {
    return geometry;
  } else {
    throw Error(`${url} has no mesh`)
  }
}

export async function loadSceneGeometry(item: MeshVisual) {
  const { size = [1, 1, 1] } = item;
  if (item.kind === "mesh") {
    if (item.mesh?.endsWith(".stl")) {
      return await (meshes[item.mesh] || (meshes[item.mesh] = stlLoader.loadAsync(item.mesh)));
    } else if (item.mesh?.endsWith(".obj")) {
      return await (meshes[item.mesh] || (meshes[item.mesh] = loadObj(item.mesh)))
    } else {
      throw new Error(`unknown mesh type ${item.mesh}`);
    }
  } else if (item.kind === "box") {
    return new BoxGeometry(size[0], size[1], size[2]);
  } else if (item.kind === "capsule") {
    const geometry = new CapsuleGeometry(size[0], size[1], 12, 24);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  } else if (item.kind === "cylinder") {
    const geometry = new CylinderGeometry(size[0], size[0], size[1], 24);
    geometry.rotateX(Math.PI / 2);
    return geometry;
  } else if (item.kind === "ellipsoid") {
    const geometry = new SphereGeometry(1, 24, 16);
    geometry.scale(size[0] / 2, size[1] / 2, size[2] / 2);
    return geometry;
  } else if (item.kind === "plane") {
    return new PlaneGeometry(size[0], size[1]);
  } else if (item.kind === "sphere") {
    return new SphereGeometry(size[0], 24, 16);
  }
  throw new Error(`Unsupported geometry type for ${item.name}`);
}

const textureLoader = new TextureLoader();
const textures: Record<string, Promise<Texture>> = { };
export function loadSceneTexture(url = ""): Promise<Texture | undefined> {
  if (!url) {
    return Promise.resolve(undefined);
  }
  return textures[url] || (textures[url] = textureLoader.loadAsync(url));
}

const materials: Record<string, MeshStandardMaterial> = {};
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

const matrix = new Matrix4();
export function applyObjectMatrix(object: Object3D, elements: number[]) {
  matrix.fromArray(elements);
  matrix.decompose(object.position, object.quaternion, object.scale);
  object.updateMatrix();
  object.updateMatrixWorld(true);
}
