import assert from "node:assert/strict";
import test from "node:test";
import { BufferGeometry, Texture } from "three";

import type { MeshVisual } from "./zapdos-scene-api";

type SceneAssetsModule = typeof import("./zapdos-scene-assets");

test("loadSceneMeshResources starts geometry and texture loads for every mesh without serial waits", async () => {
  const { loadSceneMeshResources } = await loadModule<SceneAssetsModule>("./zapdos-scene-assets.ts");
  const starts: string[] = [];
  const geometryA = deferred<BufferGeometry>();
  const geometryB = deferred<BufferGeometry>();
  const textureA = deferred<Texture | undefined>();
  const textureB = deferred<Texture | undefined>();
  const loadedGeometryA = new BufferGeometry();
  const loadedGeometryB = new BufferGeometry();
  const loadedTextureA = new Texture();
  const loadedTextureB = new Texture();
  const meshes: MeshVisual[] = [
    { name: "mesh-a", body: null, kind: "mesh", color: null, mesh: "a.obj", texture: "a.png" },
    { name: "mesh-b", body: null, kind: "mesh", color: null, mesh: "b.obj", texture: "b.png" },
  ];

  const pending = loadSceneMeshResources(meshes, {
    loadGeometry: async (item) => {
      starts.push(`geometry:${item.name}`);
      return await (item.name === "mesh-a" ? geometryA.promise : geometryB.promise);
    },
    loadTexture: async (url) => {
      starts.push(`texture:${url}`);
      return await (url === "a.png" ? textureA.promise : textureB.promise);
    },
  });

  await Promise.resolve();

  assert.deepEqual(starts, [
    "geometry:mesh-a",
    "texture:a.png",
    "geometry:mesh-b",
    "texture:b.png",
  ]);

  geometryA.resolve(loadedGeometryA);
  geometryB.resolve(loadedGeometryB);
  textureA.resolve(loadedTextureA);
  textureB.resolve(loadedTextureB);

  const loaded = await pending;

  assert.equal(loaded[0]?.item.name, "mesh-a");
  assert.equal(loaded[0]?.geometry, loadedGeometryA);
  assert.equal(loaded[0]?.image, loadedTextureA);
  assert.equal(loaded[1]?.item.name, "mesh-b");
  assert.equal(loaded[1]?.geometry, loadedGeometryB);
  assert.equal(loaded[1]?.image, loadedTextureB);
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
