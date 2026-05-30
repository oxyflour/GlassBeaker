import assert from "node:assert/strict";
import test from "node:test";
import { Group, Vector3 } from "three";

type ZapdosSceneApiModule = typeof import("./zapdos-scene-api");

test("createSetBodyPoseRequest posts body pose arguments", async () => {
  const { createSetBodyPoseRequest } = await loadModule<ZapdosSceneApiModule>("./zapdos-scene-api.ts");

  assert.deepEqual(createSetBodyPoseRequest("Scene_Crate", {
    pos: [1, 2, 3],
    quat: [1, 0, 0, 0],
  }), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(["Scene_Crate", [1, 2, 3], [1, 0, 0, 0]]),
  });
});

test("buildBodyPosePayload reads world transform from a nested object", async () => {
  const { buildBodyPosePayload } = await loadModule<ZapdosSceneApiModule>("./zapdos-scene-api.ts");
  const parent = new Group();
  parent.position.set(5, 0, 0);
  const child = new Group();
  child.position.set(0, 2, 0);
  child.quaternion.setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2);
  parent.add(child);
  parent.updateMatrixWorld(true);

  const pose = buildBodyPosePayload(child);

  assert.deepEqual(pose.pos, [5, 2, 0]);
  assert.deepEqual(pose.quat, [Math.cos(Math.PI / 4), 0, 0, Math.sin(Math.PI / 4)]);
});

test("getSceneVisual posts to the zapdos visual route", async () => {
  const { createGetVisualRequest, getSceneVisual } = await loadModule<ZapdosSceneApiModule>("./zapdos-scene-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({ bodies: [], meshes: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await getSceneVisual("sess-1");
    assert.deepEqual(payload, { bodies: [], meshes: [] });
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/get_visual");
    assert.deepEqual(calls[0]?.init, createGetVisualRequest());
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
