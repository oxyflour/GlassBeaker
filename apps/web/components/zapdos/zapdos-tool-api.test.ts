import assert from "node:assert/strict";
import test from "node:test";

type ZapdosToolApiModule = typeof import("./zapdos-tool-api");

test("createAddAssetToSceneRequest posts asset id motion and placement", async () => {
  const { createAddAssetToSceneRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");

  assert.deepEqual(
    createAddAssetToSceneRequest({
      asset_id: "table_000",
      motion: "static",
      placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(["table_000", "static", { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 }]),
    }
  );
});

test("listSceneBodies posts to the zapdos route", async () => {
  const { createSceneToolRequest, listSceneBodies } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({ items: [], scene_revision: "rev-1" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await listSceneBodies("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_scene_bodies");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest([]));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
