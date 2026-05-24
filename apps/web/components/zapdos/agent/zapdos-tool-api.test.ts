import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

type ZapdosToolApiModule = typeof import("./zapdos-tool-api");

test("createSetSceneAssetsRequest posts the full asset batch", async () => {
  const { createSetSceneAssetsRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");

  assert.deepEqual(
    createSetSceneAssetsRequest({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([[{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }]]),
    }
  );
});

test("createAddAssetsToSceneRequest posts the asset batch", async () => {
  const { createAddAssetsToSceneRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");

  assert.deepEqual(
    createAddAssetsToSceneRequest({
      assets: [{
        asset_id: "mug_000",
        motion: "dynamic",
        placement: { kind: "floor_at_xy", xy: [0, 0] },
      }],
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([[{
        asset_id: "mug_000",
        motion: "dynamic",
        placement: { kind: "floor_at_xy", xy: [0, 0] },
      }]]),
    }
  );
});

test("listPlacementBodies posts to the zapdos route", async () => {
  const { createSceneToolRequest, listPlacementBodies } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      items: [],
      robot_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      scene_revision: "rev-1",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await listPlacementBodies("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.deepEqual(payload.robot_bounds, { min: [0, 0, 0], max: [1, 1, 1] });
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_placement_bodies");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest([]));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("setSceneAssets posts to the task route and waits for the task stream", async () => {
  const { createSetSceneAssetsRequest, setSceneAssets } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: started\ndata: {"task":"set_scene_assets"}\n\n',
      'event: done\ndata: {"ok":true,"scene_revision":"rev-2","items":[{"asset_id":"table_000","body":"Scene_table_000_01","instance_id":"table_000_01"}]}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const payload = await setSceneAssets("sess-1", {
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    });

    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/set_scene_assets");
    assert.deepEqual(calls[0]?.init, createSetSceneAssetsRequest({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    }));
    assert.equal(payload.scene_revision, "rev-2");
    assert.equal(payload.items[0]?.instance_id, "table_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("placement body API does not export the old scene body names", async () => {
  const source = await readFile(new URL("./zapdos-tool-api.ts", import.meta.url), "utf8");

  assert.doesNotMatch(source, new RegExp("list" + "SceneBodies"));
  assert.doesNotMatch(source, new RegExp("List" + "SceneBodies"));
});

test("addAssetsToScene posts to the additive task route and waits for the task stream", async () => {
  const { addAssetsToScene, createAddAssetsToSceneRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: done\ndata: {"ok":true,"scene_revision":"rev-5","items":[{"asset_id":"mug_000","body":"Scene_mug_000_01","instance_id":"mug_000_01"}]}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const input = {
      assets: [{
        asset_id: "mug_000",
        motion: "dynamic" as const,
        placement: { kind: "floor_at_xy" as const, xy: [0, 0] as [number, number] },
      }],
    };
    const payload = await addAssetsToScene("sess-1", input);

    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/add_assets_to_scene");
    assert.deepEqual(calls[0]?.init, createAddAssetsToSceneRequest(input));
    assert.equal(payload.scene_revision, "rev-5");
    assert.equal(payload.items[0]?.instance_id, "mug_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("removeAssetsFromScene posts ids to the plural task route and waits for the task stream", async () => {
  const { createSceneToolRequest, removeAssetsFromScene } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: done\ndata: {"instance_ids":["table_000_01","mug_000_01"],"scene_revision":"rev-3"}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const payload = await removeAssetsFromScene("sess-1", ["table_000_01", "mug_000_01"]);

    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/remove_assets_from_scene");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest([["table_000_01", "mug_000_01"]]));
    assert.deepEqual(payload.instance_ids, ["table_000_01", "mug_000_01"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("scene task broadcasts completed scene revisions", async () => {
  const { setSceneAssets } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const { ZAPDOS_SCENE_REVISION_EVENT } = await import("../scene/zapdos-runtime");
  const eventTarget = new EventTarget();
  const revisions: unknown[] = [];
  const hadWindow = "window" in globalThis;
  const previousWindow = hadWindow ? globalThis.window : undefined;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(streamFrom([
    'event: done\ndata: {"ok":true,"scene_revision":"rev-4","items":[]}\n\n',
  ]), { status: 200 })) as typeof fetch;
  Object.defineProperty(globalThis, "window", { configurable: true, value: eventTarget });
  eventTarget.addEventListener(ZAPDOS_SCENE_REVISION_EVENT, (event) => {
    revisions.push((event as CustomEvent).detail);
  });

  try {
    await setSceneAssets("sess-1", { assets: [] as never });
    assert.deepEqual(revisions, [{ sess: "sess-1", scene_revision: "rev-4", force: true }]);
  } finally {
    globalThis.fetch = originalFetch;
    if (hadWindow) {
      Object.defineProperty(globalThis, "window", { configurable: true, value: previousWindow });
    } else {
      delete (globalThis as { window?: unknown }).window;
    }
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}

function streamFrom(chunks: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}
