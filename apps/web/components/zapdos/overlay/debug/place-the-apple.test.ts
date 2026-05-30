import assert from "node:assert/strict";
import test from "node:test";

type PlaceSelectedObjectModule = typeof import("./place-the-apple");

test("createPlaceSelectedObjectRequest posts the selected object release payload", async () => {
  const { createPlaceSelectedObjectRequest } = await loadModule<PlaceSelectedObjectModule>("./place-the-apple.ts");

  assert.deepEqual(createPlaceSelectedObjectRequest("Scene_Crate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([{ target_query: "Scene_Crate", arm: "left" }]),
  });
});

test("createPlaceSelectedObjectRequest rejects an empty selected object", async () => {
  const { createPlaceSelectedObjectRequest } = await loadModule<PlaceSelectedObjectModule>("./place-the-apple.ts");

  assert.throws(() => createPlaceSelectedObjectRequest("  "), /Select an object before placing/);
});

test("placeSelectedObject posts to the zapdos place_object route", async () => {
  const { createPlaceSelectedObjectRequest, placeSelectedObject } = await loadModule<PlaceSelectedObjectModule>("./place-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: done\ndata: {"ok":true,"arm":"left","target_body":"Scene_benchmark_building_blocks_006_01","scene_revision":"rev-4"}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const payload = await placeSelectedObject("sess-1", "Scene_benchmark_building_blocks_006_01");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/place_object");
    assert.deepEqual(calls[0]?.init, createPlaceSelectedObjectRequest("Scene_benchmark_building_blocks_006_01"));
    assert.equal(payload.target_body, "Scene_benchmark_building_blocks_006_01");
    assert.equal(payload.scene_revision, "rev-4");
  } finally {
    globalThis.fetch = originalFetch;
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
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}
