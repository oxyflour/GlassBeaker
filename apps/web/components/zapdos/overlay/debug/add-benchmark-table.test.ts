import assert from "node:assert/strict";
import test from "node:test";

type AddBenchmarkTableModule = typeof import("./add-benchmark-table");

test("createAddBenchmarkTableRequest posts the benchmark table shortcut payload", async () => {
  const { createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");

  assert.deepEqual(createAddBenchmarkTableRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([
      [{
        asset_id: "benchmark_table_000",
        motion: "static",
        placement: {
          kind: "floor_at_xy",
          xy: [0.5, 0],
          z_offset: 0,
          yaw: 0,
        },
      }, {
        asset_id: "apple",
        motion: "dynamic",
        placement: {
          kind: "on_top_of_body",
          body: "Scene_benchmark_table_000_01",
          xy: [0.34, 0.24],
          gap: 0,
          yaw: 0,
        },
      }],
    ]),
  });
});

test("addBenchmarkTable posts to the zapdos set_scene_assets route", async () => {
  const { addBenchmarkTable, createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      op_id: "op-1",
      status: "started",
      items: [{
        body: "Scene_benchmark_table_000_01",
        instance_id: "benchmark_table_000_01",
        asset_id: "benchmark_table_000",
      }, {
        body: "Scene_apple_01",
        instance_id: "apple_01",
        asset_id: "apple",
      }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await addBenchmarkTable("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/set_scene_assets");
    assert.deepEqual(calls[0]?.init, createAddBenchmarkTableRequest());
    assert.equal(payload.op_id, "op-1");
    assert.equal(payload.instance_id, "benchmark_table_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
