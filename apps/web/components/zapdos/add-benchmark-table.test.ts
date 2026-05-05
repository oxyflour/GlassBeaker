import assert from "node:assert/strict";
import test from "node:test";

type AddBenchmarkTableModule = typeof import("./add-benchmark-table");

test("createAddBenchmarkTableRequest posts the benchmark table shortcut payload", async () => {
  const { createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");

  assert.deepEqual(createAddBenchmarkTableRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([
      "benchmark_table_000",
      "static",
      {
        kind: "floor_at_xy",
        xy: [0, 0],
        z_offset: 0,
        yaw: 0,
      },
    ]),
  });
});

test("addBenchmarkTable posts to the zapdos add_asset_to_scene route", async () => {
  const { addBenchmarkTable, createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      body: "Scene_benchmark_table_000_01",
      instance_id: "benchmark_table_000_01",
      scene_revision: "rev-2",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await addBenchmarkTable("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/add_asset_to_scene");
    assert.deepEqual(calls[0]?.init, createAddBenchmarkTableRequest());
    assert.equal(payload.instance_id, "benchmark_table_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
