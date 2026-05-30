import assert from "node:assert/strict";
import test from "node:test";

type CameraOverrideSaveModule = typeof import("./camera-override-save");

test("createSaveCameraOverrideRequest posts an empty argument array", async () => {
  const { createSaveCameraOverrideRequest } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");

  assert.deepEqual(createSaveCameraOverrideRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("saveCameraOverride posts to the zapdos save route and returns a success message", async () => {
  const { createSaveCameraOverrideRequest, saveCameraOverride } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({ ok: true, saved: 3, path: "C:/Users/me/.glass-beaker/config.json" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const message = await saveCameraOverride("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/save_camera_override");
    assert.deepEqual(calls[0]?.init, createSaveCameraOverrideRequest());
    assert.equal(message, "Saved 3 camera overrides to C:/Users/me/.glass-beaker/config.json");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("saveCameraOverride throws backend text for a failed save", async () => {
  const { saveCameraOverride } = await loadModule<CameraOverrideSaveModule>("./camera-override-save.ts");
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response("Session expired", { status: 409 })) as typeof fetch;

  try {
    await assert.rejects(() => saveCameraOverride("sess-1"), /Session expired/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
