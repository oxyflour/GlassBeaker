import assert from "node:assert/strict";
import test from "node:test";

type GrabTheAppleModule = typeof import("./grab-the-apple");

test("createGrabTheAppleRequest posts the canned apple pick payload", async () => {
  const { createGrabTheAppleRequest } = await loadModule<GrabTheAppleModule>("./grab-the-apple.ts");

  assert.deepEqual(createGrabTheAppleRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("grabTheApple posts to the zapdos grab_apple route", async () => {
  const { createGrabTheAppleRequest, grabTheApple } = await loadModule<GrabTheAppleModule>("./grab-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      arm: "left",
      target_body: "Scene_apple_01",
      scene_revision: "rev-3",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await grabTheApple("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/grab_apple");
    assert.deepEqual(calls[0]?.init, createGrabTheAppleRequest());
    assert.equal(payload.target_body, "Scene_apple_01");
    assert.equal(payload.scene_revision, "rev-3");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
