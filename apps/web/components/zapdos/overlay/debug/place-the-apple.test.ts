import assert from "node:assert/strict";
import test from "node:test";

type PlaceTheAppleModule = typeof import("./place-the-apple");

test("createPlaceTheAppleRequest posts the canned apple release payload", async () => {
  const { createPlaceTheAppleRequest } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");

  assert.deepEqual(createPlaceTheAppleRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("placeTheApple posts to the zapdos place_apple route", async () => {
  const { createPlaceTheAppleRequest, placeTheApple } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      arm: "left",
      target_body: "Scene_apple_01",
      scene_revision: "rev-4",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await placeTheApple("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/place_apple");
    assert.deepEqual(calls[0]?.init, createPlaceTheAppleRequest());
    assert.equal(payload.target_body, "Scene_apple_01");
    assert.equal(payload.scene_revision, "rev-4");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
