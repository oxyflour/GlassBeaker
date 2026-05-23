import assert from "node:assert/strict";
import test from "node:test";

type ResetPoseModule = typeof import("./reset-pose");

test("createResetPoseRequest posts no arguments", async () => {
  const { createResetPoseRequest } = await loadModule<ResetPoseModule>("./reset-pose.ts");

  assert.deepEqual(createResetPoseRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("resetPose posts to the zapdos reset_pose task", async () => {
  const { createResetPoseRequest, resetPose } = await loadModule<ResetPoseModule>("./reset-pose.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: done\ndata: {"ok":true,"reset_bodies":["Scene_Crate"],"scene_revision":"rev-3"}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const payload = await resetPose("sess-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/reset_pose");
    assert.deepEqual(calls[0]?.init, createResetPoseRequest());
    assert.deepEqual(payload.reset_bodies, ["Scene_Crate"]);
    assert.equal(payload.scene_revision, "rev-3");
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
