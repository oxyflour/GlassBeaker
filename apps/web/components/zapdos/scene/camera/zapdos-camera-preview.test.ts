import assert from "node:assert/strict";
import test from "node:test";

type ZapdosCameraPreviewModule = typeof import("./zapdos-camera-preview");

test("loadCameraNames keeps backend order and limits previews to three cameras", async () => {
  const { loadCameraNames } = await loadModule<ZapdosCameraPreviewModule>("./zapdos-camera-preview.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      head_camera: [0, 0, 0],
      left_wrist_camera: [0, 0, 0],
      right_wrist_camera: [0, 0, 0],
      rear_camera: [0, 0, 0],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const cameras = await loadCameraNames("sess-1");
    assert.deepEqual(cameras, [
      "head_camera",
      "left_wrist_camera",
      "right_wrist_camera",
    ]);
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/get_camera");
    assert.deepEqual(calls[0]?.init, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([]),
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("buildCompositeCameraSlices splits the stitched frame into equal camera segments", async () => {
  const { buildCompositeCameraSlices } = await loadModule<ZapdosCameraPreviewModule>("./zapdos-camera-preview.ts");

  assert.deepEqual(
    buildCompositeCameraSlices(["head_camera", "left_wrist_camera", "right_wrist_camera"], 960, 240),
    [
      { camera: "head_camera", left: 0, width: 320, height: 240 },
      { camera: "left_wrist_camera", left: 320, width: 320, height: 240 },
      { camera: "right_wrist_camera", left: 640, width: 320, height: 240 },
    ]
  );
  assert.deepEqual(buildCompositeCameraSlices([], 960, 240), []);
});

test("buildCompositeCameraStreamUrl points at the stitched MJPEG stream", async () => {
  const {
    buildCompositeCameraStreamUrl,
  } = await loadModule<ZapdosCameraPreviewModule>("./zapdos-camera-preview.ts");

  assert.equal(buildCompositeCameraStreamUrl("sess-1"), "/python/zapdos/sess-1/multicam/stream");
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}
