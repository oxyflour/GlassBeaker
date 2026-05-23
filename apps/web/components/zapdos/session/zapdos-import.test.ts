import assert from "node:assert/strict";
import test from "node:test";

import {
  buildZapdosInitTaskUrl,
  buildZapdosSessionStorageKey,
  runZapdosInitTask,
} from "./zapdos-import";

test("buildZapdosInitTaskUrl includes encoded scene and robot params", () => {
  const url = buildZapdosInitTaskUrl("sess-1", "C:/tmp/a scene.usda", "deps/galaxea/object/r1pro/r1pro.usda");
  assert.equal(
    url,
    "/python/zapdos/sess-1/tasks/init?scene_usd=C%3A%2Ftmp%2Fa+scene.usda&robot_usd=deps%2Fgalaxea%2Fobject%2Fr1pro%2Fr1pro.usda"
  );
});

test("buildZapdosSessionStorageKey changes when scene changes", () => {
  assert.notEqual(
    buildZapdosSessionStorageKey("C:/tmp/scene-a.usda", null),
    buildZapdosSessionStorageKey("C:/tmp/scene-b.usda", null)
  );
});

test("buildZapdosSessionStorageKey keeps raw robot_usd strings distinct", () => {
  assert.notEqual(
    buildZapdosSessionStorageKey(null, "deps/galaxea/object/r1pro/r1pro.usda"),
    buildZapdosSessionStorageKey(null, "deps/spirit01_model/USD/Moz1_robot_only.usda")
  );
});

test("buildZapdosInitTaskUrl omits query when no import params are present", () => {
  assert.equal(buildZapdosInitTaskUrl("sess-1", null, null), "/python/zapdos/sess-1/tasks/init");
});

test("runZapdosInitTask reports progress and resolves on done", async () => {
  const originalFetch = globalThis.fetch;
  const messages: string[] = [];
  globalThis.fetch = (async () => new Response(streamFrom([
    'event: started\ndata: {"task":"init"}\n\n',
    'event: progress\ndata: {"message":"preparing render bundle"}\n\n',
    'event: done\ndata: {"ok":true}\n\n',
  ]), { status: 200 })) as typeof fetch;

  try {
    await runZapdosInitTask("sess-1", null, null, message => messages.push(message));
    assert.deepEqual(messages, ["starting", "preparing render bundle"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("runZapdosInitTask throws failed event details", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(streamFrom([
    'event: failed\ndata: {"detail":"scene_usd not found"}\n\n',
  ]), { status: 200 })) as typeof fetch;

  try {
    await assert.rejects(
      () => runZapdosInitTask("sess-1", null, null, () => {}),
      /scene_usd not found/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

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
