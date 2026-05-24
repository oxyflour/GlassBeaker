import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import {
  listManipulationObjectsToolArgsSchema,
  pickObjectToolArgsSchema,
} from "./zapdos-manipulation-tool-schemas";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";

type ZapdosManipulationToolApiModule = typeof import("./zapdos-manipulation-tool-api");

test("list_manipulation_objects schema only accepts an empty object", () => {
  assert.deepEqual(Object.keys(listManipulationObjectsToolArgsSchema.shape), []);
  assert.equal(listManipulationObjectsToolArgsSchema.safeParse({}).success, true);
  assert.equal(listManipulationObjectsToolArgsSchema.safeParse({ extra: true }).success, false);
});

test("pick_object schema accepts the target query, optional support query, and arm", () => {
  assert.deepEqual(
    pickObjectToolArgsSchema.parse({
      target_query: "the red mug",
      support_query: "the left table",
      arm: "left",
    }),
    {
      target_query: "the red mug",
      support_query: "the left table",
      arm: "left",
    }
  );
  assert.deepEqual(
    pickObjectToolArgsSchema.parse({ target_query: "mug" }),
    { target_query: "mug", arm: "left" }
  );
});

test("createPickObjectRequest wraps the single JSON object arg in an args array", async () => {
  const { createPickObjectRequest } = await loadModule<ZapdosManipulationToolApiModule>("./zapdos-manipulation-tool-api.ts");

  assert.deepEqual(
    createPickObjectRequest({
      target_query: "the red mug",
      support_query: "the left table",
      arm: "right",
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([{
        target_query: "the red mug",
        support_query: "the left table",
        arm: "right",
      }]),
    }
  );
});

test("listManipulationObjects posts to the zapdos route", async () => {
  const { createManipulationToolRequest, listManipulationObjects } = await loadModule<ZapdosManipulationToolApiModule>("./zapdos-manipulation-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      items: [{ body: "cup_01", label: "red mug" }],
      scene_revision: "rev-1",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await listManipulationObjects("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_manipulation_objects");
    assert.deepEqual(calls[0]?.init, createManipulationToolRequest([]));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("pickObject posts to the zapdos route with the wrapped object arg", async () => {
  const { createPickObjectRequest, pickObject } = await loadModule<ZapdosManipulationToolApiModule>("./zapdos-manipulation-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(streamFrom([
      'event: done\ndata: {"ok":true,"target_body":"Scene_mug_01","scene_revision":"rev-2"}\n\n',
    ]), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  }) as typeof fetch;

  try {
    const payload = await pickObject("sess-1", {
      target_query: "the red mug",
      arm: "left",
    });
    assert.equal(payload.scene_revision, "rev-2");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/tasks/pick_object");
    assert.deepEqual(calls[0]?.init, createPickObjectRequest({
      target_query: "the red mug",
      arm: "left",
    }));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("useZapdosAgentTools registers the manipulation tools and instructions mention ambiguity handling", async () => {
  const source = await readFile(new URL("./useZapdosAgentTools.ts", import.meta.url), "utf8");

  assert.match(source, /name:\s*"list_manipulation_objects"/);
  assert.match(source, /name:\s*"pick_object"/);
  assert.match(source, /handler:\s*async\s*\(\)\s*=>\s*await listManipulationObjects\(sess\)/);
  assert.match(source, /handler:\s*async\s*\(args\)\s*=>\s*await pickObject\(sess,\s*args\)/);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /list_manipulation_objects before pick_object/i);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /pick_object directly for simple pick commands/i);
});

test("manipulation API and schema do not export the old scene object names", async () => {
  const apiSource = await readFile(new URL("./zapdos-manipulation-tool-api.ts", import.meta.url), "utf8");
  const schemaSource = await readFile(new URL("./zapdos-manipulation-tool-schemas.ts", import.meta.url), "utf8");

  assert.doesNotMatch(apiSource, new RegExp("list" + "SceneObjects"));
  assert.doesNotMatch(apiSource, new RegExp("List" + "SceneObjects"));
  assert.doesNotMatch(schemaSource, new RegExp("list" + "SceneObjects"));
});

test("useZapdosAgentTools registers additive scene tools and plural removal", async () => {
  const source = await readFile(new URL("./useZapdosAgentTools.ts", import.meta.url), "utf8");

  assert.match(source, /name:\s*"set_scene_assets"/);
  assert.match(source, /name:\s*"add_assets_to_scene"/);
  assert.match(source, /name:\s*"remove_assets_from_scene"/);
  assert.doesNotMatch(source, /name:\s*"remove_asset_from_scene"/);
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
