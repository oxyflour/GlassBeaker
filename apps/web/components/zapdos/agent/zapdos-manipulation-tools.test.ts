import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

import {
  listSceneObjectsToolArgsSchema,
  pickObjectToolArgsSchema,
} from "./zapdos-manipulation-tool-schemas";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";

type ZapdosManipulationToolApiModule = typeof import("./zapdos-manipulation-tool-api");

test("list_scene_objects schema only accepts an empty object", () => {
  assert.deepEqual(Object.keys(listSceneObjectsToolArgsSchema.shape), []);
  assert.equal(listSceneObjectsToolArgsSchema.safeParse({}).success, true);
  assert.equal(listSceneObjectsToolArgsSchema.safeParse({ extra: true }).success, false);
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

test("listSceneObjects posts to the zapdos route", async () => {
  const { createManipulationToolRequest, listSceneObjects } = await loadModule<ZapdosManipulationToolApiModule>("./zapdos-manipulation-tool-api.ts");
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
    const payload = await listSceneObjects("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_scene_objects");
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
    return new Response(JSON.stringify({
      ok: true,
      op_id: "op-3",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  const stream = new FakeEventSource("/python/zapdos/sess-1/op/op-3");

  try {
    const pending = pickObject("sess-1", {
      target_query: "the red mug",
      arm: "left",
    }, () => stream);
    await waitFor(() => stream.listenerCount("done") > 0);
    stream.dispatch("done", {
      ok: true,
      target_body: "Scene_mug_01",
      scene_revision: "rev-2",
    });
    const payload = await pending;
    assert.equal(payload.scene_revision, "rev-2");
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/pick_object");
    assert.deepEqual(calls[0]?.init, createPickObjectRequest({
      target_query: "the red mug",
      arm: "left",
    }));
    assert.equal(stream.url, "/python/zapdos/sess-1/op/op-3");
    assert.equal(stream.closed, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("useZapdosAgentTools registers the manipulation tools and instructions mention ambiguity handling", async () => {
  const source = await readFile(new URL("./useZapdosAgentTools.ts", import.meta.url), "utf8");

  assert.match(source, /name:\s*"list_scene_objects"/);
  assert.match(source, /name:\s*"pick_object"/);
  assert.match(source, /handler:\s*async\s*\(\)\s*=>\s*await listSceneObjects\(sess\)/);
  assert.match(source, /handler:\s*async\s*\(args\)\s*=>\s*await pickObject\(sess,\s*args\)/);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /list_scene_objects before pick_object/i);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /pick_object directly for simple pick commands/i);
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}

async function waitFor(predicate: () => boolean, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error("Timed out waiting for test condition");
    }
    await Promise.resolve();
  }
}

class FakeEventSource {
  closed = false;
  onerror: ((event: Event) => void) | null = null;
  private readonly listeners = new Map<string, Array<(event: MessageEvent) => void>>();

  constructor(readonly url: string) {}

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    const current = this.listeners.get(type) ?? [];
    current.push(listener);
    this.listeners.set(type, current);
  }

  close() {
    this.closed = true;
  }

  dispatch(type: string, payload: unknown) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(new MessageEvent(type, { data: JSON.stringify(payload) }));
    }
  }

  listenerCount(type: string) {
    return this.listeners.get(type)?.length ?? 0;
  }
}
