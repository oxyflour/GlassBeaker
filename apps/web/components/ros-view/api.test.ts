import assert from "node:assert/strict";
import test from "node:test";

test("loadRosViewState requests the ROS view backend state route", async () => {
  const mod = await loadModule<typeof import("./api")>("./api.ts");
  assert.ok(mod, "ros-view api module missing");

  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      connected: true,
      last_error: null,
      topics: [{
        id: "/env_0/head_camera/image_raw",
        kind: "image",
        label: "/env_0/head_camera/image_raw",
        description: "sensor_msgs/msg/Image",
        src: "/python/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw",
      }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const state = await mod.loadRosViewState();
    assert.equal(calls[0]?.input, "/python/ros_view/state");
    assert.equal(state.connected, true);
    assert.equal(state.topics[0]?.id, "/env_0/head_camera/image_raw");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("subscribeRosViewState listens to the SSE state stream", async () => {
  const mod = await loadModule<typeof import("./api")>("./api.ts");
  assert.ok(mod, "ros-view api module missing");

  const source = new FakeEventSource(mod.createRosViewStreamUrl());
  let payload: Awaited<ReturnType<typeof mod.loadRosViewState>> | null = null;

  const close = mod.subscribeRosViewState((next) => {
    payload = next;
  }, () => source);

  source.dispatch("state", {
    connected: true,
    last_error: null,
    topics: [{
      id: "/env_0/head_camera/image_raw",
      kind: "image",
      label: "/env_0/head_camera/image_raw",
      description: "sensor_msgs/msg/Image",
      src: "/python/ros_view/render/%2Fenv_0%2Fhead_camera%2Fimage_raw",
    }],
  });

  assert.equal(source.url, "/python/ros_view/stream");
  close();
  assert.equal(source.closed, true);
});

async function loadModule<TModule>(specifier: string): Promise<TModule | null> {
  try {
    const loaded = await import(specifier);
    return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
  } catch {
    return null;
  }
}

class FakeEventSource {
  closed = false;
  onerror: ((event: Event) => void) | null = null;
  private readonly listeners = new Map<string, Array<(event: MessageEvent<string>) => void>>();

  constructor(readonly url: string) {}

  addEventListener(type: string, listener: (event: MessageEvent<string>) => void) {
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
}
