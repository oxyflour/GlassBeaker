import type { SetSceneAssetsToolArgs } from "./zapdos-agent-tool-schemas";

export type SetSceneAssetsInput = SetSceneAssetsToolArgs;
export type SceneToolOperationStart = { ok: true; op_id: string };
export type SceneOperationStreamFactory = (url: string) => SceneOperationStream;

export interface SceneOperationStream {
  addEventListener(type: string, listener: (event: MessageEvent<string>) => void): void;
  close(): void;
  onerror: ((event: Event) => void) | null;
}

export function createSceneToolRequest(args: unknown[]): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  };
}

export function createSetSceneAssetsRequest(input: SetSceneAssetsInput): RequestInit {
  return createSceneToolRequest([input.assets]);
}

export function createSceneOpStreamUrl(sess: string, opId: string): string {
  return `/python/zapdos/${sess}/op/${opId}`;
}

export async function listSceneBodies(sess: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/list_scene_bodies`, createSceneToolRequest([]));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as { items: unknown[]; scene_revision: string };
}

function defaultSceneOperationStreamFactory(url: string): SceneOperationStream {
  return new EventSource(url);
}

function readSceneOperationError(event: MessageEvent<string>): Error {
  try {
    const payload = JSON.parse(event.data) as { detail?: string };
    return new Error(payload.detail || "Zapdos scene operation failed");
  } catch {
    return new Error("Zapdos scene operation failed");
  }
}

export async function waitForSceneToolOp<T>(
  sess: string,
  opId: string,
  createEventSource: SceneOperationStreamFactory = defaultSceneOperationStreamFactory,
): Promise<T> {
  return await new Promise<T>((resolve, reject) => {
    const source = createEventSource(createSceneOpStreamUrl(sess, opId));
    const close = () => source.close();
    source.addEventListener("done", (event) => {
      close();
      resolve(JSON.parse(event.data) as T);
    });
    source.addEventListener("failed", (event) => {
      close();
      reject(readSceneOperationError(event));
    });
    source.onerror = () => {
      close();
      reject(new Error("Zapdos scene operation stream disconnected"));
    };
  });
}

async function startSceneToolOperation<T>(
  sess: string,
  url: string,
  request: RequestInit,
  createEventSource: SceneOperationStreamFactory,
): Promise<T> {
  const response = await fetch(url, request);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const started = await response.json() as SceneToolOperationStart;
  return await waitForSceneToolOp<T>(sess, started.op_id, createEventSource);
}

export async function setSceneAssets(
  sess: string,
  input: SetSceneAssetsInput,
  createEventSource: SceneOperationStreamFactory = defaultSceneOperationStreamFactory,
) {
  return await startSceneToolOperation<{
    items: Array<{ asset_id: string; body: string; instance_id: string }>;
    scene_revision: string;
  }>(
    sess,
    `/python/zapdos/${sess}/call/set_scene_assets`,
    createSetSceneAssetsRequest(input),
    createEventSource,
  );
}

export async function removeAssetFromScene(
  sess: string,
  instanceId: string,
  createEventSource: SceneOperationStreamFactory = defaultSceneOperationStreamFactory,
) {
  return await startSceneToolOperation<{ instance_id: string; scene_revision: string }>(
    sess,
    `/python/zapdos/${sess}/call/remove_asset_from_scene`,
    createSceneToolRequest([instanceId]),
    createEventSource,
  );
}
