import type { SetSceneAssetsToolArgs } from "./zapdos-agent-tool-schemas";
import { publishZapdosSceneRevision } from "../scene/zapdos-runtime";
import { streamJsonSse } from "../../../utils/sse";

export type SetSceneAssetsInput = SetSceneAssetsToolArgs;
export type SetSceneAssetsResult = {
  ok: true;
  items: Array<{ asset_id: string; body: string; instance_id: string }>;
  scene_revision: string;
};
export type Bounds3 = { min: number[]; max: number[] };
export type ListSceneBodiesResult = {
  items: unknown[];
  robot_bounds: Bounds3 | null;
  scene_revision: string;
};
export type SceneTaskOptions = { signal?: AbortSignal };

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

export function createSceneTaskUrl(sess: string, task: string): string {
  return `/python/zapdos/${sess}/tasks/${task}`;
}

export async function listSceneBodies(sess: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/list_scene_bodies`, createSceneToolRequest([]));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as ListSceneBodiesResult;
}

export async function runSceneTask<T>(
  sess: string,
  task: string,
  request: RequestInit,
  options: SceneTaskOptions = {},
): Promise<T> {
  const requestInit = options.signal ? { ...request, signal: options.signal } : request;
  for await (const event of streamJsonSse<Record<string, unknown>>(createSceneTaskUrl(sess, task), requestInit)) {
    if (event.event === "failed") {
      throw new Error(String(event.data.detail || "Zapdos scene operation failed"));
    }
    if (event.event === "done") {
      const payload = event.data as T;
      publishZapdosSceneRevision(sess, payload, { force: true });
      return payload;
    }
  }
  throw new Error("Zapdos scene operation stream ended unexpectedly");
}

export async function setSceneAssets(
  sess: string,
  input: SetSceneAssetsInput,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<SetSceneAssetsResult>(
    sess,
    "set_scene_assets",
    createSetSceneAssetsRequest(input),
    options,
  );
}

export async function removeAssetFromScene(
  sess: string,
  instanceId: string,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<{ instance_id: string; scene_revision: string }>(
    sess,
    "remove_asset_from_scene",
    createSceneToolRequest([instanceId]),
    options,
  );
}
