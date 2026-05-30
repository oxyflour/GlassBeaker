import type { AddAssetsToSceneToolArgs, SetSceneAssetsToolArgs } from "./zapdos-agent-tool-schemas";
import { publishZapdosSceneRevision } from "../scene/zapdos-runtime";
import { streamJsonSse } from "../../../utils/sse";

export type SetSceneAssetsInput = SetSceneAssetsToolArgs;
export type AddAssetsToSceneInput = AddAssetsToSceneToolArgs;
export type SceneAssetsMutationResult = {
  ok: true;
  items: Array<{ asset_id: string; body: string; instance_id: string }>;
  scene_revision: string;
};
export type SetSceneAssetsResult = SceneAssetsMutationResult;
export type AddAssetsToSceneResult = SceneAssetsMutationResult;
export type Bounds3 = { min: number[]; max: number[] };
export type ListPlacementBodiesResult = {
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

export function createAddAssetsToSceneRequest(input: AddAssetsToSceneInput): RequestInit {
  return createSceneToolRequest([input.assets]);
}

export function createSceneTaskUrl(sess: string, task: string): string {
  return `/python/zapdos/${sess}/tasks/${task}`;
}

export async function listPlacementBodies(sess: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/list_placement_bodies`, createSceneToolRequest([]));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as ListPlacementBodiesResult;
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

export async function addAssetsToScene(
  sess: string,
  input: AddAssetsToSceneInput,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<AddAssetsToSceneResult>(
    sess,
    "add_assets_to_scene",
    createAddAssetsToSceneRequest(input),
    options,
  );
}

export async function removeAssetsFromScene(
  sess: string,
  instanceIds: string[],
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<{ instance_ids: string[]; scene_revision: string }>(
    sess,
    "remove_assets_from_scene",
    createSceneToolRequest([instanceIds]),
    options,
  );
}
