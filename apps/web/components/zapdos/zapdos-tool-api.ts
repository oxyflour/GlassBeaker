import type { AddAssetToSceneToolArgs } from "./zapdos-agent-tool-schemas";

export type AddAssetToSceneInput = AddAssetToSceneToolArgs;

export function createSceneToolRequest(args: unknown[]): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  };
}

export function createAddAssetToSceneRequest(input: AddAssetToSceneInput): RequestInit {
  return createSceneToolRequest([input.asset_id, input.motion, input.placement]);
}

export async function listSceneBodies(sess: string) {
  const response = await fetch(`/python/zapdos/${sess}/call/list_scene_bodies`, createSceneToolRequest([]));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as { items: unknown[]; scene_revision: string };
}

export async function addAssetToScene(sess: string, input: AddAssetToSceneInput) {
  const response = await fetch(`/python/zapdos/${sess}/call/add_asset_to_scene`, createAddAssetToSceneRequest(input));
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as { body: string; instance_id: string; scene_revision: string };
}

export async function removeAssetFromScene(sess: string, instanceId: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/remove_asset_from_scene`,
    createSceneToolRequest([instanceId]),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as { instance_id: string; scene_revision: string };
}
