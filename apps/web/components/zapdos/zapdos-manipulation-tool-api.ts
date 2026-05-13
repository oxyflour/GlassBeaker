import type { PickObjectToolArgs } from "./zapdos-manipulation-tool-schemas";

export type ListSceneObjectsResult = {
  items: unknown[];
  scene_revision: string;
};

export type PickObjectResult = {
  scene_revision: string;
  status?: string;
};

export function createManipulationToolRequest(args: unknown[]): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  };
}

export function createPickObjectRequest(input: PickObjectToolArgs): RequestInit {
  return createManipulationToolRequest([input]);
}

export async function listSceneObjects(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/list_scene_objects`,
    createManipulationToolRequest([])
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as ListSceneObjectsResult;
}

export async function pickObject(sess: string, input: PickObjectToolArgs) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/pick_object`,
    createPickObjectRequest(input)
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as PickObjectResult;
}
