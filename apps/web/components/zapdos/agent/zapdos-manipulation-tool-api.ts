import type { PickObjectToolArgs } from "./zapdos-manipulation-tool-schemas";
import {
  type SceneOperationStreamFactory,
  type SceneToolOperationStart,
  waitForSceneToolOp,
} from "./zapdos-tool-api";

export type ListSceneObjectsResult = {
  items: unknown[];
  scene_revision: string;
};

export type PickObjectResult = {
  arm?: string;
  ok?: boolean;
  scene_revision: string;
  status?: string;
  target_body?: string;
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

export async function pickObject(
  sess: string,
  input: PickObjectToolArgs,
  createEventSource: SceneOperationStreamFactory = (url) => new EventSource(url),
) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/pick_object`,
    createPickObjectRequest(input)
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const started = await response.json() as SceneToolOperationStart;
  return await waitForSceneToolOp<PickObjectResult>(sess, started.op_id, createEventSource);
}
