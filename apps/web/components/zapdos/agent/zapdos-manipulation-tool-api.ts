import type { PickObjectToolArgs } from "./zapdos-manipulation-tool-schemas";
import {
  runSceneTask,
  type SceneTaskOptions,
} from "./zapdos-tool-api";

export type ListManipulationObjectsResult = {
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

export async function listManipulationObjects(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/list_manipulation_objects`,
    createManipulationToolRequest([])
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as ListManipulationObjectsResult;
}

export async function pickObject(
  sess: string,
  input: PickObjectToolArgs,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<PickObjectResult>(sess, "pick_object", createPickObjectRequest(input), options);
}
