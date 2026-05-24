import {
  createManipulationToolRequest,
} from "../../agent/zapdos-manipulation-tool-api";
import {
  runSceneTask,
  type SceneTaskOptions,
} from "../../agent/zapdos-tool-api";

function buildPlaceSelectedObjectInput(selectedBody: string) {
  const target_query = selectedBody.trim();
  if (!target_query) {
    throw new Error("Select an object before placing");
  }
  return { target_query, arm: "left" as const };
}

export function createPlaceSelectedObjectRequest(selectedBody: string): RequestInit {
  return createManipulationToolRequest([buildPlaceSelectedObjectInput(selectedBody)]);
}

export async function placeSelectedObject(
  sess: string,
  selectedBody: string,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<{
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  }>(sess, "place_object", createPlaceSelectedObjectRequest(selectedBody), options);
}
