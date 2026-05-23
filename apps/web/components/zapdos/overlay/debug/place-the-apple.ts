import {
  createManipulationToolRequest,
} from "../../agent/zapdos-manipulation-tool-api";
import {
  runSceneTask,
  type SceneTaskOptions,
} from "../../agent/zapdos-tool-api";

export function createPlaceTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function placeTheApple(
  sess: string,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<{
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  }>(sess, "place_apple", createPlaceTheAppleRequest(), options);
}
