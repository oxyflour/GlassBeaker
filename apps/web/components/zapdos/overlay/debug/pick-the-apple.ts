import {
  createManipulationToolRequest,
} from "../../agent/zapdos-manipulation-tool-api";
import {
  runSceneTask,
  type SceneTaskOptions,
} from "../../agent/zapdos-tool-api";

export function createPickTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function pickTheApple(
  sess: string,
  options: SceneTaskOptions = {},
) {
  return await runSceneTask<{
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  }>(sess, "pick_apple", createPickTheAppleRequest(), options);
}
