import {
  createManipulationToolRequest,
} from "../../agent/zapdos-manipulation-tool-api";
import {
  type SceneOperationStreamFactory,
  type SceneToolOperationStart,
  waitForSceneToolOp,
} from "../../agent/zapdos-tool-api";

export function createPlaceTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function placeTheApple(
  sess: string,
  createEventSource: SceneOperationStreamFactory = (url) => new EventSource(url),
) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/place_apple`,
    createPlaceTheAppleRequest(),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const started = await response.json() as SceneToolOperationStart;
  return await waitForSceneToolOp<{
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  }>(sess, started.op_id, createEventSource);
}
