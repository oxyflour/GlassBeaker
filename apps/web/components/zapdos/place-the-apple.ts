import {
  createManipulationToolRequest,
} from "./zapdos-manipulation-tool-api";

export function createPlaceTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function placeTheApple(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/place_apple`,
    createPlaceTheAppleRequest(),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as {
    arm?: string;
    ok?: boolean;
    scene_revision: string;
    status?: string;
    target_body?: string;
  };
}
