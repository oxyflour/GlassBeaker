import {
  createManipulationToolRequest,
} from "./zapdos-manipulation-tool-api";

export function createGrabTheAppleRequest(): RequestInit {
  return createManipulationToolRequest([]);
}

export async function grabTheApple(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/grab_apple`,
    createGrabTheAppleRequest(),
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
