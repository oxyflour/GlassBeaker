import {
  createPickObjectRequest,
  pickObject,
} from "../../agent/zapdos-manipulation-tool-api";
import {
  type SceneTaskOptions,
} from "../../agent/zapdos-tool-api";

function buildPickSelectedObjectInput(selectedBody: string) {
  const target_query = selectedBody.trim();
  if (!target_query) {
    throw new Error("Select an object before picking");
  }
  return { target_query, arm: "left" as const };
}

export function createPickSelectedObjectRequest(selectedBody: string): RequestInit {
  return createPickObjectRequest(buildPickSelectedObjectInput(selectedBody));
}

export async function pickSelectedObject(
  sess: string,
  selectedBody: string,
  options: SceneTaskOptions = {},
) {
  return await pickObject(sess, buildPickSelectedObjectInput(selectedBody), options);
}
