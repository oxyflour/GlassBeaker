import { streamJsonSse } from "../../../utils/sse";

export type ZapdosInitPhase = "loading" | "started" | "error";

export function buildZapdosInitTaskUrl(
  sess: string,
  sceneUsd: string | null,
  robotUsd: string | null
) {
  const query = new URLSearchParams();
  if (sceneUsd?.trim()) {
    query.set("scene_usd", sceneUsd.trim());
  }
  if (robotUsd?.trim()) {
    query.set("robot_usd", robotUsd.trim());
  }
  const suffix = query.toString();
  return suffix ? `/python/zapdos/${sess}/tasks/init?${suffix}` : `/python/zapdos/${sess}/tasks/init`;
}

export function buildZapdosSessionStorageKey(sceneUsd: string | null, robotUsd: string | null) {
  return ["zapdos-session", sceneUsd?.trim() || "", robotUsd?.trim() || ""].join("|");
}

export async function runZapdosInitTask(
  sess: string,
  sceneUsd: string | null,
  robotUsd: string | null,
  onProgress: (message: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  for await (const event of streamJsonSse<Record<string, unknown>>(buildZapdosInitTaskUrl(sess, sceneUsd, robotUsd), {
    signal,
  })) {
    if (event.event === "started") {
      onProgress("starting");
    } else if (event.event === "progress") {
      onProgress(String(event.data.message || "loading"));
    } else if (event.event === "failed") {
      throw new Error(String(event.data.detail || "Session bootstrap failed"));
    } else if (event.event === "done") {
      return;
    }
  }
  throw new Error("Session bootstrap stream ended unexpectedly");
}
