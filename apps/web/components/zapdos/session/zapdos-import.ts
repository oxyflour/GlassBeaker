export type ZapdosInitPhase = "loading" | "started" | "error";

export function buildZapdosInitStreamUrl(
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
  return suffix ? `/python/zapdos/${sess}/init/start?${suffix}` : `/python/zapdos/${sess}/init/start`;
}

export function buildZapdosSessionStorageKey(sceneUsd: string | null, robotUsd: string | null) {
  return ["zapdos-session", sceneUsd?.trim() || "", robotUsd?.trim() || ""].join("|");
}

export function parseZapdosInitEvent(data: string): { phase: ZapdosInitPhase; message: string } {
  if (data === "started") {
    return { phase: "started", message: "started" };
  }
  if (data.startsWith("error:")) {
    return {
      phase: "error",
      message: data.slice(6).trim() || "Session bootstrap failed",
    };
  }
  return { phase: "loading", message: data || "loading" };
}
