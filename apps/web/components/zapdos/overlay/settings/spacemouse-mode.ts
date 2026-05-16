export type SpaceMouseMode = "off" | "left" | "right";

export type SpaceMouseStatus = {
  running?: boolean;
  mode?: string;
  active_arm?: string;
};

export function deriveSpaceMouseMode(status: SpaceMouseStatus): SpaceMouseMode {
  if (status.mode === "off" || status.mode === "left" || status.mode === "right") {
    return status.mode;
  }
  if (status.running === false) {
    return "off";
  }
  return status.active_arm === "left" ? "left" : "right";
}

export function createSpaceMouseModeRequest(mode: SpaceMouseMode): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  };
}
