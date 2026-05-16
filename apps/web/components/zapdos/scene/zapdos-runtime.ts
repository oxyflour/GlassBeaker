export const ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE = "Session disconnected. Refresh to reload scene.";

const SESSION_ERROR_DETAILS = new Set(["Session expired", "Session not initialized"]);

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return typeof error === "string" ? error : "";
}

function getErrorDetail(message: string) {
  if (!message) {
    return "";
  }
  try {
    const payload = JSON.parse(message) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : "";
  } catch {
    return "";
  }
}

export function getZapdosRuntimeErrorMessage(error: unknown) {
  const message = getErrorMessage(error);
  return SESSION_ERROR_DETAILS.has(getErrorDetail(message))
    ? ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE
    : (message || "Session runtime failed");
}

export function isZapdosInactivePayload(payload: unknown) {
  return !!payload
    && typeof payload === "object"
    && "inactive" in payload
    && (payload as { inactive?: unknown }).inactive === true;
}

export function getZapdosSceneRevision(payload: unknown) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const revision = (payload as { scene_revision?: unknown }).scene_revision;
  return typeof revision === "string" && revision.trim() ? revision : null;
}
