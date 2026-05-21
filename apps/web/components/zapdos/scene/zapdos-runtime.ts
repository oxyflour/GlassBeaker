export const ZAPDOS_RUNTIME_DISCONNECTED_MESSAGE = "Session disconnected. Refresh to reload scene.";
export const ZAPDOS_SCENE_REVISION_EVENT = "zapdos:scene-revision";

const SESSION_ERROR_DETAILS = new Set(["Session expired", "Session not initialized"]);

export type ZapdosSceneRevisionEventDetail = {
  sess: string;
  scene_revision: string;
};

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

export function createZapdosSceneRevisionEvent(sess: string, sceneRevision: string) {
  return new CustomEvent<ZapdosSceneRevisionEventDetail>(ZAPDOS_SCENE_REVISION_EVENT, {
    detail: {
      sess,
      scene_revision: sceneRevision,
    },
  });
}

export function getZapdosSceneRevisionEventDetail(event: Event, sess: string) {
  if (event.type !== ZAPDOS_SCENE_REVISION_EVENT || !("detail" in event)) {
    return null;
  }
  const detail = (event as CustomEvent<Partial<ZapdosSceneRevisionEventDetail>>).detail;
  if (!detail || detail.sess !== sess || typeof detail.scene_revision !== "string") {
    return null;
  }
  return {
    sess: detail.sess,
    scene_revision: detail.scene_revision,
  };
}

export function publishZapdosSceneRevision(sess: string, payload: unknown) {
  const revision = getZapdosSceneRevision(payload);
  if (!revision || typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(createZapdosSceneRevisionEvent(sess, revision));
}
