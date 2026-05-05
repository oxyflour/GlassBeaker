export type ZapdosTransformMode = "translate" | "rotate";

export interface ZapdosSceneState {
  mode: ZapdosTransformMode;
  selectedBody: string | null;
}

export interface ZapdosPickHit {
  editable: boolean;
  body: string | null;
}

export interface ZapdosPointerPoint {
  x: number;
  y: number;
}

export function applySceneHotkey(state: ZapdosSceneState, key: string): ZapdosSceneState {
  if (key === "Escape") {
    return { ...state, selectedBody: null };
  }
  if (key.toLowerCase() === "w") {
    return { ...state, mode: "translate" };
  }
  if (key.toLowerCase() === "e") {
    return { ...state, mode: "rotate" };
  }
  return state;
}

export function shouldApplyBodyPose(body: string, draggingBody: string | null): boolean {
  return body !== draggingBody;
}

export function shouldReloadSceneRevision(current: string | null, next: string | null): boolean {
  return !!next && next !== current;
}

export function clearMissingSelection(selectedBody: string | null, nextBodies: Set<string>): string | null {
  if (!selectedBody) {
    return null;
  }
  return nextBodies.has(selectedBody) ? selectedBody : null;
}

export function pickEditableBodyFromHits(hits: ZapdosPickHit[]): string | null {
  for (const hit of hits) {
    if (hit.editable && hit.body) {
      return hit.body;
    }
  }
  return null;
}

export function isSelectionClick(
  start: ZapdosPointerPoint,
  end: ZapdosPointerPoint,
  thresholdPx = 5,
): boolean {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  return (dx * dx) + (dy * dy) <= thresholdPx * thresholdPx;
}
