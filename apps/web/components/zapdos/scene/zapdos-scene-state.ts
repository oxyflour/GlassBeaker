import { Matrix4 } from "three";

export type ZapdosTransformMode = "translate" | "rotate";

export interface ZapdosSceneState {
  mode: ZapdosTransformMode;
  selectedBody: string | null;
}

export interface ZapdosPickHit {
  editable: boolean;
  body: string | null;
  selectionBody: string | null;
}

export interface ZapdosPointerPoint {
  x: number;
  y: number;
}

export interface ZapdosBodyState {
  movable: boolean;
  selectionBody: string | null;
  matrix: number[];
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

export function shouldApplyBodyPose(
  body: string,
  draggingSelectionBody: string | null,
  bodies: Record<string, ZapdosBodyState>
): boolean {
  if (!draggingSelectionBody) {
    return true;
  }
  return bodies[body]?.selectionBody !== draggingSelectionBody;
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

export function pickSelectableBodyFromHits(hits: ZapdosPickHit[]): string | null {
  for (const hit of hits) {
    if (hit.selectionBody) {
      return hit.selectionBody;
    }
    if (hit.body) {
      return hit.body;
    }
  }
  return null;
}

export function getTransformBodyName(
  selectedBody: string | null,
  bodies: Record<string, ZapdosBodyState>
): string | null {
  if (!selectedBody) {
    return null;
  }
  return bodies[selectedBody]?.movable ? selectedBody : null;
}

export function getDraggedBodyMatrices(
  draggingSelectionBody: string | null,
  nextSelectionMatrix: number[],
  bodies: Record<string, ZapdosBodyState>
): Record<string, number[]> {
  if (!draggingSelectionBody) {
    return {};
  }
  const selectionState = bodies[draggingSelectionBody];
  if (!selectionState) {
    return {};
  }
  const startSelectionMatrix = new Matrix4().fromArray(selectionState.matrix);
  const inverseStartSelectionMatrix = startSelectionMatrix.clone().invert();
  const dragDelta = new Matrix4().multiplyMatrices(
    new Matrix4().fromArray(nextSelectionMatrix),
    inverseStartSelectionMatrix,
  );
  const result: Record<string, number[]> = {};
  for (const [body, state] of Object.entries(bodies)) {
    if (state.selectionBody !== draggingSelectionBody) {
      continue;
    }
    const nextBodyMatrix = new Matrix4().multiplyMatrices(
      dragDelta,
      new Matrix4().fromArray(state.matrix),
    );
    result[body] = nextBodyMatrix.toArray();
  }
  return result;
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
