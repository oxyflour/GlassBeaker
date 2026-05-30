import assert from "node:assert/strict";
import test from "node:test";
import { Matrix4 } from "three";

import {
  applySceneHotkey,
  clearMissingSelection,
  getDraggedBodyMatrices,
  getTransformBodyName,
  isSelectionClick,
  pickSelectableBodyFromHits,
  shouldReloadSceneRevision,
  shouldApplyBodyPose,
  type ZapdosSceneState,
} from "./zapdos-scene-state";

function translationMatrix(x: number, y: number, z: number) {
  return new Matrix4().makeTranslation(x, y, z).toArray();
}

test("applySceneHotkey switches transform mode with W and E", () => {
  const state: ZapdosSceneState = { mode: "translate", selectedBody: "Scene_Crate" };

  assert.deepEqual(applySceneHotkey(state, "e"), { mode: "rotate", selectedBody: "Scene_Crate" });
  assert.deepEqual(applySceneHotkey(state, "w"), state);
});

test("applySceneHotkey clears the current selection on Escape", () => {
  const state: ZapdosSceneState = { mode: "rotate", selectedBody: "Scene_Crate" };

  assert.deepEqual(applySceneHotkey(state, "Escape"), { mode: "rotate", selectedBody: null });
});

test("shouldApplyBodyPose ignores remote updates for all bodies linked to the dragged selection", () => {
  const bodies = {
    Scene_Crate: { movable: true, selectionBody: "Scene_Crate", matrix: translationMatrix(5, 0, 0) },
    Root_base_link: { movable: true, selectionBody: "Root_base_link", matrix: translationMatrix(0, 0, 0) },
    Arm_link: { movable: false, selectionBody: "Root_base_link", matrix: translationMatrix(1, 0, 0) },
  };

  assert.equal(shouldApplyBodyPose("Root_base_link", "Root_base_link", bodies), false);
  assert.equal(shouldApplyBodyPose("Arm_link", "Root_base_link", bodies), false);
  assert.equal(shouldApplyBodyPose("Scene_Crate", "Root_base_link", bodies), true);
});

test("pickSelectableBodyFromHits skips empty hits and picks the first body", () => {
  assert.equal(pickSelectableBodyFromHits([
    { editable: false, body: null, selectionBody: null },
    { editable: true, body: "Scene_Crate", selectionBody: null },
  ]), "Scene_Crate");
});

test("pickSelectableBodyFromHits returns the mapped selection body before the raw hit body", () => {
  assert.equal(pickSelectableBodyFromHits([
    { editable: false, body: "Arm_link", selectionBody: "Root_base_link" },
    { editable: true, body: "Scene_Crate", selectionBody: null },
  ]), "Root_base_link");
});

test("pickSelectableBodyFromHits returns null when no body is hit", () => {
  assert.equal(pickSelectableBodyFromHits([
    { editable: false, body: null, selectionBody: null },
    { editable: true, body: null, selectionBody: null },
  ]), null);
});

test("getTransformBodyName only returns movable selections", () => {
  assert.equal(getTransformBodyName("Root_base_link", {
    Root_base_link: { movable: true, selectionBody: "Root_base_link", matrix: translationMatrix(0, 0, 0) },
    Arm_link: { movable: false, selectionBody: "Root_base_link", matrix: translationMatrix(1, 0, 0) },
  }), "Root_base_link");
  assert.equal(getTransformBodyName("Arm_link", {
    Arm_link: { movable: false, selectionBody: "Root_base_link", matrix: translationMatrix(1, 0, 0) },
  }), null);
  assert.equal(getTransformBodyName(null, {
    Root_base_link: { movable: true, selectionBody: "Root_base_link", matrix: translationMatrix(0, 0, 0) },
  }), null);
});

test("getDraggedBodyMatrices applies the root drag delta to the whole robot selection", () => {
  const bodies = {
    Root_base_link: { movable: true, selectionBody: "Root_base_link", matrix: translationMatrix(0, 0, 0) },
    Arm_link: { movable: false, selectionBody: "Root_base_link", matrix: translationMatrix(1, 0, 0) },
    Wrist_link: { movable: false, selectionBody: "Root_base_link", matrix: translationMatrix(1, 2, 0) },
    Scene_Crate: { movable: true, selectionBody: "Scene_Crate", matrix: translationMatrix(9, 0, 0) },
  };

  const next = getDraggedBodyMatrices("Root_base_link", translationMatrix(3, 0, 0), bodies);

  assert.deepEqual(next, {
    Root_base_link: translationMatrix(3, 0, 0),
    Arm_link: translationMatrix(4, 0, 0),
    Wrist_link: translationMatrix(4, 2, 0),
  });
});

test("isSelectionClick stays true for small pointer movement", () => {
  assert.equal(isSelectionClick({ x: 10, y: 20 }, { x: 13, y: 23 }), true);
});

test("isSelectionClick rejects drag-sized movement", () => {
  assert.equal(isSelectionClick({ x: 10, y: 20 }, { x: 20, y: 20 }), false);
});

test("shouldReloadSceneRevision ignores duplicate revisions", () => {
  assert.equal(shouldReloadSceneRevision("rev-1", "rev-1"), false);
  assert.equal(shouldReloadSceneRevision("rev-1", "rev-2"), true);
});

test("shouldReloadSceneRevision reloads duplicate revisions when forced", () => {
  assert.equal(shouldReloadSceneRevision("rev-1", "rev-1", { force: true }), true);
});

test("clearMissingSelection drops a body that is no longer present", () => {
  assert.equal(clearMissingSelection("Scene_table_000_01", new Set(["Scene_crate_000_01"])), null);
  assert.equal(clearMissingSelection("Scene_table_000_01", new Set(["Scene_table_000_01"])), "Scene_table_000_01");
});
