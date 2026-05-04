import assert from "node:assert/strict";
import test from "node:test";

import {
  applySceneHotkey,
  isSelectionClick,
  pickEditableBodyFromHits,
  shouldApplyBodyPose,
  type ZapdosSceneState,
} from "./zapdos-scene-state";

test("applySceneHotkey switches transform mode with W and E", () => {
  const state: ZapdosSceneState = { mode: "translate", selectedBody: "Scene_Crate" };

  assert.deepEqual(applySceneHotkey(state, "e"), { mode: "rotate", selectedBody: "Scene_Crate" });
  assert.deepEqual(applySceneHotkey(state, "w"), state);
});

test("applySceneHotkey clears the current selection on Escape", () => {
  const state: ZapdosSceneState = { mode: "rotate", selectedBody: "Scene_Crate" };

  assert.deepEqual(applySceneHotkey(state, "Escape"), { mode: "rotate", selectedBody: null });
});

test("shouldApplyBodyPose ignores remote updates for the body being dragged", () => {
  assert.equal(shouldApplyBodyPose("Scene_Crate", "Scene_Crate"), false);
  assert.equal(shouldApplyBodyPose("RobotLink", "Scene_Crate"), true);
});

test("pickEditableBodyFromHits skips non-editable hits and picks the first editable body", () => {
  assert.equal(pickEditableBodyFromHits([
    { editable: false, body: "RobotLink" },
    { editable: false, body: null },
    { editable: true, body: "Scene_Crate" },
  ]), "Scene_Crate");
});

test("pickEditableBodyFromHits returns null when nothing editable is hit", () => {
  assert.equal(pickEditableBodyFromHits([
    { editable: false, body: "RobotLink" },
    { editable: false, body: null },
  ]), null);
});

test("isSelectionClick stays true for small pointer movement", () => {
  assert.equal(isSelectionClick({ x: 10, y: 20 }, { x: 13, y: 23 }), true);
});

test("isSelectionClick rejects drag-sized movement", () => {
  assert.equal(isSelectionClick({ x: 10, y: 20 }, { x: 20, y: 20 }), false);
});
