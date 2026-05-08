import assert from "node:assert/strict";
import test from "node:test";

import { countLeaves, createInitialLayout, removeLeaf, splitLeaf, updateLeafTopic } from "./layout";

test("createInitialLayout builds a single panel", () => {
  const layout = createInitialLayout("/topic/a");
  assert.equal(countLeaves(layout), 1);
});

test("splitLeaf supports right and down splits", () => {
  let layout = createInitialLayout("/topic/a");
  layout = splitLeaf(layout, "panel-1", "right", "/topic/b");
  layout = splitLeaf(layout, "panel-2", "down", "/topic/c");

  assert.equal(countLeaves(layout), 3);
  const updated = updateLeafTopic(layout, "panel-3", "/topic/d");
  assert.equal(countLeaves(updated), 3);
});

test("removeLeaf collapses redundant split parents", () => {
  let layout = createInitialLayout("/topic/a");
  layout = splitLeaf(layout, "panel-1", "right", "/topic/b");
  const collapsed = removeLeaf(layout, "panel-2");

  assert.equal(countLeaves(collapsed), 1);
});
