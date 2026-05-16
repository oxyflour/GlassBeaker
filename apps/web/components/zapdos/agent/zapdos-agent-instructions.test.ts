import assert from "node:assert/strict";
import test from "node:test";

import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";
import { SET_SCENE_ASSETS_DESCRIPTION } from "./useZapdosAgentTools";

test("Zapdos additional instructions include a canonical set_scene_assets example", () => {
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /set_scene_assets/i);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /asset_id/i);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /motion/i);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /floor_at_xy/);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /on_top_of_body/);
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /world_pose/);
});

test("Zapdos additional instructions tell the agent not to retry successful scene mutations", () => {
  assert.match(ZAPDOS_ADDITIONAL_INSTRUCTIONS, /do not call set_scene_assets again/i);
});

test("set_scene_assets tool description includes the canonical payload example", () => {
  assert.match(SET_SCENE_ASSETS_DESCRIPTION, /asset_id/i);
  assert.match(SET_SCENE_ASSETS_DESCRIPTION, /motion/i);
  assert.match(SET_SCENE_ASSETS_DESCRIPTION, /floor_at_xy/);
  assert.match(SET_SCENE_ASSETS_DESCRIPTION, /on_top_of_body/);
  assert.match(SET_SCENE_ASSETS_DESCRIPTION, /world_pose/);
});
