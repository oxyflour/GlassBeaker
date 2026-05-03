import assert from "node:assert/strict";
import test from "node:test";

import { SCENE_ADDITIONAL_INSTRUCTIONS } from "./scene-instructions";

test("scene prompt includes runnable code patterns and execute safety rules", () => {
  assert.match(SCENE_ADDITIONAL_INSTRUCTIONS, /Copy this skeleton and fill in asset ids/i);
  assert.match(SCENE_ADDITIONAL_INSTRUCTIONS, /def place_on_top\(/);
  assert.match(SCENE_ADDITIONAL_INSTRUCTIONS, /Never put @register\(\) on helpers that do not return Shape/i);
  assert.match(SCENE_ADDITIONAL_INSTRUCTIONS, /Never call \.add\(\.\.\.\) on keywords, Shape, or any Python list/i);
  assert.match(SCENE_ADDITIONAL_INSTRUCTIONS, /Never use undefined type names like Scene/i);
  assert.match(
    SCENE_ADDITIONAL_INSTRUCTIONS,
    /Call extra registered scene functions via library_call\("function_name"/i
  );
});
