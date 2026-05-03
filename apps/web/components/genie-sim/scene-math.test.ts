import assert from "node:assert/strict";
import test from "node:test";

import { simPointToThree } from "./scene-math";

test("simPointToThree maps genie_sim z-up coordinates into Three.js space", () => {
  assert.deepEqual(simPointToThree([1, 2, 3]), [-2, 3, -1]);
});
