import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRobotModelHref,
  DEFAULT_ROBOT_MODEL_KEY,
  getRobotModelKeyFromUsd,
  getRobotUsdForModel,
  readPersistedRobotModelKey,
  ROBOT_MODEL_STORAGE_KEY,
  resolveEffectiveRobotUsd,
  writePersistedRobotModelKey,
} from "./robot-model";

test("resolveEffectiveRobotUsd prefers the URL robot path over persisted state", () => {
  assert.equal(
    resolveEffectiveRobotUsd("deps/moz01/spirit01_model/urdf/moz1.urdf", "r1pro"),
    "deps/moz01/spirit01_model/urdf/moz1.urdf"
  );
});

test("resolveEffectiveRobotUsd falls back to the persisted robot key when the URL is absent", () => {
  assert.equal(
    resolveEffectiveRobotUsd(null, "moz1"),
    "deps/moz01/spirit01_model/urdf/moz1.urdf"
  );
});

test("resolveEffectiveRobotUsd falls back to the default robot for stale persisted values", () => {
  assert.equal(
    resolveEffectiveRobotUsd(null, "stale-value"),
    getRobotUsdForModel(DEFAULT_ROBOT_MODEL_KEY)
  );
});

test("getRobotModelKeyFromUsd only reverse-maps known robot paths", () => {
  assert.equal(getRobotModelKeyFromUsd("deps/galaxea/object/r1pro/r1pro.usda"), "r1pro");
  assert.equal(getRobotModelKeyFromUsd("deps/custom/custom.usd"), null);
});

test("robot model storage helpers persist the robot key instead of the robot path", () => {
  const memory = new Map<string, string>();
  const storage = {
    getItem(key: string) {
      return memory.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      memory.set(key, value);
    },
  };

  writePersistedRobotModelKey("moz1", storage);

  assert.equal(memory.get(ROBOT_MODEL_STORAGE_KEY), "moz1");
  assert.equal(readPersistedRobotModelKey(storage), "moz1");
});

test("buildRobotModelHref preserves unrelated query params and replaces robot_usd", () => {
  assert.equal(
    buildRobotModelHref(
      "/demo/zapdos",
      "scene_usd=C%3A%2Ftmp%2Fscene.usda&view=debug&robot_usd=old-value",
      "moz1"
    ),
    "/demo/zapdos?scene_usd=C%3A%2Ftmp%2Fscene.usda&view=debug&robot_usd=deps%2Fmoz01%2Fspirit01_model%2Furdf%2Fmoz1.urdf"
  );
});
