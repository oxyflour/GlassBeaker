import assert from "node:assert/strict";
import test from "node:test";

import {
  buildZapdosInitStreamUrl,
  buildZapdosSessionStorageKey,
  parseZapdosInitEvent,
} from "./zapdos-import";

test("buildZapdosInitStreamUrl includes encoded scene and robot params", () => {
  const url = buildZapdosInitStreamUrl("sess-1", "C:/tmp/a scene.usda", "deps/galaxea/object/r1pro/r1pro.usda");
  assert.equal(
    url,
    "/python/zapdos/sess-1/init/start?scene_usd=C%3A%2Ftmp%2Fa+scene.usda&robot_usd=deps%2Fgalaxea%2Fobject%2Fr1pro%2Fr1pro.usda"
  );
});

test("buildZapdosSessionStorageKey changes when scene changes", () => {
  assert.notEqual(
    buildZapdosSessionStorageKey("C:/tmp/scene-a.usda", null),
    buildZapdosSessionStorageKey("C:/tmp/scene-b.usda", null)
  );
});

test("buildZapdosSessionStorageKey keeps raw robot_usd strings distinct", () => {
  assert.notEqual(
    buildZapdosSessionStorageKey(null, "deps/galaxea/object/r1pro/r1pro.usda"),
    buildZapdosSessionStorageKey(null, "deps/spirit01_model/USD/Moz1_robot_only.usda")
  );
});

test("parseZapdosInitEvent recognizes error payloads", () => {
  assert.deepEqual(parseZapdosInitEvent("error: scene_usd not found"), {
    phase: "error",
    message: "scene_usd not found",
  });
});

test("buildZapdosInitStreamUrl omits query when no import params are present", () => {
  assert.equal(buildZapdosInitStreamUrl("sess-1", null, null), "/python/zapdos/sess-1/init/start");
});
