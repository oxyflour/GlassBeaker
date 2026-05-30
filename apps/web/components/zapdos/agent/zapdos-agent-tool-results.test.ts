import assert from "node:assert/strict";
import test from "node:test";

import {
  summarizeAddAssetsToSceneResult,
  summarizeRemoveAssetsFromSceneResult,
  summarizeSetSceneAssetsResult,
} from "./zapdos-agent-tool-results";

test("summarizeSetSceneAssetsResult returns a clear success message for one asset", () => {
  const result = summarizeSetSceneAssetsResult({
    scene_revision: "rev-2",
    items: [{
      asset_id: "table_000",
      body: "Scene_table_000_01",
      instance_id: "table_000_01",
    }],
  });

  assert.equal(result.ok, true);
  assert.equal(result.asset_count, 1);
  assert.equal(
    result.message,
    "Replaced the Zapdos overlay with 1 asset. Instance: table_000_01 (table_000) on body Scene_table_000_01. Scene revision: rev-2.",
  );
});

test("summarizeSetSceneAssetsResult lists multiple created instances", () => {
  const result = summarizeSetSceneAssetsResult({
    scene_revision: "rev-3",
    items: [
      {
        asset_id: "table_000",
        body: "Scene_table_000_01",
        instance_id: "table_000_01",
      },
      {
        asset_id: "mug_000",
        body: "Scene_mug_000_01",
        instance_id: "mug_000_01",
      },
    ],
  });

  assert.equal(result.asset_count, 2);
  assert.equal(
    result.message,
    "Replaced the Zapdos overlay with 2 assets. Instances: table_000_01 (table_000) on body Scene_table_000_01; mug_000_01 (mug_000) on body Scene_mug_000_01. Scene revision: rev-3.",
  );
});

test("summarizeAddAssetsToSceneResult returns an additive success message", () => {
  const result = summarizeAddAssetsToSceneResult({
    scene_revision: "rev-5",
    items: [{
      asset_id: "mug_000",
      body: "Scene_mug_000_01",
      instance_id: "mug_000_01",
    }],
  });

  assert.equal(result.ok, true);
  assert.equal(result.asset_count, 1);
  assert.equal(
    result.message,
    "Added 1 Zapdos overlay asset. Instance: mug_000_01 (mug_000) on body Scene_mug_000_01. Scene revision: rev-5.",
  );
});

test("summarizeRemoveAssetsFromSceneResult returns a clear success message", () => {
  const result = summarizeRemoveAssetsFromSceneResult({
    instance_ids: ["table_000_01", "mug_000_01"],
    scene_revision: "rev-4",
  });

  assert.equal(result.ok, true);
  assert.equal(
    result.message,
    "Removed 2 overlay assets: table_000_01, mug_000_01. Scene revision: rev-4.",
  );
});
