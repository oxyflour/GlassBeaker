import assert from "node:assert/strict";
import test from "node:test";

import {
  summarizeRemoveAssetFromSceneResult,
  summarizeSetSceneAssetsResult,
} from "./zapdos-agent-tool-results";

test("summarizeSetSceneAssetsResult returns a clear success message for one asset", () => {
  const result = summarizeSetSceneAssetsResult({
    op_id: "op-1",
    status: "started",
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
    "Started replacing the Zapdos overlay with 1 asset. Planned instance: table_000_01 (table_000) on body Scene_table_000_01. Operation: op-1.",
  );
});

test("summarizeSetSceneAssetsResult lists multiple created instances", () => {
  const result = summarizeSetSceneAssetsResult({
    op_id: "op-2",
    status: "started",
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
    "Started replacing the Zapdos overlay with 2 assets. Planned instances: table_000_01 (table_000) on body Scene_table_000_01; mug_000_01 (mug_000) on body Scene_mug_000_01. Operation: op-2.",
  );
});

test("summarizeRemoveAssetFromSceneResult returns a clear success message", () => {
  const result = summarizeRemoveAssetFromSceneResult({
    instance_id: "table_000_01",
    scene_revision: "rev-4",
  });

  assert.equal(result.ok, true);
  assert.equal(
    result.message,
    "Removed overlay asset table_000_01. Scene revision: rev-4.",
  );
});
