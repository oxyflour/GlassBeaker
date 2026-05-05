import assert from "node:assert/strict";
import test from "node:test";

import {
  addAssetToSceneToolArgsSchema,
  listSceneBodiesToolArgsSchema,
  removeAssetFromSceneToolArgsSchema,
  searchAssetsToolArgsSchema,
} from "./zapdos-agent-tool-schemas";

test("search_assets schema exposes query and top_k fields", () => {
  assert.deepEqual(Object.keys(searchAssetsToolArgsSchema.shape), ["query", "top_k"]);
  assert.equal(searchAssetsToolArgsSchema.safeParse({ query: "table" }).success, true);
  assert.equal(searchAssetsToolArgsSchema.safeParse({ top_k: 8 }).success, false);
});

test("list_scene_bodies schema only accepts an empty object", () => {
  assert.deepEqual(Object.keys(listSceneBodiesToolArgsSchema.shape), []);
  assert.equal(listSceneBodiesToolArgsSchema.safeParse({}).success, true);
  assert.equal(listSceneBodiesToolArgsSchema.safeParse({ extra: true }).success, false);
});

test("add_asset_to_scene schema keeps motion as an enum", () => {
  assert.deepEqual(addAssetToSceneToolArgsSchema.shape.motion.options, ["static", "dynamic"]);
});

test("add_asset_to_scene schema accepts the supported placement kinds", () => {
  assert.deepEqual(
    addAssetToSceneToolArgsSchema.parse({
      asset_id: "table_000",
      motion: "static",
      placement: {
        kind: "floor_at_xy",
        xy: [0, 0],
        z_offset: 0,
        yaw: 0,
      },
    }),
    {
      asset_id: "table_000",
      motion: "static",
      placement: {
        kind: "floor_at_xy",
        xy: [0, 0],
        z_offset: 0,
        yaw: 0,
      },
    }
  );

  assert.equal(
    addAssetToSceneToolArgsSchema.safeParse({
      asset_id: "table_000",
      motion: "dynamic",
      placement: {
        kind: "world_pose",
        pos: [0, 0, 0],
        quat: [1, 0, 0, 0],
      },
    }).success,
    true
  );
});

test("add_asset_to_scene schema rejects malformed placement payloads", () => {
  assert.throws(
    () => addAssetToSceneToolArgsSchema.parse({
      asset_id: "table_000",
      motion: "static",
      placement: {
        kind: "on_top_of_body",
        xy: [0, 0],
      },
    }),
    /body/i
  );
});

test("remove_asset_from_scene schema trims and requires a non-empty id", () => {
  assert.deepEqual(
    removeAssetFromSceneToolArgsSchema.parse({ instance_id: "  table_000_01  " }),
    { instance_id: "table_000_01" }
  );
  assert.throws(() => removeAssetFromSceneToolArgsSchema.parse({ instance_id: "   " }), /instance_id/i);
});
