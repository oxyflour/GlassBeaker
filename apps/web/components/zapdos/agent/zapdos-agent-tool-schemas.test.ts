import assert from "node:assert/strict";
import test from "node:test";

import {
  addAssetsToSceneToolArgsSchema,
  listPlacementBodiesToolArgsSchema,
  removeAssetsFromSceneToolArgsSchema,
  searchAssetsToolArgsSchema,
  sceneAssetSchema,
  setSceneAssetsToolArgsSchema,
} from "./zapdos-agent-tool-schemas";

test("search_assets schema exposes query and top_k fields", () => {
  assert.deepEqual(Object.keys(searchAssetsToolArgsSchema.shape), ["query", "top_k"]);
  assert.equal(searchAssetsToolArgsSchema.safeParse({ query: "table" }).success, true);
  assert.equal(searchAssetsToolArgsSchema.safeParse({ top_k: 8 }).success, false);
});

test("list_placement_bodies schema only accepts an empty object", () => {
  assert.deepEqual(Object.keys(listPlacementBodiesToolArgsSchema.shape), []);
  assert.equal(listPlacementBodiesToolArgsSchema.safeParse({}).success, true);
  assert.equal(listPlacementBodiesToolArgsSchema.safeParse({ extra: true }).success, false);
});

test("set_scene_assets schema requires a non-empty assets array", () => {
  const parsed = setSceneAssetsToolArgsSchema.parse({
    assets: [{
      asset_id: "table_000",
      motion: "static",
      placement: {
        kind: "floor_at_xy",
        xy: [0, 0],
        z_offset: 0,
        yaw: 0,
      },
    }],
  });

  assert.equal(parsed.assets.length, 1);
  assert.throws(() => setSceneAssetsToolArgsSchema.parse({ assets: [] }), /assets/i);
});

test("add_assets_to_scene schema requires a non-empty assets array", () => {
  const parsed = addAssetsToSceneToolArgsSchema.parse({
    assets: [{
      asset_id: "mug_000",
      motion: "dynamic",
      placement: {
        kind: "floor_at_xy",
        xy: [0.5, 0.25],
      },
    }],
  });

  assert.equal(parsed.assets.length, 1);
  assert.throws(() => addAssetsToSceneToolArgsSchema.parse({ assets: [] }), /assets/i);
});

test("set_scene_assets schema keeps motion as an enum", () => {
  const parsed = setSceneAssetsToolArgsSchema.parse({
    assets: [{
      asset_id: "table_000",
      motion: "dynamic",
      placement: {
        kind: "world_pose",
        pos: [0, 0, 0],
        quat: [1, 0, 0, 0],
      },
    }],
  });

  assert.equal(parsed.assets[0]?.motion, "dynamic");
});

test("set_scene_assets schema accepts the supported placement kinds", () => {
  assert.deepEqual(
    setSceneAssetsToolArgsSchema.parse({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: {
          kind: "floor_at_xy",
          xy: [0, 0],
          z_offset: 0,
          yaw: 0,
        },
      }],
    }),
    {
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: {
          kind: "floor_at_xy",
          xy: [0, 0],
          z_offset: 0,
          yaw: 0,
        },
      }],
    }
  );

  assert.equal(
    setSceneAssetsToolArgsSchema.safeParse({
      assets: [{
        asset_id: "table_000",
        motion: "dynamic",
        placement: {
          kind: "world_pose",
          pos: [0, 0, 0],
          quat: [1, 0, 0, 0],
        },
      }],
    }).success,
    true
  );
});

test("set_scene_assets schema rejects malformed placement payloads", () => {
  assert.throws(
    () => setSceneAssetsToolArgsSchema.parse({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: {
          kind: "on_top_of_body",
          xy: [0, 0],
        },
      }],
    }),
    /body/i
  );
});

test("set_scene_assets schema descriptions include a concrete payload shape", () => {
  const assetsDescription = setSceneAssetsToolArgsSchema.shape.assets.description ?? "";
  const assetDescription = sceneAssetSchema.description ?? "";

  assert.match(assetsDescription, /asset_id/i);
  assert.match(assetsDescription, /motion/i);
  assert.match(assetsDescription, /floor_at_xy/);
  assert.match(assetsDescription, /on_top_of_body/);
  assert.match(assetsDescription, /world_pose/);
  assert.match(assetDescription, /placement/i);
});

test("add_assets_to_scene schema description uses the additive tool name", () => {
  const assetsDescription = addAssetsToSceneToolArgsSchema.shape.assets.description ?? "";

  assert.match(assetsDescription, /add_assets_to_scene/);
  assert.doesNotMatch(assetsDescription, /set_scene_assets\(/);
});

test("remove_assets_from_scene schema trims and requires non-empty ids", () => {
  assert.deepEqual(
    removeAssetsFromSceneToolArgsSchema.parse({ instance_ids: ["  table_000_01  ", "mug_000_01"] }),
    { instance_ids: ["table_000_01", "mug_000_01"] }
  );
  assert.throws(() => removeAssetsFromSceneToolArgsSchema.parse({ instance_ids: [] }), /instance_ids/i);
  assert.throws(() => removeAssetsFromSceneToolArgsSchema.parse({ instance_ids: ["   "] }), /instance_ids/i);
});
