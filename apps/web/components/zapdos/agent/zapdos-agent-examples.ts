export const ZAPDOS_PLACEMENT_GUIDE = [
  'Use one of these placement shapes:',
  'floor_at_xy: { kind: "floor_at_xy", xy: [x, y], z_offset?: number, yaw?: number, quat?: [w, x, y, z], payload_quat?: [w, x, y, z] }',
  'on_top_of_body: { kind: "on_top_of_body", body: "Scene_table_000_01", xy: [x, y], gap?: number, yaw?: number, quat?: [w, x, y, z], payload_quat?: [w, x, y, z] }',
  'world_pose: { kind: "world_pose", pos: [x, y, z], quat: [w, x, y, z], payload_quat?: [w, x, y, z] }',
].join("\n");

export const ZAPDOS_SET_SCENE_ASSETS_EXAMPLE = [
  "set_scene_assets({",
  "  assets: [",
  "    {",
  '      asset_id: "table_000",',
  '      motion: "static",',
  '      placement: { kind: "floor_at_xy", xy: [0.0, 0.0], yaw: 0.0 },',
  "    },",
  "    {",
  '      asset_id: "benchmark_mug_001",',
  '      motion: "dynamic",',
  '      placement: { kind: "on_top_of_body", body: "Scene_table_000_01", xy: [0.0, 0.25], gap: 0.0 },',
  "    },",
  "    {",
  '      asset_id: "crate_000",',
  '      motion: "dynamic",',
  '      placement: { kind: "world_pose", pos: [0.4, -0.2, 0.5], quat: [1.0, 0.0, 0.0, 0.0] },',
  "    },",
  "  ],",
  "})",
].join("\n");
