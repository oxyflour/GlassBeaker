import { createAddAssetToSceneRequest } from "./zapdos-tool-api";

export function createAddBenchmarkTableRequest(): RequestInit {
  return createAddAssetToSceneRequest({
    asset_id: "benchmark_table_000",
    motion: "static",
    placement: {
      kind: "floor_at_xy",
      xy: [0, 0],
      z_offset: 0,
      yaw: 0,
    },
  });
}

export async function addBenchmarkTable(sess: string) {
  const response = await fetch(
    `/python/zapdos/${sess}/call/add_asset_to_scene`,
    createAddBenchmarkTableRequest(),
  );
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return await response.json() as {
    body: string;
    instance_id: string;
    scene_revision: string;
  };
}
