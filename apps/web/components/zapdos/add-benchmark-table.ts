import { createSetSceneAssetsRequest, setSceneAssets, type SceneOperationStreamFactory } from "./zapdos-tool-api";

export function createAddBenchmarkTableRequest(): RequestInit {
  return createSetSceneAssetsRequest({
    assets: [{
      asset_id: "benchmark_table_000",
      motion: "static",
      placement: {
        kind: "floor_at_xy",
        xy: [0.5, 0],
        z_offset: 0,
        yaw: 0,
      },
    }],
  });
}

export async function addBenchmarkTable(
  sess: string,
  createEventSource?: SceneOperationStreamFactory,
) {
  const payload = await setSceneAssets(sess, {
    assets: [{
      asset_id: "benchmark_table_000",
      motion: "static",
      placement: {
        kind: "floor_at_xy",
        xy: [0.5, 0],
        z_offset: 0,
        yaw: 0,
      },
    }],
  }, createEventSource);
  const item = payload.items[0];
  if (!item) {
    throw new Error("Benchmark table was not created");
  }
  return {
    body: item.body,
    instance_id: item.instance_id,
    scene_revision: payload.scene_revision,
  };
}
