import {
  createSetSceneAssetsRequest,
  setSceneAssets,
  type SceneTaskOptions,
  type SetSceneAssetsInput,
} from "../../agent/zapdos-tool-api";

const BENCHMARK_TABLE_ASSETS: SetSceneAssetsInput["assets"] = [{
  asset_id: "benchmark_table_000",
  motion: "static",
  placement: {
    kind: "floor_at_xy",
    xy: [0.45, 0],
    z_offset: 0,
    yaw: 0,
  },
}, {
  asset_id: "benchmark_building_blocks_006",
  motion: "dynamic",
  placement: {
    kind: "on_top_of_body",
    body: "Scene_benchmark_table_000_01",
    xy: [0.20, 0.24],
    gap: 0,
    yaw: 0,
  },
}];

export function createAddBenchmarkTableRequest(): RequestInit {
  return createSetSceneAssetsRequest({
    assets: BENCHMARK_TABLE_ASSETS,
  });
}

export async function addBenchmarkTable(
  sess: string,
  options?: SceneTaskOptions,
  onSceneRevision?: (revision: string) => void,
) {
  const payload = await setSceneAssets(sess, {
    assets: BENCHMARK_TABLE_ASSETS,
  }, options);
  const item = payload.items.find((candidate) => candidate.asset_id === "benchmark_table_000");
  if (!item) {
    throw new Error("Benchmark table was not created");
  }
  onSceneRevision?.(payload.scene_revision);
  return {
    body: item.body,
    instance_id: item.instance_id,
    scene_revision: payload.scene_revision,
  };
}
