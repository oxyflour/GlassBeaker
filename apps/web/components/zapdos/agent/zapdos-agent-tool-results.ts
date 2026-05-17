type SetSceneAssetsResult = {
  op_id: string;
  items: Array<{ asset_id: string; body: string; instance_id: string }>;
  status: "started";
};

type RemoveAssetFromSceneResult = {
  instance_id: string;
  scene_revision: string;
};

export function summarizeSetSceneAssetsResult(result: SetSceneAssetsResult) {
  const assetCount = result.items.length;
  const summary = result.items
    .map((item) => `${item.instance_id} (${item.asset_id}) on body ${item.body}`)
    .join("; ");
  return {
    ok: true as const,
    ...result,
    asset_count: assetCount,
    message: `Started replacing the Zapdos overlay with ${assetCount} asset${assetCount === 1 ? "" : "s"}. Planned instance${assetCount === 1 ? "" : "s"}: ${summary}. Operation: ${result.op_id}.`,
  };
}

export function summarizeRemoveAssetFromSceneResult(result: RemoveAssetFromSceneResult) {
  return {
    ok: true as const,
    ...result,
    message: `Removed overlay asset ${result.instance_id}. Scene revision: ${result.scene_revision}.`,
  };
}
