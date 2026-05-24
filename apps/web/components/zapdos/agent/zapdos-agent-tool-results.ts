type SetSceneAssetsResult = {
  items: Array<{ asset_id: string; body: string; instance_id: string }>;
  scene_revision: string;
};

type RemoveAssetsFromSceneResult = {
  instance_ids: string[];
  scene_revision: string;
};

export function summarizeSetSceneAssetsResult(result: SetSceneAssetsResult) {
  const assetCount = result.items.length;
  const summary = summarizeAssetItems(result.items);
  return {
    ok: true as const,
    ...result,
    asset_count: assetCount,
    message: `Replaced the Zapdos overlay with ${assetCount} asset${assetCount === 1 ? "" : "s"}. Instance${assetCount === 1 ? "" : "s"}: ${summary}. Scene revision: ${result.scene_revision}.`,
  };
}

export function summarizeAddAssetsToSceneResult(result: SetSceneAssetsResult) {
  const assetCount = result.items.length;
  const summary = summarizeAssetItems(result.items);
  return {
    ok: true as const,
    ...result,
    asset_count: assetCount,
    message: `Added ${assetCount} Zapdos overlay asset${assetCount === 1 ? "" : "s"}. Instance${assetCount === 1 ? "" : "s"}: ${summary}. Scene revision: ${result.scene_revision}.`,
  };
}

export function summarizeRemoveAssetsFromSceneResult(result: RemoveAssetsFromSceneResult) {
  const assetCount = result.instance_ids.length;
  return {
    ok: true as const,
    ...result,
    asset_count: assetCount,
    message: `Removed ${assetCount} overlay asset${assetCount === 1 ? "" : "s"}: ${result.instance_ids.join(", ")}. Scene revision: ${result.scene_revision}.`,
  };
}

function summarizeAssetItems(items: SetSceneAssetsResult["items"]) {
  return items
    .map((item) => `${item.instance_id} (${item.asset_id}) on body ${item.body}`)
    .join("; ");
}
