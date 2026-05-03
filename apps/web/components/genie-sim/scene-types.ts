export type AssetSearchHit = {
  asset_id: string;
  description: Record<string, unknown>;
  url: string;
};

export type SceneObject = {
  assetId: string;
  description: Record<string, unknown>;
  id: string;
  keywords: string[];
  position: [number, number, number];
  quaternion: [number, number, number, number];
  size: [number, number, number];
};

export type SceneRelationNode = {
  id: string;
  tags?: string[];
};

export type SceneRelationEdge = {
  source: string;
  target: string;
};

export type SceneData = {
  assetsRoot: string;
  code: string;
  description: string;
  objects: SceneObject[];
  relations: {
    nodes?: SceneRelationNode[];
    links?: SceneRelationEdge[];
    edges?: SceneRelationEdge[];
  };
  sceneUsdaPath: string;
  seed: number;
};
