import { z } from "zod";

import {
  ZAPDOS_ADD_ASSETS_TO_SCENE_EXAMPLE,
  ZAPDOS_PLACEMENT_GUIDE,
  ZAPDOS_SET_SCENE_ASSETS_EXAMPLE,
} from "./zapdos-agent-examples";

const xySchema = z.tuple([z.number(), z.number()]);
const posSchema = z.tuple([z.number(), z.number(), z.number()]);
const quatSchema = z.tuple([z.number(), z.number(), z.number(), z.number()]);

export const floorAtXyPlacementSchema = z.object({
  kind: z.literal("floor_at_xy").describe("Place the asset on the floor at an x/y location."),
  xy: xySchema.describe("Floor x/y position as [x, y]."),
  z_offset: z.number().optional().describe("Optional vertical offset from the floor."),
  yaw: z.number().optional().describe("Optional yaw in radians when quat is not provided."),
  quat: quatSchema.optional().describe("Optional world quaternion [w, x, y, z]."),
  payload_quat: quatSchema.optional().describe("Optional payload-local quaternion [w, x, y, z]."),
}).strict();

export const onTopOfBodyPlacementSchema = z.object({
  kind: z.literal("on_top_of_body").describe("Place the asset on top of an editable scene body."),
  body: z.string().trim().min(1).describe("Editable scene body name from list_placement_bodies."),
  xy: xySchema.describe("Target x/y position in world coordinates as [x, y]."),
  gap: z.number().optional().describe("Optional vertical gap above the support surface."),
  yaw: z.number().optional().describe("Optional yaw in radians when quat is not provided."),
  quat: quatSchema.optional().describe("Optional world quaternion [w, x, y, z]."),
  payload_quat: quatSchema.optional().describe("Optional payload-local quaternion [w, x, y, z]."),
}).strict();

export const worldPosePlacementSchema = z.object({
  kind: z.literal("world_pose").describe("Place the asset at an explicit world pose."),
  pos: posSchema.describe("World position as [x, y, z]."),
  quat: quatSchema.describe("World quaternion as [w, x, y, z]."),
  payload_quat: quatSchema.optional().describe("Optional payload-local quaternion [w, x, y, z]."),
}).strict();

export const placementSchema = z.discriminatedUnion("kind", [
  floorAtXyPlacementSchema,
  onTopOfBodyPlacementSchema,
  worldPosePlacementSchema,
]).describe(ZAPDOS_PLACEMENT_GUIDE);

export const searchAssetsToolArgsSchema = z.object({
  query: z.string().trim().min(1).describe("English asset query or exact asset id."),
  top_k: z.number().optional().describe("Maximum number of asset candidates to return."),
}).strict();

export const listPlacementBodiesToolArgsSchema = z.object({}).strict();

export const sceneAssetSchema = z.object({
  asset_id: z.string().trim().min(1).describe("Exact asset id from search_assets."),
  motion: z.enum(["static", "dynamic"]).describe("Use static or dynamic."),
  placement: placementSchema,
}).strict().describe('Each assets[] item must be { asset_id, motion, placement }.');

export const setSceneAssetsToolArgsSchema = z.object({
  assets: z.array(sceneAssetSchema).min(1).describe([
    "Full replacement asset list for the current Zapdos overlay scene.",
    'Each item must include asset_id, motion, and placement.',
    ZAPDOS_PLACEMENT_GUIDE,
    "Example:",
    ZAPDOS_SET_SCENE_ASSETS_EXAMPLE,
  ].join("\n")),
}).strict();

export const addAssetsToSceneToolArgsSchema = z.object({
  assets: z.array(sceneAssetSchema).min(1).describe([
    "Asset list to add to the current Zapdos overlay scene.",
    'Each item must include asset_id, motion, and placement.',
    ZAPDOS_PLACEMENT_GUIDE,
    "Example:",
    ZAPDOS_ADD_ASSETS_TO_SCENE_EXAMPLE,
  ].join("\n")),
}).strict();

export const removeAssetsFromSceneToolArgsSchema = z.object({
  instance_ids: z.array(z.string().trim().min(1)).min(1).describe("Overlay instance ids to remove."),
}).strict();

export type SearchAssetsToolArgs = z.infer<typeof searchAssetsToolArgsSchema>;
export type SceneAsset = z.infer<typeof sceneAssetSchema>;
export type SetSceneAssetsToolArgs = z.infer<typeof setSceneAssetsToolArgsSchema>;
export type AddAssetsToSceneToolArgs = z.infer<typeof addAssetsToSceneToolArgsSchema>;
export type RemoveAssetsFromSceneToolArgs = z.infer<typeof removeAssetsFromSceneToolArgsSchema>;
