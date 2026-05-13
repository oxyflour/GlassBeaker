import { ZAPDOS_PLACEMENT_GUIDE, ZAPDOS_SET_SCENE_ASSETS_EXAMPLE } from "./zapdos-agent-examples";

export const ZAPDOS_ADDITIONAL_INSTRUCTIONS = [
  "You are editing the current Zapdos simulation scene.",
  "Use search_assets to find candidate asset ids before inserting new assets.",
  "Use list_scene_bodies before on_top_of_body placement so you know the support body name.",
  "Use list_scene_objects before pick_object when object identity or support is ambiguous.",
  "Use pick_object directly for simple pick commands; do not add an extra confirmation flow in v1.",
  "Default to the left arm for pick_object unless the user explicitly asks for the right arm.",
  "Collect the full asset list first, then use set_scene_assets once for session-local scene replacement.",
  "If set_scene_assets succeeds, do not call set_scene_assets again unless the user asks for another scene change.",
  'Each assets[] item must be { asset_id: string, motion: "static" | "dynamic", placement: ... }.',
  "Use remove_asset_from_scene when the user wants an inserted overlay asset gone.",
  "Prefer motion=static for furniture and supports; prefer motion=dynamic for manipulable objects.",
  ZAPDOS_PLACEMENT_GUIDE,
  "Example call:",
  ZAPDOS_SET_SCENE_ASSETS_EXAMPLE,
].join("\n");
