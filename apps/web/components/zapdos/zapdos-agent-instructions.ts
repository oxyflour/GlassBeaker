export const ZAPDOS_ADDITIONAL_INSTRUCTIONS = [
  "You are editing the current Zapdos simulation scene.",
  "Use search_assets to find candidate asset ids before inserting new assets.",
  "Use list_scene_bodies before on_top_of_body placement so you know the support body name.",
  "Use add_asset_to_scene for session-local insertion only.",
  "Use remove_asset_from_scene when the user wants an inserted overlay asset gone.",
  "Prefer motion=static for furniture and supports; prefer motion=dynamic for manipulable objects.",
].join("\n");
