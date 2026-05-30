import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";

import { SEARCH_ASSETS_DESCRIPTION } from "../../genie-sim";
import { postToolJson } from "../../genie-sim/tool-client";
import { ZAPDOS_ADD_ASSETS_TO_SCENE_EXAMPLE, ZAPDOS_SET_SCENE_ASSETS_EXAMPLE } from "./zapdos-agent-examples";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";
import {
  summarizeAddAssetsToSceneResult,
  summarizeRemoveAssetsFromSceneResult,
  summarizeSetSceneAssetsResult,
} from "./zapdos-agent-tool-results";
import {
  addAssetsToSceneToolArgsSchema,
  listPlacementBodiesToolArgsSchema,
  removeAssetsFromSceneToolArgsSchema,
  searchAssetsToolArgsSchema,
  setSceneAssetsToolArgsSchema,
} from "./zapdos-agent-tool-schemas";
import {
  listManipulationObjectsToolArgsSchema,
  pickObjectToolArgsSchema,
} from "./zapdos-manipulation-tool-schemas";
import { listManipulationObjects, pickObject } from "./zapdos-manipulation-tool-api";
import { addAssetsToScene, listPlacementBodies, removeAssetsFromScene, setSceneAssets } from "./zapdos-tool-api";
import { useTypedTool } from "../../../utils/agent/tool";

export const SET_SCENE_ASSETS_DESCRIPTION = [
  "Replace the session-local Zapdos overlay asset list and rebuild the runtime once.",
  'Pass { assets: [{ asset_id, motion, placement }] } where placement uses floor_at_xy, on_top_of_body, or world_pose.',
  "Example:",
  ZAPDOS_SET_SCENE_ASSETS_EXAMPLE,
].join("\n");

export const ADD_ASSETS_TO_SCENE_DESCRIPTION = [
  "Add assets to the session-local Zapdos overlay scene without replacing existing overlay assets.",
  'Pass { assets: [{ asset_id, motion, placement }] } where placement uses floor_at_xy, on_top_of_body, or world_pose.',
  "Example:",
  ZAPDOS_ADD_ASSETS_TO_SCENE_EXAMPLE,
].join("\n");

export function useZapdosAgentTools(sess: string) {
  useCopilotAdditionalInstructions({ instructions: ZAPDOS_ADDITIONAL_INSTRUCTIONS }, [sess]);

  useTypedTool({
    name: "search_assets",
    description: SEARCH_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: searchAssetsToolArgsSchema,
    handler: async ({ query, top_k }) => {
      return await postToolJson(
        fetch,
        "/python/genie_sim/search_assets",
        { query, top_k: top_k ?? 8 },
        "Asset search failed"
      );
    },
  }, []);

  useTypedTool({
    name: "list_placement_bodies",
    description: "List editable body names and support metadata for scene placement.",
    followUp: true,
    parameters: listPlacementBodiesToolArgsSchema,
    handler: async () => await listPlacementBodies(sess),
  }, [sess]);

  useTypedTool({
    name: "list_manipulation_objects",
    description: "List semantic manipulation objects for resolving pick targets and supports.",
    followUp: true,
    parameters: listManipulationObjectsToolArgsSchema,
    handler: async () => await listManipulationObjects(sess),
  }, [sess]);

  useTypedTool({
    name: "pick_object",
    description: "Execute a natural-language pick command. Use directly for simple picks; include support_query when needed.",
    followUp: true,
    parameters: pickObjectToolArgsSchema,
    handler: async (args) => await pickObject(sess, args),
  }, [sess]);

  useTypedTool({
    name: "set_scene_assets",
    description: SET_SCENE_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: setSceneAssetsToolArgsSchema,
    handler: async (args) => summarizeSetSceneAssetsResult(await setSceneAssets(sess, args)),
  }, [sess]);

  useTypedTool({
    name: "add_assets_to_scene",
    description: ADD_ASSETS_TO_SCENE_DESCRIPTION,
    followUp: true,
    parameters: addAssetsToSceneToolArgsSchema,
    handler: async (args) => summarizeAddAssetsToSceneResult(await addAssetsToScene(sess, args)),
  }, [sess]);

  useTypedTool({
    name: "remove_assets_from_scene",
    description: "Remove session-local overlay assets by instance id.",
    followUp: true,
    parameters: removeAssetsFromSceneToolArgsSchema,
    handler: async ({ instance_ids }) => {
      return summarizeRemoveAssetsFromSceneResult(await removeAssetsFromScene(sess, instance_ids));
    },
  }, [sess]);
}
