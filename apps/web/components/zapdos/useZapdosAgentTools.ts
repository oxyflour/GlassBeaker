import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";

import { SEARCH_ASSETS_DESCRIPTION } from "../genie-sim";
import { postToolJson } from "../genie-sim/tool-client";
import { ZAPDOS_SET_SCENE_ASSETS_EXAMPLE } from "./zapdos-agent-examples";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";
import {
  summarizeRemoveAssetFromSceneResult,
  summarizeSetSceneAssetsResult,
} from "./zapdos-agent-tool-results";
import {
  listSceneBodiesToolArgsSchema,
  removeAssetFromSceneToolArgsSchema,
  searchAssetsToolArgsSchema,
  setSceneAssetsToolArgsSchema,
} from "./zapdos-agent-tool-schemas";
import { listSceneBodies, removeAssetFromScene, setSceneAssets } from "./zapdos-tool-api";
import { useTypedTool } from "../../utils/agent/tool";

export const SET_SCENE_ASSETS_DESCRIPTION = [
  "Replace the session-local Zapdos overlay asset list and rebuild the runtime once.",
  'Pass { assets: [{ asset_id, motion, placement }] } where placement uses floor_at_xy, on_top_of_body, or world_pose.',
  "Example:",
  ZAPDOS_SET_SCENE_ASSETS_EXAMPLE,
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
    name: "list_scene_bodies",
    description: "List editable scene bodies with support metadata for placement.",
    followUp: true,
    parameters: listSceneBodiesToolArgsSchema,
    handler: async () => await listSceneBodies(sess),
  }, [sess]);

  useTypedTool({
    name: "set_scene_assets",
    description: SET_SCENE_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: setSceneAssetsToolArgsSchema,
    handler: async (args) => summarizeSetSceneAssetsResult(await setSceneAssets(sess, args)),
  }, [sess]);

  useTypedTool({
    name: "remove_asset_from_scene",
    description: "Remove a session-local overlay asset by instance id.",
    followUp: true,
    parameters: removeAssetFromSceneToolArgsSchema,
    handler: async ({ instance_id }) => {
      return summarizeRemoveAssetFromSceneResult(await removeAssetFromScene(sess, instance_id));
    },
  }, [sess]);
}
