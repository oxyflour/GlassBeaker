import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";

import { SEARCH_ASSETS_DESCRIPTION } from "../genie-sim";
import { postToolJson } from "../genie-sim/tool-client";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";
import {
  addAssetToSceneToolArgsSchema,
  listSceneBodiesToolArgsSchema,
  removeAssetFromSceneToolArgsSchema,
  searchAssetsToolArgsSchema,
} from "./zapdos-agent-tool-schemas";
import { addAssetToScene, listSceneBodies, removeAssetFromScene } from "./zapdos-tool-api";
import { useTypedTool } from "../../utils/agent/tool";

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
    name: "add_asset_to_scene",
    description: "Insert a session-local asset into the active Zapdos scene and rebuild the runtime.",
    followUp: true,
    parameters: addAssetToSceneToolArgsSchema,
    handler: async (args) => await addAssetToScene(sess, args),
  }, [sess]);

  useTypedTool({
    name: "remove_asset_from_scene",
    description: "Remove a session-local overlay asset by instance id.",
    followUp: true,
    parameters: removeAssetFromSceneToolArgsSchema,
    handler: async ({ instance_id }) => {
      return await removeAssetFromScene(sess, instance_id);
    },
  }, [sess]);
}
