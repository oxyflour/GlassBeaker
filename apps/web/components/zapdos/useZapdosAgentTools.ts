import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";

import { SEARCH_ASSETS_DESCRIPTION, SEARCH_ASSETS_PARAMETERS_CPK } from "../genie-sim";
import { postToolJson } from "../genie-sim/tool-client";
import { ZAPDOS_ADDITIONAL_INSTRUCTIONS } from "./zapdos-agent-instructions";
import { addAssetToScene, listSceneBodies, removeAssetFromScene } from "./zapdos-tool-api";
import { useTool } from "../../utils/agent/tool";

export function useZapdosAgentTools(sess: string) {
  useCopilotAdditionalInstructions({ instructions: ZAPDOS_ADDITIONAL_INSTRUCTIONS }, [sess]);

  useTool({
    name: "search_assets",
    description: SEARCH_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: SEARCH_ASSETS_PARAMETERS_CPK,
    handler: async (args) => postToolJson(
      fetch,
      "/python/genie_sim/search_assets",
      { query: args.query, top_k: typeof args.top_k === "number" ? args.top_k : 8 },
      "Asset search failed"
    ),
  }, []);

  useTool({
    name: "list_scene_bodies",
    description: "List editable scene bodies with support metadata for placement.",
    followUp: true,
    parameters: [] as never[],
    handler: async () => await listSceneBodies(sess),
  }, [sess]);

  useTool({
    name: "add_asset_to_scene",
    description: "Insert a session-local asset into the active Zapdos scene and rebuild the runtime.",
    followUp: true,
    parameters: [
      { name: "asset_id", type: "string", required: true, description: "Exact asset id from search_assets." },
      { name: "motion", type: "string", required: true, description: "Use static or dynamic." },
      { name: "placement", type: "object", required: true, description: "Placement payload using floor_at_xy, on_top_of_body, or world_pose." },
    ] as never[],
    handler: async (args) => await addAssetToScene(sess, args as AddAssetToolArgs),
  }, [sess]);

  useTool({
    name: "remove_asset_from_scene",
    description: "Remove a session-local overlay asset by instance id.",
    followUp: true,
    parameters: [
      { name: "instance_id", type: "string", required: true, description: "Overlay instance id to remove." },
    ] as never[],
    handler: async (args) => await removeAssetFromScene(sess, String((args as RemoveAssetToolArgs).instance_id)),
  }, [sess]);
}

type AddAssetToolArgs = {
  asset_id: string;
  motion: "dynamic" | "static";
  placement: Record<string, unknown>;
};

type RemoveAssetToolArgs = {
  instance_id: string;
};
