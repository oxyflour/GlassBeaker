"use client";

import { CopilotKit, useCopilotAdditionalInstructions, useFrontendTool } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { Group, Panel, Separator } from "react-resizable-panels";

import {
  GENIE_SIM_AGENT_NAME,
  OpenInZapdosLink,
  SCENE_ADDITIONAL_INSTRUCTIONS,
  SCENE_PARAMETERS_CPK,
  SceneViewer,
  SceneUsdaBadge,
  SEARCH_ASSETS_DESCRIPTION,
  SEARCH_ASSETS_PARAMETERS_CPK,
  SET_SCENE_DESCRIPTION,
  useSceneState,
} from "../../../components/genie-sim";
import { postToolJson } from "../../../components/genie-sim/tool-client";
import type { AssetSearchHit, SceneData } from "../../../components/genie-sim";

export default function AgentGenieSimPage() {
  return (
    <CopilotKit
      agent={ GENIE_SIM_AGENT_NAME }
      enableInspector={ false }
      runtimeUrl="/api/copilotkit"
      showDevConsole={ false }>
      <GenieSimWorkspace />
    </CopilotKit>
  );
}

function GenieSimWorkspace() {
  const { hasScene, scene, setSceneData } = useSceneState();

  useCopilotAdditionalInstructions({ instructions: SCENE_ADDITIONAL_INSTRUCTIONS }, []);

  useFrontendTool({
    name: "search_assets",
    description: SEARCH_ASSETS_DESCRIPTION,
    followUp: true,
    parameters: SEARCH_ASSETS_PARAMETERS_CPK,
    handler: async (args: Record<string, unknown>) => {
      const data = await postToolJson<{ assetsRoot: string; items?: AssetSearchHit[] }>(
        fetch,
        "/python/genie_sim/search_assets",
        {
          query: args.query,
          top_k: typeof args.top_k === "number" ? args.top_k : 8,
        },
        "Asset search failed"
      );
      if ("error" in data) {
        return data;
      }
      const items = Array.isArray(data.items) ? data.items as AssetSearchHit[] : [];
      return {
        assets_root: data.assetsRoot,
        items,
      };
    },
    render: ({ status }) => (
      <div className="tool-badge">
        {status === "inProgress" ? "Searching assets..." : "Asset search complete"}
      </div>
    ),
  }, []);

  useFrontendTool({
    name: "generate_scene",
    description: SET_SCENE_DESCRIPTION,
    followUp: true,
    parameters: SCENE_PARAMETERS_CPK,
    handler: async (args: Record<string, unknown>) => {
      const data = await postToolJson<SceneData>(
        fetch,
        "/python/genie_sim/execute",
        { code: args.code },
        "Scene execution failed"
      );
      if ("error" in data) {
        return data;
      }
      setSceneData(data);
      return {
        object_count: data.objects?.length || 0,
        ok: true,
        scene_id: data.description,
      };
    },
    render: ({ status }) => (
      <div className="tool-badge">
        {status === "inProgress" ? "Building scene..." : "Scene updated"}
      </div>
    ),
  }, [setSceneData]);

  return (
    <Group>
      <Panel defaultSize={ hasScene ? 42 : 100 } minSize={ 28 }>
        <CopilotChat className="copilotkit-fix" />
      </Panel>
      {hasScene && scene && (
        <>
          <Separator />
          <Panel defaultSize={ 58 } minSize={ 30 }>
            <div className="relative h-full min-h-0">
              <div className="absolute inset-0"><SceneViewer scene={ scene } /></div>
              <OpenInZapdosLink sceneUsdaPath={ scene.sceneUsdaPath } />
              <SceneUsdaBadge path={ scene.sceneUsdaPath } />
            </div>
          </Panel>
        </>
      )}
    </Group>
  );
}
