"use client";

import { useCopilotAdditionalInstructions, useFrontendTool } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { Group, Panel, Separator } from "react-resizable-panels";

import {
  SCENE_ADDITIONAL_INSTRUCTIONS,
  SCENE_PARAMETERS_CPK,
  SET_SCENE_DESCRIPTION,
  SceneViewer,
  useSceneState,
} from "../../../components/genie-sim";
import type { SceneData } from "../../../components/genie-sim";

export default function GenieSimPage() {
  const { scene, hasScene, setSceneData } = useSceneState();

  useCopilotAdditionalInstructions(
    { instructions: SCENE_ADDITIONAL_INSTRUCTIONS },
    [],
  );

  useFrontendTool({
    name: "generate_scene",
    description: SET_SCENE_DESCRIPTION,
    followUp: true,
    parameters: SCENE_PARAMETERS_CPK,
    handler: async (args: Record<string, unknown>) => {
      try {
        const res = await fetch("/python/genie_sim/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: args.code as string }),
        });
        const data = await res.json();
        if (!res.ok) return { error: data.detail || "Execution failed" };
        if (!data?.objects || !Array.isArray(data.objects))
          return { error: "Invalid scene data returned" };
        setSceneData(data as SceneData);
        return { ok: true };
      } catch (e) {
        return { error: String(e) };
      }
    },
    render: ({ status }) =>
      status === "inProgress"
        ? <div className="tool-badge">Running genie_sim...</div>
        : <div className="tool-badge">Scene rendered</div>,
  }, [setSceneData]);

  return (
    <Group>
      <Panel>
        <CopilotChat
          className="copilotkit-fix"
        />
      </Panel>
      {hasScene && (
        <>
          <Separator />
          <Panel>
            <SceneViewer scene={scene} />
          </Panel>
        </>
      )}
    </Group>
  );
}
