"use client";

import { useCopilotAdditionalInstructions, useCopilotReadable, useFrontendTool } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-core/v2";

import {
  AgentPreview,
  PREVIEW_ADDITIONAL_INSTRUCTIONS,
  PREVIEW_CODE_PARAMETERS_CPK,
  PREVIEW_LIBRARY_CATALOG_PROMPT,
  PREVIEW_PROPS_PARAMETERS_CPK,
  PREVIEW_SET_APP_CODE_DESCRIPTION,
  PREVIEW_SET_APP_PROPS_DESCRIPTION,
  getPreviewLibraryReadableCatalog,
  useAgentPreviewState,
} from "../../../components/agent/preview";

export default function HomePage() {
  const preview = useAgentPreviewState();

  useCopilotAdditionalInstructions({ instructions: [PREVIEW_ADDITIONAL_INSTRUCTIONS, PREVIEW_LIBRARY_CATALOG_PROMPT].join("\n\n") }, []);

  useCopilotReadable({
    description: "Current entry component properties that the assistant can revise or keep stable while changing props.",
    value: preview.props,
  }, [preview.props]);
  useCopilotReadable({
    description: "Whitelisted repo preview components that can be imported directly instead of copying source into files.",
    value: getPreviewLibraryReadableCatalog(),
  }, []);

  useFrontendTool({
    name: "set_app_code",
    description: PREVIEW_SET_APP_CODE_DESCRIPTION,
    followUp: true,
    parameters: PREVIEW_CODE_PARAMETERS_CPK,
    handler: ({ entry, props, files }) => preview.setAppCode({ entry, props, files }),
    render: ({ status }) => (status === "inProgress" ? <div className="tool-badge">Building preview...</div> : <div className="tool-badge">Preview updated</div>),
  }, [preview.setAppCode]);

  useFrontendTool({
    name: "set_app_props",
    description: PREVIEW_SET_APP_PROPS_DESCRIPTION,
    followUp: true,
    parameters: PREVIEW_PROPS_PARAMETERS_CPK,
    handler: ({ props }) => preview.setAppProps(props),
    render: ({ status }) => (status === "inProgress" ? <div className="tool-badge">Updating props...</div> : <div className="tool-badge">Props updated</div>),
  }, [preview.setAppProps]);

  return (
    <div className="flex h-full w-full">
      { preview.hasApp ? (
        <div className="h-full flex-1">
          <AgentPreview className="h-full" files={ preview.files } props={ preview.props } />
        </div>
      ) : null }
      <CopilotChat className="copilotkit-fix" style={{ width: preview.hasApp ? 400 : "100%" }} />
    </div>
  );
}
