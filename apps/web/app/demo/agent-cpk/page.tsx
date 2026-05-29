"use client";

import { useCopilotAdditionalInstructions } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-core/v2";
import { Group, Panel, Separator } from "react-resizable-panels";

import {
  IframeUrlPreview,
  useIframeUrlPreviewState,
} from "../../../components/agent/iframe-url-preview";
import {
  IFRAME_URL_ADDITIONAL_INSTRUCTIONS,
  SET_IFRAME_URL_DESCRIPTION,
  SET_IFRAME_URL_SCHEMA,
  SET_IFRAME_URL_TOOL_NAME,
} from "../../../components/agent/iframe-url-tool";
import { useTypedTool } from "../../../utils/agent/tool";

export default function HomePage() {
  const preview = useIframeUrlPreviewState();

  useCopilotAdditionalInstructions({ instructions: IFRAME_URL_ADDITIONAL_INSTRUCTIONS }, []);

  useTypedTool({
    name: SET_IFRAME_URL_TOOL_NAME,
    description: SET_IFRAME_URL_DESCRIPTION,
    followUp: true,
    parameters: SET_IFRAME_URL_SCHEMA,
    handler: preview.setIframeUrl,
    render: ({ status }) => (status === "inProgress" ? <div className="tool-badge">Opening URL...</div> : <div className="tool-badge">URL opened</div>),
  }, [preview.setIframeUrl]);

  return <Group>
    <Panel>
      <CopilotChat className="copilotkit-fix" />
    </Panel>
    {
        preview.url && <>
            <Separator />
            <Panel>
                <IframeUrlPreview url={ preview.url } />
            </Panel>
        </>
    }
  </Group>
}
