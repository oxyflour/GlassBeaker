"use client";

import dynamic from "next/dynamic";
import { Group, Panel, Separator } from "react-resizable-panels";

import {
  IframeUrlPreview,
  useIframeUrlPreviewState,
} from "../../../components/agent/iframe-url-preview";

import {
  IFRAME_URL_ADDITIONAL_INSTRUCTIONS,
  SET_IFRAME_URL_DESCRIPTION,
  SET_IFRAME_URL_PARAMETERS,
  SET_IFRAME_URL_TOOL_NAME,
} from "../../../components/agent/iframe-url-tool";
import { PiFrontendToolProvider, usePiFrontendTool } from "../../../components/agent/pi/frontend-tools";

const Pi = dynamic(() => import("../../../components/agent/pi"), { ssr: false });

function PiWebAgentPreviewDemo() {
  const preview = useIframeUrlPreviewState();

  usePiFrontendTool({
    name: SET_IFRAME_URL_TOOL_NAME,
    description: SET_IFRAME_URL_DESCRIPTION,
    followUp: true,
    parameters: SET_IFRAME_URL_PARAMETERS,
    handler: preview.setIframeUrl,
    render: ({ status }: { status: string }) => (status === "inProgress" ? "Opening URL..." : "URL opened"),
  }, [preview.setIframeUrl]);

  return <Group>
    <Panel>
      <Pi className="h-full" systemPrompt={ IFRAME_URL_ADDITIONAL_INSTRUCTIONS } />
    </Panel>
    {
      preview.url && <>
        <Separator />
        <Panel>
          <IframeUrlPreview url={ preview.url } />
        </Panel>
      </>
    }
  </Group>;
}

export default function PiWebDemoPage() {
  return (
    <PiFrontendToolProvider>
      <PiWebAgentPreviewDemo />
    </PiFrontendToolProvider>
  );
}
