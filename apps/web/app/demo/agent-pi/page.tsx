"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

import { PiFrontendToolProvider, usePiFrontendTool } from "../../../components/agent/pi/frontend-tools";
import {
  AgentPreview,
  buildPreviewSystemPrompt,
  PREVIEW_CODE_PARAMETERS,
  PREVIEW_PROPS_PARAMETERS,
  PREVIEW_SET_APP_CODE_DESCRIPTION,
  PREVIEW_SET_APP_PROPS_DESCRIPTION,
  useAgentPreviewState,
} from "../../../components/agent/preview";

const Pi = dynamic(() => import("../../../components/agent/pi"), { ssr: false });

function PiAgentPreviewDemo() {
  const preview = useAgentPreviewState();

  usePiFrontendTool({
    name: "set_app_code",
    description: PREVIEW_SET_APP_CODE_DESCRIPTION,
    followUp: true,
    parameters: PREVIEW_CODE_PARAMETERS,
    handler: ({ entry, props, files }) => preview.setAppCode({ entry, props, files }),
    render: ({ status }: { status: string }) => (status === "inProgress" ? "Building preview..." : "Preview updated"),
  }, [preview.setAppCode]);

  usePiFrontendTool({
    name: "set_app_props",
    description: PREVIEW_SET_APP_PROPS_DESCRIPTION,
    followUp: true,
    parameters: PREVIEW_PROPS_PARAMETERS,
    handler: ({ props }) => preview.setAppProps(props),
    render: ({ status }: { status: string }) => (status === "inProgress" ? "Updating props..." : "Props updated"),
  }, [preview.setAppProps]);

  const systemPrompt = useMemo(() => buildPreviewSystemPrompt(preview.props), [preview.props]);

  return (
    <div className="flex h-screen w-full">
      { preview.hasApp ? (
        <div className="h-full flex-1">
          <AgentPreview className="h-full" files={ preview.files } props={ preview.props } />
        </div>
      ) : null }
      <div className="h-full" style={{ width: preview.hasApp ? 400 : "100%" }}>
        <Pi className="h-full" systemPrompt={ systemPrompt } />
      </div>
    </div>
  );
}

export default function PiDemoPage() {
  return (
    <PiFrontendToolProvider>
      <PiAgentPreviewDemo />
    </PiFrontendToolProvider>
  );
}
