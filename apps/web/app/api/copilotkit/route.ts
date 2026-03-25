import { CopilotRuntime, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { BuiltInAgent } from "@copilotkit/runtime/v2";
import type { NextRequest } from "next/server";

const agent = new BuiltInAgent({
  model: process.env.COPILOTKIT_MODEL?.trim() || "openai:gpt-5.2",
  prompt: [
    "You are the GlassBeaker assistant for a minimal Electron + Next.js starter.",
    "Help users understand how the renderer, bundled standalone server, and packaging flow work.",
    "Keep answers concise, practical, and specific to this app when possible."
  ].join(" ")
});

const runtime = new CopilotRuntime({
  agents: {
    default: agent
  }
});

export const POST = async (request: NextRequest) => {
  if (!process.env.OPENAI_API_KEY?.trim()) {
    return Response.json(
      {
        error: "Missing OPENAI_API_KEY. Add it to apps/web/.env.local before using CopilotKit."
      },
      { status: 500 }
    );
  }

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    endpoint: request.nextUrl.pathname
  });

  return handleRequest(request);
};
