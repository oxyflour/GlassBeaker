import { CopilotRuntime, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { BuiltInAgent } from "@copilotkit/runtime/v2";
import type { NextRequest } from "next/server";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"

import { PREVIEW_ADDITIONAL_INSTRUCTIONS } from "../../../components/agent/preview/instructions";

const createModel = createOpenAICompatible({
  name: "custom",
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || "https://api.moonshot.cn/v1",
})

const agent = new BuiltInAgent({
  model: createModel(process.env.COPILOTKIT_MODEL?.trim() || "gpt-5.2"),
  prompt: [
    "Write React apps for users.",
    PREVIEW_ADDITIONAL_INSTRUCTIONS,
    "When the UI needs styling, include plain `.css` files in the frontend tool `files` payload and import them from the component tree.",
  ].join("\n\n")
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
