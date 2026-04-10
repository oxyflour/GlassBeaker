import { CopilotRuntime, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { BuiltInAgent } from "@copilotkit/runtime/v2";
import type { NextRequest } from "next/server";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"

import { PREVIEW_ADDITIONAL_INSTRUCTIONS, PREVIEW_LIBRARY_CATALOG_PROMPT } from "../../../components/preview/instructions";

import { AbstractAgent } from "@copilotkit/react-core/v2";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";

const createModel = createOpenAICompatible({
  name: "custom",
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL || "https://api.moonshot.cn/v1",
})

// FIXME: https://github.com/CopilotKit/CopilotKit/issues/3623
const builtin = new BuiltInAgent({
  model: createModel(process.env.COPILOTKIT_MODEL?.trim() || "gpt-5.2"),
  prompt: [
    "Write React apps for users.",
    PREVIEW_ADDITIONAL_INSTRUCTIONS,
    PREVIEW_LIBRARY_CATALOG_PROMPT,
    "When the UI needs styling, include plain `.css` files in the frontend tool `files` payload and import them from the component tree.",
  ].join("\n\n")
});

const agents = { } as Record<string, AbstractAgent>
for (const { path, name } of JSON.parse(process.env.API_RUNTIME || '{}').agents || []) {
  agents[name] = new LangGraphHttpAgent({ url: `${process.env.API_REWRITE}${path.slice(1)}` })
}
agents.default = Object.values(agents)[0]
agents.builtin = builtin

const runtime = new CopilotRuntime({ agents });

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
