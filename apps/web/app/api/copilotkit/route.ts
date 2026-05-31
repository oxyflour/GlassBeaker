import { CopilotRuntime, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { BuiltInAgent } from "@copilotkit/runtime/v2";
import type { NextRequest } from "next/server";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"

import { IFRAME_URL_ADDITIONAL_INSTRUCTIONS } from "../../../components/agent/iframe-url-tool";

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
    "Control the live iframe by opening browser URLs for users.",
    IFRAME_URL_ADDITIONAL_INSTRUCTIONS,
  ].join("\n\n")
});

type PythonRuntime = {
  agents?: Array<{ path?: string; name?: string }>;
};

let runtimePromise: Promise<CopilotRuntime> | null = null;

async function fetchPythonRuntime(): Promise<PythonRuntime> {
  const response = await fetch(new URL("runtime", process.env.API_REWRITE || "http://localhost:13001/"));
  if (!response.ok) {
    throw new Error(`Python runtime returned ${response.status}: ${await response.text()}`);
  }
  return await response.json() as PythonRuntime;
}

function createRuntime(apiRuntime: PythonRuntime): CopilotRuntime {
  const agents = {} as Record<string, AbstractAgent>;
  for (const { path, name } of apiRuntime.agents || []) {
    if (path && name) {
      agents[name] = new LangGraphHttpAgent({
        url: new URL(path, process.env.API_REWRITE || "http://localhost:13001/").toString(),
      });
    }
  }
  agents.default = Object.values(agents)[0] || builtin;
  agents.builtin = builtin;
  return new CopilotRuntime({ agents });
}

async function getRuntime(): Promise<CopilotRuntime> {
  runtimePromise ??= fetchPythonRuntime()
    .then(createRuntime)
    .catch((error) => {
      runtimePromise = null;
      throw error;
    });
  return await runtimePromise;
}

export const POST = async (request: NextRequest) => {
  if (!process.env.OPENAI_API_KEY?.trim()) {
    return Response.json(
      {
        error: "Missing OPENAI_API_KEY. Add it to apps/web/.env.local before using CopilotKit."
      },
      { status: 500 }
    );
  }

  let runtime: CopilotRuntime;
  try {
    runtime = await getRuntime();
  } catch (error) {
    console.warn("Python runtime is not ready", error);
    return Response.json(
      { error: "Python runtime is not ready" },
      { status: 503 }
    );
  }

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    endpoint: request.nextUrl.pathname
  });

  return handleRequest(request);
};
