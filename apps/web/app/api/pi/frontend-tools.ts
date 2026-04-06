import type { ToolDefinition } from "@mariozechner/pi-coding-agent";

import type {
  PiFrontendToolDefinition,
  PiFrontendToolParameter,
  PiFrontendToolRequestEvent,
  PiFrontendToolResultPayload,
} from "../../../components/agent/pi/protocol";

type PendingFrontendToolCall = {
  reject: (error: Error) => void;
  requestId: string;
  resolve: (payload: PiFrontendToolResultPayload) => void;
};

const pendingFrontendToolCalls = new Map<string, PendingFrontendToolCall>();

function createKey(requestId: string, toolCallId: string) {
  return `${requestId}:${toolCallId}`;
}

function buildParameterSchema(parameter: PiFrontendToolParameter) {
  const description = parameter.description ? { description: parameter.description } : {};
  switch (parameter.type) {
    case "number":
      return { type: "number", ...description };
    case "boolean":
      return { type: "boolean", ...description };
    case "array":
      return { type: "array", items: {}, ...description };
    case "object":
      return { additionalProperties: true, type: "object", ...description };
    default:
      return { type: "string", ...description };
  }
}

function buildParameters(parameters: PiFrontendToolParameter[]) {
  return {
    additionalProperties: false,
    properties: Object.fromEntries(parameters.map((parameter) => [parameter.name, buildParameterSchema(parameter)])),
    required: parameters.filter((parameter) => parameter.required !== false).map((parameter) => parameter.name),
    type: "object",
  };
}

function formatFrontendToolResult(result: unknown) {
  if (typeof result === "string") {
    return result;
  }
  if (result === undefined) {
    return "Done";
  }
  try {
    return JSON.stringify(result, null, 2);
  } catch {
    return String(result);
  }
}

function normalizeFrontendToolResult(payload: PiFrontendToolResultPayload) {
  if (payload.error) {
    return { content: [{ type: "text" as const, text: payload.error }], isError: true };
  }

  const result = payload.result;
  if (result && typeof result === "object" && "content" in result && Array.isArray((result as any).content)) {
    return result as any;
  }

  return {
    content: [{ type: "text" as const, text: formatFrontendToolResult(result) }],
    details: result === undefined ? undefined : result,
  };
}

function waitForFrontendToolResult(requestId: string, toolCallId: string, signal?: AbortSignal) {
  return new Promise<PiFrontendToolResultPayload>((resolve, reject) => {
    const key = createKey(requestId, toolCallId);
    const cleanup = () => signal?.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      pendingFrontendToolCalls.delete(key);
      reject(new Error(`Frontend tool "${toolCallId}" was aborted.`));
    };

    pendingFrontendToolCalls.set(key, {
      reject: (error) => {
        cleanup();
        reject(error);
      },
      requestId,
      resolve: (payload) => {
        cleanup();
        resolve(payload);
      },
    });
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export function createFrontendTools(
  requestId: string,
  tools: PiFrontendToolDefinition[] | undefined,
  emit: (event: PiFrontendToolRequestEvent) => void,
) {
  const enabledTools = new Map<string, PiFrontendToolDefinition>();
  for (const tool of tools || []) {
    if (tool.available === "disabled") {
      continue;
    }
    enabledTools.set(tool.name, tool);
  }

  return Array.from(enabledTools.values()).map(
    (tool) =>
      ({
        description: tool.description,
        execute: async (toolCallId: string, params: unknown, signal?: AbortSignal) => {
          const response = waitForFrontendToolResult(requestId, toolCallId, signal);
          emit({ type: "frontend_tool_request", requestId, toolCallId, toolName: tool.name, args: params });
          return normalizeFrontendToolResult(await response);
        },
        label: tool.name,
        name: tool.name,
        parameters: buildParameters(tool.parameters) as any,
        promptSnippet: tool.description,
      }) satisfies ToolDefinition,
  );
}

export function resolveFrontendToolCall(payload: PiFrontendToolResultPayload) {
  const entry = pendingFrontendToolCalls.get(createKey(payload.requestId, payload.toolCallId));
  if (!entry) {
    return false;
  }

  pendingFrontendToolCalls.delete(createKey(payload.requestId, payload.toolCallId));
  entry.resolve(payload);
  return true;
}

export function cleanupFrontendToolCalls(requestId: string) {
  for (const [key, entry] of pendingFrontendToolCalls.entries()) {
    if (entry.requestId !== requestId) {
      continue;
    }

    pendingFrontendToolCalls.delete(key);
    entry.reject(new Error(`Frontend tools for request "${requestId}" were cancelled.`));
  }
}
