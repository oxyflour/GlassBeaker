import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";

import { AuthStorage, createAgentSession, ModelRegistry, SessionManager } from "@mariozechner/pi-coding-agent";

import type { PiFrontendToolDefinition, PiRequestMessage } from "../../../components/agent/pi/protocol";
import { cleanupFrontendToolCalls, createFrontendTools } from "./frontend-tools";
import { convertPiMessages } from "./messages";

export const runtime = "nodejs";

type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
type PiEvent = Record<string, any> & { type: string };
type PiModel = {
  api?: string;
  id?: string;
  provider?: string;
};

type PiRequestBody = {
  model?: PiModel;
  context?: {
    frontendTools?: PiFrontendToolDefinition[];
    systemPrompt?: string;
    messages?: PiRequestMessage[];
  };
  options?: {
    apiKey?: string;
    reasoning?: ThinkingLevel;
  };
};

const encoder = new TextEncoder();
const WORKSPACE_DIR = resolveWorkspaceDir();
const AGENT_DIR = path.join(WORKSPACE_DIR, ".pi", "agent");
const ATTACHMENTS_DIR = path.join(AGENT_DIR, "uploads");

function resolveWorkspaceDir() {
  let current = process.cwd();

  for (let i = 0; i < 6; i += 1) {
    if (existsSync(path.join(current, "pnpm-workspace.yaml"))) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }

  return process.cwd();
}

function sendEvent(controller: ReadableStreamDefaultController<Uint8Array>, event: PiEvent) {
  controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
}

export async function POST(request: Request) {
  let body: PiRequestBody;

  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const model = body.model;
  const context = body.context;
  const messages = context?.messages;

  if (!model?.api || !model.provider || !model.id || !Array.isArray(messages) || messages.length === 0) {
    return Response.json({ error: "Invalid Pi request payload." }, { status: 400 });
  }

  try {
    const convertedMessages = await convertPiMessages(messages, WORKSPACE_DIR, ATTACHMENTS_DIR);
    const prompt = convertedMessages[convertedMessages.length - 1];
    if (!prompt || prompt.role !== "user") {
      return Response.json({ error: "Invalid Pi request payload." }, { status: 400 });
    }

    const authStorage = AuthStorage.inMemory();
    if (body.options?.apiKey) {
      authStorage.setRuntimeApiKey(model.provider, body.options.apiKey);
    }

    const modelRegistry = ModelRegistry.inMemory(authStorage);
    const requestId = randomUUID();
    let controllerRef: { current: ReadableStreamDefaultController<Uint8Array> | null } = { current: null };
    const { session } = await createAgentSession({
      cwd: WORKSPACE_DIR,
      agentDir: AGENT_DIR,
      customTools: createFrontendTools(requestId, context?.frontendTools, (event) => sendEvent(controllerRef.current!, event)),
      model: model as never,
      thinkingLevel: body.options?.reasoning ?? "off",
      authStorage,
      modelRegistry,
      sessionManager: SessionManager.inMemory(),
    });

    if (context?.systemPrompt) {
      session.agent.state.systemPrompt = `${session.agent.state.systemPrompt}\n\n${context.systemPrompt}`;
    }
    session.agent.state.messages = convertedMessages.slice(0, -1) as never[];

    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controllerRef = { current: controller };
          let closed = false;
          let unsubscribe = () => {};

          const finish = () => {
            if (closed) {
              return;
            }

            closed = true;
            unsubscribe();
            request.signal.removeEventListener("abort", abort);
            cleanupFrontendToolCalls(requestId);
            session.dispose();
            controller.close();
          };

          const abort = () => {
            session.agent.abort();
          };

          unsubscribe = session.subscribe((event) => {
            if (closed) {
              return;
            }

            sendEvent(controller, event as PiEvent);
            if (event.type === "agent_end") {
              finish();
            }
          });

          request.signal.addEventListener("abort", abort);

          void session.agent.prompt(prompt as never).catch(() => {
            if (closed) {
              return;
            }

            finish();
          });
        },
        cancel() {
          cleanupFrontendToolCalls(requestId);
          session.agent.abort();
          session.dispose();
        },
      }),
      {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          Connection: "keep-alive",
        },
      },
    );
  } catch (error) {
    return Response.json(
      {
        error: error instanceof Error ? error.message : "Failed to start Pi coding agent.",
      },
      { status: 500 },
    );
  }
}
