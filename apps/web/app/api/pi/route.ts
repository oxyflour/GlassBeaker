import { existsSync } from "node:fs";
import path from "node:path";

import { AuthStorage, createAgentSession, ModelRegistry, SessionManager } from "@mariozechner/pi-coding-agent";

export const runtime = "nodejs";

type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

type PiUsage = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
};

type PiStreamEvent =
  | { type: "start" }
  | { type: "text_start"; contentIndex: number }
  | { type: "text_delta"; contentIndex: number; delta: string }
  | { type: "text_end"; contentIndex: number }
  | { type: "done"; reason: "stop" | "length" | "toolUse"; usage: PiUsage }
  | { type: "error"; reason: "aborted" | "error"; errorMessage?: string; usage: PiUsage };

type PiRequestBody = {
  model?: {
    api?: string;
    provider?: string;
    id?: string;
  };
  context?: {
    systemPrompt?: string;
    messages?: PiMessage[];
  };
  options?: {
    apiKey?: string;
    reasoning?: ThinkingLevel;
  };
};

type PiMessage = {
  role?: string;
  content?: Array<{ type?: string; text?: string }>;
  usage?: PiUsage;
  stopReason?: string;
  errorMessage?: string;
};

const encoder = new TextEncoder();
const WORKSPACE_DIR = resolveWorkspaceDir();
const AGENT_DIR = path.join(WORKSPACE_DIR, ".pi", "agent");

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

function createUsage(): PiUsage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

function sumUsage(total: PiUsage, usage?: PiUsage) {
  if (!usage) {
    return total;
  }

  total.input += usage.input || 0;
  total.output += usage.output || 0;
  total.cacheRead += usage.cacheRead || 0;
  total.cacheWrite += usage.cacheWrite || 0;
  total.totalTokens += usage.totalTokens || 0;
  total.cost.input += usage.cost?.input || 0;
  total.cost.output += usage.cost?.output || 0;
  total.cost.cacheRead += usage.cost?.cacheRead || 0;
  total.cost.cacheWrite += usage.cost?.cacheWrite || 0;
  total.cost.total += usage.cost?.total || 0;

  return total;
}

function getMessageText(message: { content?: Array<{ type?: string; text?: string }> } | undefined) {
  if (!message?.content) {
    return "";
  }

  return message.content
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text || "")
    .join("");
}

function sendEvent(controller: ReadableStreamDefaultController<Uint8Array>, event: PiStreamEvent) {
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
  const prompt = messages?.[messages.length - 1];

  if (!model?.api || !model.provider || !model.id || !Array.isArray(messages) || !prompt || prompt.role !== "user") {
    return Response.json({ error: "Invalid Pi request payload." }, { status: 400 });
  }

  try {
    const authStorage = AuthStorage.inMemory();
    if (body.options?.apiKey) {
      authStorage.setRuntimeApiKey(model.provider, body.options.apiKey);
    }

    const modelRegistry = ModelRegistry.inMemory(authStorage);
    const { session } = await createAgentSession({
      cwd: WORKSPACE_DIR,
      agentDir: AGENT_DIR,
      model: model as never,
      thinkingLevel: body.options?.reasoning ?? "off",
      authStorage,
      modelRegistry,
      sessionManager: SessionManager.inMemory(),
    });

    session.agent.state.systemPrompt = context?.systemPrompt || session.agent.state.systemPrompt;
    session.agent.state.messages = messages.slice(0, -1) as never[];

    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          let closed = false;
          let streamStarted = false;
          let pendingSeparator = false;
          let currentAssistantText = "";
          let usage = createUsage();
          let lastAssistant: PiMessage | undefined;

          const finish = () => {
            if (closed) {
              return;
            }

            closed = true;
            unsubscribe();
            request.signal.removeEventListener("abort", abort);
            session.dispose();
            controller.close();
          };

          const ensureTextStream = () => {
            if (streamStarted) {
              return;
            }

            streamStarted = true;
            sendEvent(controller, { type: "start" });
            sendEvent(controller, { type: "text_start", contentIndex: 0 });
          };

          const pushText = (delta: string) => {
            if (!delta || closed) {
              return;
            }

            ensureTextStream();
            if (pendingSeparator) {
              sendEvent(controller, { type: "text_delta", contentIndex: 0, delta: "\n\n" });
              pendingSeparator = false;
            }
            sendEvent(controller, { type: "text_delta", contentIndex: 0, delta });
          };

          const finalize = () => {
            if (closed) {
              return;
            }

            const stopReason = lastAssistant?.stopReason;
            if (streamStarted) {
              sendEvent(controller, { type: "text_end", contentIndex: 0 });
            }

            if (stopReason === "error" || stopReason === "aborted") {
              sendEvent(controller, {
                type: "error",
                reason: stopReason,
                errorMessage: lastAssistant?.errorMessage,
                usage,
              });
            } else {
              sendEvent(controller, {
                type: "done",
                reason: stopReason === "length" || stopReason === "toolUse" ? stopReason : "stop",
                usage,
              });
            }

            finish();
          };

          const abort = () => {
            session.agent.abort();
          };

          const unsubscribe = session.subscribe((event) => {
            if (closed) {
              return;
            }

            if (event.type === "message_start" && event.message.role === "assistant") {
              currentAssistantText = "";
              pendingSeparator = streamStarted;
              return;
            }

            if ((event.type === "message_update" || event.type === "message_end") && event.message.role === "assistant") {
              const nextText = getMessageText(event.message as PiMessage);
              if (nextText.length > currentAssistantText.length) {
                pushText(nextText.slice(currentAssistantText.length));
              }
              currentAssistantText = nextText;

              if (event.type === "message_end") {
                lastAssistant = event.message as PiMessage;
                usage = sumUsage(usage, event.message.usage);
                pendingSeparator = false;
              }
              return;
            }

            if (event.type === "agent_end") {
              finalize();
            }
          });

          request.signal.addEventListener("abort", abort);

          void session.agent.prompt(prompt as never).catch((error) => {
            if (closed) {
              return;
            }

            if (streamStarted) {
              sendEvent(controller, { type: "text_end", contentIndex: 0 });
            }
            sendEvent(controller, {
              type: "error",
              reason: request.signal.aborted ? "aborted" : "error",
              errorMessage: error instanceof Error ? error.message : String(error),
              usage,
            });
            finish();
          });
        },
        cancel() {
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
