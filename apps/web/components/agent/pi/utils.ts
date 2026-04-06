import type {
  AgentMessage,
  Attachment,
  Model,
  UserMessageWithAttachments,
} from "@mariozechner/pi-web-ui";

import type { PiRequestMessage, PiSerializableAttachment } from "./protocol";
import type { ToolResultMessage, UserMessage } from "./types";

export const DEFAULT_SYSTEM_PROMPT = "";

export function classNames(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

function createEmptyUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

export function createUserMessage(input: string, attachments: Attachment[]): AgentMessage {
  if (attachments.length > 0) {
    const message: UserMessageWithAttachments = {
      role: "user-with-attachments",
      content: input,
      attachments,
      timestamp: Date.now(),
    };
    return message;
  }

  const message: UserMessage = {
    role: "user",
    content: [{ type: "text", text: input }],
    timestamp: Date.now(),
  };
  return message;
}

export function createTerminalAssistantMessage(
  errorMessage: string,
  model: Model<any> | undefined,
  stopReason: "error" | "aborted",
  partial: AgentMessage | null,
) {
  if (partial?.role === "assistant") {
    return {
      ...partial,
      stopReason,
      errorMessage: stopReason === "error" ? errorMessage : partial.errorMessage,
      usage: partial.usage ?? createEmptyUsage(),
    };
  }

  return {
    role: "assistant" as const,
    content: [{ type: "text" as const, text: "" }],
    api: model?.api || "",
    provider: model?.provider || "",
    model: model?.id || "",
    usage: createEmptyUsage(),
    stopReason,
    errorMessage: stopReason === "error" ? errorMessage : undefined,
    timestamp: Date.now(),
  };
}

export async function readRouteError(response: Response) {
  try {
    const data = await response.json();
    if (typeof data?.error === "string" && data.error) {
      return data.error;
    }
  } catch {
  }

  return `Pi route error: ${response.status} ${response.statusText}`;
}

export function isSameMessage(left: AgentMessage | undefined, right: AgentMessage | undefined) {
  return !!left && !!right && JSON.stringify(left) === JSON.stringify(right);
}

export function collectToolResultsById(messages: AgentMessage[]) {
  const toolResultsById = new Map<string, ToolResultMessage>();
  for (const message of messages) {
    if (message.role === "toolResult") {
      toolResultsById.set(message.toolCallId, message);
    }
  }
  return toolResultsById;
}

function serializeAttachment(attachment: Attachment): PiSerializableAttachment {
  const { id, type, fileName, mimeType, size, content, extractedText } = attachment;
  return { id, type, fileName, mimeType, size, content, extractedText };
}

export function serializeMessagesForPi(messages: AgentMessage[]) {
  return messages.map((message) => {
    if (message.role !== "user-with-attachments" || !message.attachments) {
      return message as PiRequestMessage;
    }

    return {
      ...message,
      attachments: message.attachments.map(serializeAttachment),
    } satisfies PiRequestMessage;
  });
}
