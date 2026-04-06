import type {
  AgentMessage,
  Attachment,
  Model,
  ThinkingLevel,
} from "@mariozechner/pi-web-ui";

import type { PiFrontendToolRequestEvent } from "./protocol";

export type AgentEvent =
  | { type: "agent_start" }
  | { type: "agent_end"; messages: AgentMessage[] }
  | { type: "turn_start" }
  | { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
  | { type: "message_start"; message: AgentMessage }
  | { type: "message_update"; message: AgentMessage }
  | { type: "message_end"; message: AgentMessage }
  | { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
  | { type: "tool_execution_update"; toolCallId: string; toolName: string; args: any; partialResult: any }
  | { type: "tool_execution_end"; toolCallId: string; toolName: string; result: any; isError: boolean }
  | PiFrontendToolRequestEvent;

export type ToolResultMessage = Extract<AgentMessage, { role: "toolResult" }>;
export type UserMessage = Extract<AgentMessage, { role: "user" }>;

export type MessageListElement = HTMLElement & {
  isStreaming: boolean;
  messages: AgentMessage[];
  onCostClick?: () => void;
  pendingToolCalls?: Set<string>;
  tools: any[];
};

export type MessageEditorElement = HTMLElement & {
  currentModel?: Model<any>;
  isStreaming: boolean;
  value: string;
  attachments: Attachment[];
  onAbort?: () => void;
  onModelSelect?: () => void;
  onSend?: (input: string, attachments: Attachment[]) => void | Promise<void>;
  onThinkingChange?: (level: ThinkingLevel) => void;
  showAttachmentButton: boolean;
  showModelSelector: boolean;
  showThinkingSelector: boolean;
  thinkingLevel: ThinkingLevel;
};

export type StreamingMessageContainerElement = HTMLElement & {
  isStreaming: boolean;
  onCostClick?: () => void;
  pendingToolCalls?: Set<string>;
  setMessage: (message: AgentMessage | null, immediate?: boolean) => void;
  toolResultsById?: Map<string, ToolResultMessage>;
  tools: any[];
};
