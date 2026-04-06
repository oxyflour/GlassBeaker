import type { AgentMessage } from "@mariozechner/pi-web-ui";

import type { AgentEvent } from "./types";

type SessionActions = {
  addPendingToolCall: (toolCallId: string) => void;
  appendMessage: (message: AgentMessage) => void;
  appendTerminalMessage: (message: AgentMessage) => void;
  clearPendingToolCalls: () => void;
  removePendingToolCall: (toolCallId: string) => void;
  setIsStreaming: (value: boolean) => void;
  setStreamingMessage: (message: AgentMessage | null) => void;
};

export function applyAgentEvent(event: AgentEvent, actions: SessionActions) {
  switch (event.type) {
    case "agent_start":
      actions.setIsStreaming(true);
      return;
    case "message_start":
      if (event.message.role !== "user") {
        actions.setStreamingMessage(event.message);
      }
      return;
    case "message_update":
      actions.setStreamingMessage(event.message);
      return;
    case "message_end":
      if (event.message.role !== "user") {
        actions.setStreamingMessage(null);
        actions.appendMessage(event.message);
      }
      return;
    case "tool_execution_start":
      actions.addPendingToolCall(event.toolCallId);
      return;
    case "tool_execution_end":
      actions.removePendingToolCall(event.toolCallId);
      return;
    case "agent_end": {
      const terminalMessage = event.messages.find(
        (message) => message.role === "assistant" && (message.stopReason === "error" || message.stopReason === "aborted"),
      );

      actions.setStreamingMessage(null);
      actions.clearPendingToolCalls();
      actions.setIsStreaming(false);
      if (terminalMessage) {
        actions.appendTerminalMessage(terminalMessage);
      }
    }
  }
}
