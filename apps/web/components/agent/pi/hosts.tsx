import type {
  AgentMessage,
  Attachment,
  Model,
  ThinkingLevel,
} from "@mariozechner/pi-web-ui";
import { createElement, useEffect, useRef } from "react";

import type {
  MessageEditorElement,
  MessageListElement,
  StreamingMessageContainerElement,
  ToolResultMessage,
} from "./types";

export function MessageListHost(props: { isStreaming: boolean; messages: AgentMessage[]; pendingToolCalls: Set<string> }) {
  const ref = useRef<MessageListElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (element) {
      element.messages = props.messages;
      element.tools = [];
      element.pendingToolCalls = props.pendingToolCalls;
      element.isStreaming = props.isStreaming;
    }
  }, [props.isStreaming, props.messages, props.pendingToolCalls]);

  return createElement("message-list", { ref });
}

export function StreamingMessageHost(props: {
  isStreaming: boolean;
  message: AgentMessage | null;
  pendingToolCalls: Set<string>;
  toolResultsById: Map<string, ToolResultMessage>;
}) {
  const ref = useRef<StreamingMessageContainerElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (element) {
      element.tools = [];
      element.pendingToolCalls = props.pendingToolCalls;
      element.toolResultsById = props.toolResultsById;
      element.isStreaming = props.isStreaming;
      element.setMessage(props.message, !props.isStreaming || props.message === null);
    }
  }, [props.isStreaming, props.message, props.pendingToolCalls, props.toolResultsById]);

  return createElement("streaming-message-container", { ref });
}

export function MessageEditorHost(props: {
  currentModel?: Model<any>;
  isStreaming: boolean;
  onAbort: () => void;
  onModelSelect: () => void;
  onSend: (input: string, attachments: Attachment[], onAccepted?: () => void) => void;
  onThinkingChange: (level: ThinkingLevel) => void;
  thinkingLevel: ThinkingLevel;
}) {
  const ref = useRef<MessageEditorElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (element) {
      element.currentModel = props.currentModel;
      element.isStreaming = props.isStreaming;
      element.onAbort = props.onAbort;
      element.onModelSelect = props.onModelSelect;
      element.onSend = (input, attachments) => {
        props.onSend(input, attachments, () => {
          element.value = "";
          element.attachments = [];
        });
      };
      element.onThinkingChange = props.onThinkingChange;
      element.showAttachmentButton = true;
      element.showModelSelector = true;
      element.showThinkingSelector = true;
      element.thinkingLevel = props.thinkingLevel;
    }
  }, [props]);

  return createElement("message-editor", { ref });
}
