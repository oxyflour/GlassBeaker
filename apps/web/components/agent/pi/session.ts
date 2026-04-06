import {
  ApiKeyPromptDialog,
  type AgentMessage,
  type AppStorage,
  type Attachment,
  defaultConvertToLlm,
  type Model,
  type ThinkingLevel,
} from "@mariozechner/pi-web-ui";
import { useEffect, useRef, useState } from "react";

import { applyAgentEvent } from "./session-events";
import { consumeAgentStream } from "./stream";
import {
  collectToolResultsById,
  createTerminalAssistantMessage,
  createUserMessage,
  DEFAULT_SYSTEM_PROMPT,
  isSameMessage,
} from "./utils";

type SessionOptions = {
  currentModel?: Model<any>;
  currentThinkingLevel: ThinkingLevel;
  ensureStorage: () => Promise<AppStorage>;
  systemPrompt?: string;
};

export function usePiSession(options: SessionOptions) {
  const abortControllerRef = useRef<AbortController | null>(null);
  const isStreamingRef = useRef(false);
  const messagesRef = useRef<AgentMessage[]>([]);
  const streamingMessageRef = useRef<AgentMessage | null>(null);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [streamingMessage, setStreamingMessage] = useState<AgentMessage | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingToolCalls, setPendingToolCalls] = useState<Set<string>>(new Set());
  useEffect(() => void (isStreamingRef.current = isStreaming), [isStreaming]);
  useEffect(() => void (messagesRef.current = messages), [messages]);
  useEffect(() => void (streamingMessageRef.current = streamingMessage), [streamingMessage]);
  useEffect(() => () => abortControllerRef.current?.abort(), []);
  function appendTerminalMessage(errorMessage: string, stopReason: "error" | "aborted", model = options.currentModel) {
    const terminalMessage = createTerminalAssistantMessage(errorMessage, model, stopReason, streamingMessageRef.current);
    setStreamingMessage(null);
    setPendingToolCalls(new Set());
    setIsStreaming(false);
    setMessages((previousMessages) =>
      isSameMessage(previousMessages[previousMessages.length - 1], terminalMessage)
        ? previousMessages
        : [...previousMessages, terminalMessage],
    );
  }
  function send(input: string, attachments: Attachment[], onAccepted?: () => void) {
    void (async () => {
      if (isStreamingRef.current) {
        return;
      }

      const model = options.currentModel;
      if (!model) {
        appendTerminalMessage("No model configured.", "error");
        return;
      }

      const storage = await options.ensureStorage();
      let apiKey = await storage.providerKeys.get(model.provider);
      if (!apiKey) {
        const success = await ApiKeyPromptDialog.prompt(model.provider);
        if (!success) {
          return;
        }
        apiKey = await storage.providerKeys.get(model.provider);
        if (!apiKey) {
          return;
        }
      }
      abortControllerRef.current?.abort();
      const userMessage = createUserMessage(input, attachments);
      const nextMessages = [...messagesRef.current, userMessage];
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setMessages(nextMessages);
      setStreamingMessage(null);
      setPendingToolCalls(new Set());
      setIsStreaming(true);
      onAccepted?.();

      try {
        const response = await fetch("/api/pi", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model,
            context: {
              systemPrompt: options.systemPrompt || DEFAULT_SYSTEM_PROMPT,
              messages: defaultConvertToLlm(nextMessages),
            },
            options: { apiKey, reasoning: options.currentThinkingLevel },
          }),
          signal: controller.signal,
        });

        await consumeAgentStream({
          controller,
          onEvent: (event) =>
            applyAgentEvent(event, {
              addPendingToolCall: (toolCallId) => setPendingToolCalls((previous) => new Set(previous).add(toolCallId)),
              appendMessage: (message) => setMessages((previous) => [...previous, message]),
              appendTerminalMessage: (message) =>
                setMessages((previous) => (isSameMessage(previous[previous.length - 1], message) ? previous : [...previous, message])),
              clearPendingToolCalls: () => setPendingToolCalls(new Set()),
              removePendingToolCall: (toolCallId) =>
                setPendingToolCalls((previous) => {
                  const next = new Set(previous);
                  next.delete(toolCallId);
                  return next;
                }),
              setIsStreaming,
              setStreamingMessage,
            }),
          onTerminalFailure: (message, stopReason) => appendTerminalMessage(message, stopReason, model),
          response,
        });
      } catch (error) {
        const aborted = controller.signal.aborted;
        const message = aborted ? "Request aborted" : error instanceof Error ? error.message : String(error);
        appendTerminalMessage(message, aborted ? "aborted" : "error", model);
      } finally {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    })();
  }

  function abort() {
    abortControllerRef.current?.abort();
  }

  return {
    abort,
    isStreaming,
    messages,
    pendingToolCalls,
    send,
    streamingMessage,
    toolResultsById: collectToolResultsById(messages),
  };
}
