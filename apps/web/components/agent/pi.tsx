'use client'

import "@mariozechner/pi-web-ui";

import {
  ApiKeyPromptDialog,
  type AgentMessage,
  AppStorage,
  type Attachment,
  type CustomProvider,
  CustomProvidersStore,
  IndexedDBStorageBackend,
  type Model,
  ModelSelector,
  ProviderKeysStore,
  SessionsStore,
  SettingsStore,
  type ThinkingLevel,
  type UserMessageWithAttachments,
  defaultConvertToLlm,
  setAppStorage,
} from "@mariozechner/pi-web-ui";
import { createElement, DetailedHTMLProps, HTMLAttributes, useEffect, useRef, useState } from "react";

import "./app.css";

function setupStorage(options: { provider?: CustomProvider; settings?: Record<string, any> }) {
  const settings = new SettingsStore();
  const providerKeys = new ProviderKeysStore();
  const sessions = new SessionsStore();
  const customProvider = new CustomProvidersStore();

  const backend = new IndexedDBStorageBackend({
    dbName: "glass-beaker-pi",
    version: 1,
    stores: [
      settings.getConfig(),
      providerKeys.getConfig(),
      sessions.getConfig(),
      customProvider.getConfig(),
      SessionsStore.getMetadataConfig(),
    ],
  });

  settings.setBackend(backend);
  providerKeys.setBackend(backend);
  sessions.setBackend(backend);
  customProvider.setBackend(backend);

  if (options.settings) {
    for (const key in options.settings) {
      settings.set(key, options.settings[key]);
    }
  }

  if (options.provider) {
    customProvider.set(options.provider);
  }

  const storage = new AppStorage(settings, providerKeys, sessions, customProvider, backend);
  setAppStorage(storage);

  return storage;
}

function classNames(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

function createEmptyUsage() {
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

function createUserMessage(input: string, attachments: Attachment[]): AgentMessage {
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

function createTerminalAssistantMessage(
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

async function readRouteError(response: Response) {
  try {
    const data = await response.json();
    if (typeof data?.error === "string" && data.error) {
      return data.error;
    }
  } catch {
  }

  return `Pi route error: ${response.status} ${response.statusText}`;
}

function isSameMessage(left: AgentMessage | undefined, right: AgentMessage | undefined) {
  if (!left || !right) {
    return false;
  }

  return JSON.stringify(left) === JSON.stringify(right);
}

type DivProps = DetailedHTMLProps<HTMLAttributes<HTMLDivElement>, HTMLDivElement>;
type AgentEvent =
  | { type: "agent_start" }
  | { type: "agent_end"; messages: AgentMessage[] }
  | { type: "turn_start" }
  | { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
  | { type: "message_start"; message: AgentMessage }
  | { type: "message_update"; message: AgentMessage }
  | { type: "message_end"; message: AgentMessage }
  | { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
  | { type: "tool_execution_update"; toolCallId: string; toolName: string; args: any; partialResult: any }
  | { type: "tool_execution_end"; toolCallId: string; toolName: string; result: any; isError: boolean };
type MessageListElement = HTMLElement & {
  isStreaming: boolean;
  messages: AgentMessage[];
  onCostClick?: () => void;
  pendingToolCalls?: Set<string>;
  tools: any[];
};
type MessageEditorElement = HTMLElement & {
  currentModel?: Model<any>;
  isStreaming: boolean;
  onAbort?: () => void;
  onModelSelect?: () => void;
  onSend?: (input: string, attachments: Attachment[]) => void;
  onThinkingChange?: (level: ThinkingLevel) => void;
  showAttachmentButton: boolean;
  showModelSelector: boolean;
  showThinkingSelector: boolean;
  thinkingLevel: ThinkingLevel;
};
type StreamingMessageContainerElement = HTMLElement & {
  isStreaming: boolean;
  onCostClick?: () => void;
  pendingToolCalls?: Set<string>;
  setMessage: (message: AgentMessage | null, immediate?: boolean) => void;
  toolResultsById?: Map<string, ToolResultMessage>;
  tools: any[];
};
type ToolResultMessage = Extract<AgentMessage, { role: "toolResult" }>;
type UserMessage = Extract<ReturnType<typeof defaultConvertToLlm>[number], { role: "user" }>;

function MessageListHost(props: { isStreaming: boolean; messages: AgentMessage[]; pendingToolCalls: Set<string> }) {
  const ref = useRef<MessageListElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    element.messages = props.messages;
    element.tools = [];
    element.pendingToolCalls = props.pendingToolCalls;
    element.isStreaming = props.isStreaming;
  }, [props.isStreaming, props.messages, props.pendingToolCalls]);

  return createElement("message-list", { ref });
}

function StreamingMessageHost(props: {
  isStreaming: boolean;
  message: AgentMessage | null;
  pendingToolCalls: Set<string>;
  toolResultsById: Map<string, ToolResultMessage>;
}) {
  const ref = useRef<StreamingMessageContainerElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    element.tools = [];
    element.pendingToolCalls = props.pendingToolCalls;
    element.toolResultsById = props.toolResultsById;
    element.isStreaming = props.isStreaming;
    element.setMessage(props.message, !props.isStreaming || props.message === null);
  }, [props.isStreaming, props.message, props.pendingToolCalls, props.toolResultsById]);

  return createElement("streaming-message-container", { ref });
}

function MessageEditorHost(props: {
  currentModel?: Model<any>;
  isStreaming: boolean;
  onAbort: () => void;
  onModelSelect: () => void;
  onSend: (input: string, attachments: Attachment[]) => void;
  onThinkingChange: (level: ThinkingLevel) => void;
  thinkingLevel: ThinkingLevel;
}) {
  const ref = useRef<MessageEditorElement | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }

    element.currentModel = props.currentModel;
    element.isStreaming = props.isStreaming;
    element.onAbort = props.onAbort;
    element.onModelSelect = props.onModelSelect;
    element.onSend = props.onSend;
    element.onThinkingChange = props.onThinkingChange;
    element.showAttachmentButton = true;
    element.showModelSelector = true;
    element.showThinkingSelector = true;
    element.thinkingLevel = props.thinkingLevel;
  }, [
    props.currentModel,
    props.isStreaming,
    props.onAbort,
    props.onModelSelect,
    props.onSend,
    props.onThinkingChange,
    props.thinkingLevel,
  ]);

  return createElement("message-editor", { ref });
}

export default function Pi(
  props: DivProps & {
    provider?: CustomProvider;
    settings?: Record<string, any>;
    systemPrompt?: string;
    thinkingLevel?: ThinkingLevel;
  },
) {
  const { className, provider, settings, style, systemPrompt, thinkingLevel, ...divProps } = props;

  const storageRef = useRef<AppStorage | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentModelRef = useRef<Model<any> | undefined>(provider?.models?.[0]);
  const isStreamingRef = useRef(false);
  const messagesRef = useRef<AgentMessage[]>([]);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const streamingMessageRef = useRef<AgentMessage | null>(null);
  const thinkingLevelRef = useRef<ThinkingLevel>(thinkingLevel ?? "off");
  const autoScrollRef = useRef(true);

  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [streamingMessage, setStreamingMessage] = useState<AgentMessage | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingToolCalls, setPendingToolCalls] = useState<Set<string>>(new Set());
  const [currentModel, setCurrentModel] = useState<Model<any> | undefined>(provider?.models?.[0]);
  const [currentThinkingLevel, setCurrentThinkingLevel] = useState<ThinkingLevel>(thinkingLevel ?? "off");

  useEffect(() => {
    storageRef.current = setupStorage({ settings, provider });
  }, [provider, settings]);

  useEffect(() => {
    if (!provider) {
      return;
    }

    setCurrentModel((previousModel: Model<any> | undefined) => {
      if (previousModel?.provider === provider.id) {
        return previousModel;
      }

      return provider.models?.[0];
    });
  }, [provider]);

  useEffect(() => {
    if (!thinkingLevel) {
      return;
    }

    setCurrentThinkingLevel(thinkingLevel);
  }, [thinkingLevel]);

  useEffect(() => {
    currentModelRef.current = currentModel;
  }, [currentModel]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    streamingMessageRef.current = streamingMessage;
  }, [streamingMessage]);

  useEffect(() => {
    thinkingLevelRef.current = currentThinkingLevel;
  }, [currentThinkingLevel]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !autoScrollRef.current) {
      return;
    }

    element.scrollTop = element.scrollHeight;
  }, [isStreaming, messages, pendingToolCalls, streamingMessage]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const toolResultsById = new Map<string, ToolResultMessage>();
  for (const message of messages) {
    if (message.role === "toolResult") {
      toolResultsById.set(message.toolCallId, message);
    }
  }

  async function ensureStorage() {
    if (!storageRef.current) {
      storageRef.current = setupStorage({ settings, provider });
    }

    return storageRef.current;
  }

  function appendTerminalMessage(errorMessage: string, stopReason: "error" | "aborted") {
    const terminalMessage = createTerminalAssistantMessage(
      errorMessage,
      currentModelRef.current,
      stopReason,
      streamingMessageRef.current,
    );

    setStreamingMessage(null);
    setPendingToolCalls(new Set());
    setIsStreaming(false);
    setMessages((previousMessages) =>
      isSameMessage(previousMessages[previousMessages.length - 1], terminalMessage)
        ? previousMessages
        : [...previousMessages, terminalMessage],
    );
  }

  function handleAgentEvent(event: AgentEvent) {
    switch (event.type) {
      case "agent_start":
        setIsStreaming(true);
        return;
      case "message_start":
        if (event.message.role === "user") {
          return;
        }
        setStreamingMessage(event.message);
        return;
      case "message_update":
        setStreamingMessage(event.message);
        return;
      case "message_end":
        if (event.message.role === "user") {
          return;
        }
        setStreamingMessage(null);
        setMessages((previousMessages) => [...previousMessages, event.message]);
        return;
      case "tool_execution_start":
        setPendingToolCalls((previousToolCalls) => {
          const nextToolCalls = new Set(previousToolCalls);
          nextToolCalls.add(event.toolCallId);
          return nextToolCalls;
        });
        return;
      case "tool_execution_end":
        setPendingToolCalls((previousToolCalls) => {
          const nextToolCalls = new Set(previousToolCalls);
          nextToolCalls.delete(event.toolCallId);
          return nextToolCalls;
        });
        return;
      case "agent_end": {
        const terminalMessage = event.messages.find(
          (message) => message.role === "assistant" && (message.stopReason === "error" || message.stopReason === "aborted"),
        );

        setStreamingMessage(null);
        setPendingToolCalls(new Set());
        setIsStreaming(false);
        if (terminalMessage) {
          setMessages((previousMessages) =>
            isSameMessage(previousMessages[previousMessages.length - 1], terminalMessage)
              ? previousMessages
              : [...previousMessages, terminalMessage],
          );
        }
      }
    }
  }

  async function consumeAgentStream(response: Response, controller: AbortController) {
    if (!response.ok) {
      appendTerminalMessage(await readRouteError(response), "error");
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      appendTerminalMessage("Pi route returned an empty stream.", "error");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) {
            continue;
          }

          const event = JSON.parse(line.slice(6).trim()) as AgentEvent;
          handleAgentEvent(event);
          if (event.type === "agent_end") {
            finished = true;
          }
        }
      }

      if (!finished) {
        appendTerminalMessage(controller.signal.aborted ? "Request aborted" : "Pi stream ended unexpectedly.", controller.signal.aborted ? "aborted" : "error");
      }
    } catch (error) {
      appendTerminalMessage(
        controller.signal.aborted ? "Request aborted" : error instanceof Error ? error.message : String(error),
        controller.signal.aborted ? "aborted" : "error",
      );
    }
  }

  async function handleSend(input: string, attachments: Attachment[]) {
    if (isStreamingRef.current) {
      return;
    }

    const model = currentModelRef.current;
    if (!model) {
      appendTerminalMessage("No model configured.", "error");
      return;
    }

    const storage = await ensureStorage();
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

    autoScrollRef.current = true;
    abortControllerRef.current = controller;
    setMessages(nextMessages);
    setStreamingMessage(null);
    setPendingToolCalls(new Set());
    setIsStreaming(true);

    try {
      const response = await fetch("/api/pi", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model,
          context: {
            systemPrompt: systemPrompt || "You are a helpful assistant.",
            messages: defaultConvertToLlm(nextMessages),
          },
          options: {
            apiKey,
            reasoning: thinkingLevelRef.current,
          },
        }),
        signal: controller.signal,
      });

      await consumeAgentStream(response, controller);
    } catch (error) {
      appendTerminalMessage(
        controller.signal.aborted ? "Request aborted" : error instanceof Error ? error.message : String(error),
        controller.signal.aborted ? "aborted" : "error",
      );
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  }

  function handleAbort() {
    abortControllerRef.current?.abort();
  }

  function handleScroll() {
    const element = scrollRef.current;
    if (!element) {
      return;
    }

    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    autoScrollRef.current = distanceFromBottom < 48;
  }

  function handleModelSelect() {
    if (!currentModelRef.current) {
      return;
    }

    ModelSelector.open(
      currentModelRef.current,
      (nextModel) => {
        setCurrentModel(nextModel);
      },
      provider ? [provider.id] : undefined,
    );
  }

  return (
    <div
      className={ classNames("flex h-full min-h-0 flex-col overflow-hidden bg-background text-foreground", className) }
      style={ style }
      { ...divProps }
    >
      <div ref={ scrollRef } className="flex-1 overflow-y-auto" onScroll={ handleScroll }>
        <div className="mx-auto flex max-w-4xl flex-col gap-3 px-4 py-4">
          <MessageListHost
            isStreaming={ isStreaming }
            messages={ messages }
            pendingToolCalls={ pendingToolCalls }
          />
          <StreamingMessageHost
            isStreaming={ isStreaming }
            message={ streamingMessage }
            pendingToolCalls={ pendingToolCalls }
            toolResultsById={ toolResultsById }
          />
        </div>
      </div>

      <div className="shrink-0 px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <MessageEditorHost
            currentModel={ currentModel }
            isStreaming={ isStreaming }
            onAbort={ handleAbort }
            onModelSelect={ handleModelSelect }
            onSend={ handleSend }
            onThinkingChange={ (level) => setCurrentThinkingLevel(level) }
            thinkingLevel={ currentThinkingLevel }
          />
        </div>
      </div>
    </div>
  );
}
