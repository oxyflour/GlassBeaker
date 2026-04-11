'use client'

import "@mariozechner/pi-web-ui";

import type {
  CustomProvider,
  ThinkingLevel,
} from "@mariozechner/pi-web-ui";
import type { DetailedHTMLProps, HTMLAttributes } from "react";

import "./app.css";

import { usePiAutoScroll } from "./pi/auto-scroll";
import { usePiFrontendToolsBridge } from "./pi/frontend-tools";
import { MessageEditorHost, MessageListHost, StreamingMessageHost } from "./pi/hosts";
import { usePiModelState } from "./pi/model-state";
import { usePiSession } from "./pi/session";
import { usePiStorage } from "./pi/storage";
import { usePiToolCallCollapse } from "./pi/tool-call-collapse";
import { classNames } from "./pi/utils";

const baseUrl = 'http://localhost:13000/cors/moonshot/v1'
const PROVIDER = {
    id: 'moonshot',
    name: 'moonshot',
    baseUrl,
    type: 'openai-completions',
    models: [{
        id: 'kimi-k2.5',
        name: 'Kimi K2.5',
        api: 'openai-completions',
        provider: 'moonshot',
        baseUrl,
        reasoning: false,
        input: ['text'],
        contextWindow: 131072,
        maxTokens: 32000,
        cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
        }
    }]
} satisfies CustomProvider

type DivProps = DetailedHTMLProps<HTMLAttributes<HTMLDivElement>, HTMLDivElement>;
type PiProps = DivProps & {
  provider?: CustomProvider;
  settings?: Record<string, any>;
  systemPrompt?: string;
  thinkingLevel?: ThinkingLevel;
};

export default function Pi(props: PiProps) {
  const { className, provider = PROVIDER, settings, style, systemPrompt, thinkingLevel, ...divProps } = props;
  const { ensureStorage } = usePiStorage({ provider, settings });
  const frontendTools = usePiFrontendToolsBridge();
  const modelState = usePiModelState({ provider, thinkingLevel });
  const session = usePiSession({
    currentModel: modelState.currentModel,
    currentThinkingLevel: modelState.currentThinkingLevel,
    ensureStorage,
    executeFrontendTool: frontendTools?.executeTool,
    frontendTools: frontendTools?.definitions,
    systemPrompt,
  });
  const { onScroll, scrollRef } = usePiAutoScroll({
    isStreaming: session.isStreaming,
    messages: session.messages,
    pendingToolCalls: session.pendingToolCalls,
    streamingMessage: session.streamingMessage,
  });
  usePiToolCallCollapse(scrollRef);

  return (
    <div
      className={ classNames("flex h-full min-h-0 flex-col overflow-hidden bg-background text-foreground", className) }
      style={ style }
      { ...divProps }
    >
      <div ref={ scrollRef } className="flex-1 overflow-y-auto" onScroll={ onScroll }>
        <div className="mx-auto flex max-w-4xl flex-col gap-3 px-4 py-4">
          <MessageListHost
            isStreaming={ session.isStreaming }
            messages={ session.messages }
            pendingToolCalls={ session.pendingToolCalls }
          />
          <StreamingMessageHost
            isStreaming={ session.isStreaming }
            message={ session.streamingMessage }
            pendingToolCalls={ session.pendingToolCalls }
            toolResultsById={ session.toolResultsById }
          />
        </div>
      </div>

      <div className="shrink-0 px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <MessageEditorHost
            currentModel={ modelState.currentModel }
            isStreaming={ session.isStreaming }
            onAbort={ session.abort }
            onModelSelect={ modelState.openModelSelector }
            onSend={ session.send }
            onThinkingChange={ modelState.setCurrentThinkingLevel }
            thinkingLevel={ modelState.currentThinkingLevel }
          />
        </div>
      </div>
    </div>
  );
}

export { PiFrontendToolProvider, usePiFrontendTool } from "./pi/frontend-tools";
