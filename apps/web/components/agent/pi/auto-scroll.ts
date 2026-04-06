import type { AgentMessage } from "@mariozechner/pi-web-ui";
import { useEffect, useRef } from "react";

type AutoScrollOptions = {
  isStreaming: boolean;
  messages: AgentMessage[];
  pendingToolCalls: Set<string>;
  streamingMessage: AgentMessage | null;
};

export function usePiAutoScroll(options: AutoScrollOptions) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const autoScrollRef = useRef(true);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !autoScrollRef.current) {
      return;
    }

    element.scrollTop = element.scrollHeight;
  }, [options.isStreaming, options.messages, options.pendingToolCalls, options.streamingMessage]);

  function onScroll() {
    const element = scrollRef.current;
    if (!element) {
      return;
    }

    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    autoScrollRef.current = distanceFromBottom < 48;
  }

  return { onScroll, scrollRef };
}
