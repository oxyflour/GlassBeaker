import type { AgentEvent } from "./types";
import { readRouteError } from "./utils";

type StreamOptions = {
  controller: AbortController;
  onEvent: (event: AgentEvent) => void;
  onTerminalFailure: (message: string, stopReason: "error" | "aborted") => void;
  response: Response;
};

export async function consumeAgentStream(options: StreamOptions) {
  if (!options.response.ok) {
    options.onTerminalFailure(await readRouteError(options.response), "error");
    return;
  }

  const reader = options.response.body?.getReader();
  if (!reader) {
    options.onTerminalFailure("Pi route returned an empty stream.", "error");
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
        options.onEvent(event);
        if (event.type === "agent_end") {
          finished = true;
        }
      }
    }

    if (!finished) {
      const aborted = options.controller.signal.aborted;
      options.onTerminalFailure(aborted ? "Request aborted" : "Pi stream ended unexpectedly.", aborted ? "aborted" : "error");
    }
  } catch (error) {
    const aborted = options.controller.signal.aborted;
    const message = aborted ? "Request aborted" : error instanceof Error ? error.message : String(error);
    options.onTerminalFailure(message, aborted ? "aborted" : "error");
  }
}
