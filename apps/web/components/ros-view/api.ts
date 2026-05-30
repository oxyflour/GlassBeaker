import type { RosTopic } from "./demo-data";

export interface RosViewState {
  connected: boolean;
  last_error: string | null;
  topics: RosTopic[];
}

export interface RosViewStateStream {
  addEventListener(type: string, listener: (event: MessageEvent<string>) => void): void;
  close(): void;
  onerror: ((event: Event) => void) | null;
}

export type RosViewStreamFactory = (url: string) => RosViewStateStream;

export async function loadRosViewState(): Promise<RosViewState> {
  const response = await fetch("/python/ros_view/state", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`ROS view state request failed: ${response.status}`);
  }
  return await response.json() as RosViewState;
}

export function createRosViewStreamUrl(): string {
  return "/python/ros_view/stream";
}

function defaultRosViewStreamFactory(url: string): RosViewStateStream {
  return new EventSource(url);
}

export function subscribeRosViewState(
  onState: (state: RosViewState) => void,
  createEventSource: RosViewStreamFactory = defaultRosViewStreamFactory,
  onError?: (error: Error) => void,
): () => void {
  const source = createEventSource(createRosViewStreamUrl());
  source.addEventListener("state", (event) => {
    onState(JSON.parse(event.data) as RosViewState);
  });
  source.onerror = () => {
    onError?.(new Error("ROS view stream disconnected"));
  };
  return () => source.close();
}
