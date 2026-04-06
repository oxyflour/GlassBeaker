export const HOST_MESSAGE_SOURCE = "agent-preview-host";
export const FRAME_MESSAGE_SOURCE = "agent-preview-frame";

export type PreviewHostMessage = {
  props: unknown;
  source: typeof HOST_MESSAGE_SOURCE;
  type: "set-props";
};

export type PreviewFrameMessage = {
  error?: string;
  source: typeof FRAME_MESSAGE_SOURCE;
  type: "error" | "ready";
};

export function createPreviewHostMessage(props: unknown): PreviewHostMessage {
  return { props, source: HOST_MESSAGE_SOURCE, type: "set-props" };
}

export function isPreviewFrameMessage(value: unknown): value is PreviewFrameMessage {
  return !!value && typeof value === "object" && (value as PreviewFrameMessage).source === FRAME_MESSAGE_SOURCE;
}
