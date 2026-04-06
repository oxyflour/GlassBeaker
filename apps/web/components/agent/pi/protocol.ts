export type PiFrontendToolParameterType =
  | "string"
  | "number"
  | "boolean"
  | "object"
  | "array";

export type PiFrontendToolParameter = {
  name: string;
  type: PiFrontendToolParameterType;
  description?: string;
  required?: boolean;
};

export type PiFrontendToolDefinition = {
  name: string;
  description: string;
  parameters: PiFrontendToolParameter[];
  followUp?: boolean;
  available?: "disabled" | "enabled";
};

export type PiSerializableAttachment = {
  id: string;
  type: "image" | "document";
  fileName: string;
  mimeType: string;
  size: number;
  content: string;
  extractedText?: string;
};

export type PiRequestMessage = Record<string, any> & {
  role: string;
  timestamp?: number;
  attachments?: PiSerializableAttachment[];
};

export type PiFrontendToolRequestEvent = {
  type: "frontend_tool_request";
  requestId: string;
  toolCallId: string;
  toolName: string;
  args: unknown;
};

export type PiFrontendToolResultPayload = {
  requestId: string;
  toolCallId: string;
  result?: unknown;
  error?: string;
};
