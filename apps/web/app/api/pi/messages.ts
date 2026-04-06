import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import type { PiRequestMessage, PiSerializableAttachment } from "../../../components/agent/pi/protocol";

type HydratedAttachment = {
  extractedPath?: string;
  originalPath: string;
};

type PiContentBlock = Record<string, any> & { type: string };
type PiMessage = Record<string, any> & { role: "assistant" | "toolResult" | "user" };

function normalizePath(filePath: string, workspaceDir: string) {
  return path.relative(workspaceDir, filePath).replace(/\\/g, "/");
}

function sanitizeFileName(fileName: string) {
  return fileName.replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_");
}

function attachmentDirectory(cacheDir: string, attachment: PiSerializableAttachment) {
  const hash = createHash("sha1").update(attachment.id).digest("hex").slice(0, 12);
  return path.join(cacheDir, hash);
}

async function hydrateAttachment(
  attachment: PiSerializableAttachment,
  cacheDir: string,
  workspaceDir: string,
  cache: Map<string, HydratedAttachment>,
) {
  const cached = cache.get(attachment.id);
  if (cached) {
    return cached;
  }

  const directory = attachmentDirectory(cacheDir, attachment);
  const fileName = sanitizeFileName(attachment.fileName || attachment.id);
  const originalPath = path.join(directory, fileName);
  const extractedPath = attachment.extractedText ? `${originalPath}.extracted.txt` : undefined;

  await mkdir(directory, { recursive: true });
  if (!existsSync(originalPath)) {
    await writeFile(originalPath, Buffer.from(attachment.content, "base64"));
  }
  if (extractedPath && !existsSync(extractedPath)) {
    await writeFile(extractedPath, attachment.extractedText || "", "utf8");
  }

  const hydrated = {
    extractedPath: extractedPath ? normalizePath(extractedPath, workspaceDir) : undefined,
    originalPath: normalizePath(originalPath, workspaceDir),
  };
  cache.set(attachment.id, hydrated);
  return hydrated;
}

async function createAttachmentNote(
  attachments: PiSerializableAttachment[],
  cacheDir: string,
  workspaceDir: string,
  cache: Map<string, HydratedAttachment>,
) {
  const lines = ["Uploaded files for this message:"];
  for (const attachment of attachments) {
    const hydrated = await hydrateAttachment(attachment, cacheDir, workspaceDir, cache);
    const preferredPath = hydrated.extractedPath || hydrated.originalPath;
    lines.push(`- ${attachment.fileName}: read ${preferredPath}`);
    if (hydrated.extractedPath) {
      lines.push(`  original file: ${hydrated.originalPath}`);
    }
  }
  lines.push("Use the read tool on the listed path when you need file contents.");
  return lines.join("\n");
}

function toTextContent(content: unknown) {
  if (typeof content === "string") {
    return [{ type: "text", text: content } satisfies PiContentBlock];
  }
  if (Array.isArray(content)) {
    return content.filter((item) => item && typeof item === "object" && "type" in item) as PiContentBlock[];
  }
  return [];
}

async function convertUserWithAttachments(
  message: PiRequestMessage,
  cacheDir: string,
  workspaceDir: string,
  cache: Map<string, HydratedAttachment>,
) {
  const attachments = message.attachments || [];
  const content = [...toTextContent(message.content)];

  if (attachments.length > 0) {
    content.push({ type: "text", text: await createAttachmentNote(attachments, cacheDir, workspaceDir, cache) });
  }

  for (const attachment of attachments) {
    if (attachment.type === "image") {
      content.push({ type: "image", data: attachment.content, mimeType: attachment.mimeType });
    }
  }

  return { content, role: "user" as const, timestamp: message.timestamp };
}

export async function convertPiMessages(messages: PiRequestMessage[], workspaceDir: string, cacheDir: string) {
  const cache = new Map<string, HydratedAttachment>();
  const converted: PiMessage[] = [];

  for (const message of messages) {
    if (message.role === "artifact") {
      continue;
    }
    if (message.role === "user-with-attachments") {
      converted.push(await convertUserWithAttachments(message, cacheDir, workspaceDir, cache));
      continue;
    }
    if (message.role === "user" || message.role === "assistant" || message.role === "toolResult") {
      converted.push(message as PiMessage);
    }
  }

  return converted;
}
