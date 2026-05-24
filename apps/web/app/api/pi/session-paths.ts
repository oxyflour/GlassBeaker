import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";

const PI_SESSION_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function createPiSessionPaths(agentDir: string, sessionId: unknown) {
  const safeSessionId = typeof sessionId === "string" && PI_SESSION_ID_PATTERN.test(sessionId) ? sessionId.toLowerCase() : randomUUID();
  const sessionDir = path.join(agentDir, "sessions", safeSessionId);
  return {
    attachmentsDir: path.join(sessionDir, "uploads"),
    sessionDir,
    sessionId: safeSessionId,
  };
}

export function resolveWorkspaceDir(startDir = process.cwd()) {
  let current = startDir;

  for (let i = 0; i < 6; i += 1) {
    if (existsSync(path.join(current, "pnpm-workspace.yaml"))) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }

  return startDir;
}
