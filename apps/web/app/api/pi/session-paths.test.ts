import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import { createPiSessionPaths } from "./session-paths";

const SESSION_ID = "11111111-1111-4111-8111-111111111111";

test("createPiSessionPaths creates one directory namespace for a session", () => {
  const agentDir = path.join("workspace", ".pi", "agent");

  const paths = createPiSessionPaths(agentDir, SESSION_ID);

  assert.equal(paths.sessionId, SESSION_ID);
  assert.equal(paths.sessionDir, path.join(agentDir, "sessions", SESSION_ID));
  assert.equal(paths.attachmentsDir, path.join(agentDir, "sessions", SESSION_ID, "uploads"));
});

test("createPiSessionPaths replaces unsafe session ids", () => {
  const agentDir = path.join("workspace", ".pi", "agent");

  const paths = createPiSessionPaths(agentDir, "../escape");

  assert.match(paths.sessionId, /^[0-9a-f-]{36}$/);
  assert.notEqual(paths.sessionId, "../escape");
  assert.equal(path.dirname(paths.sessionDir), path.join(agentDir, "sessions"));
});
