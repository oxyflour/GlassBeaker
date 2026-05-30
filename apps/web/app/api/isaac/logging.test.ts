import assert from "node:assert/strict";
import test from "node:test";

type LoggingModule = typeof import("./logging");

test("logIsaacProcessEvent writes an unexpected quit message with the log path", async () => {
  const { logIsaacProcessEvent } = await loadModule<LoggingModule>("./logging.ts");
  const calls: string[] = [];
  const logger = {
    log: (_message: string) => undefined,
    error: (message: string) => {
      calls.push(message);
    },
  };

  logIsaacProcessEvent(logger, "quit", {
    id: "renderer-1",
    pid: 42,
    exitCode: 7,
    logPath: "C:/tmp/renderer.log",
  });

  assert.deepEqual(calls, [
    "IsaacSim quit unexpectedly, check C:/tmp/renderer.log (id=renderer-1 pid=42 exitCode=7)",
  ]);
});

test("logIsaacProcessEvent writes launch errors with the log path and error text", async () => {
  const { logIsaacProcessEvent } = await loadModule<LoggingModule>("./logging.ts");
  const calls: string[] = [];
  const logger = {
    log: (_message: string) => undefined,
    error: (message: string) => {
      calls.push(message);
    },
  };

  logIsaacProcessEvent(logger, "launch-error", {
    id: "renderer-2",
    pid: null,
    logPath: "C:/tmp/renderer.log",
    error: "spawn EPERM",
  });

  assert.deepEqual(calls, [
    "IsaacSim failed to launch, check C:/tmp/renderer.log (id=renderer-2 pid=n/a error=spawn EPERM)",
  ]);
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  const namespace = (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
  return namespace;
}
