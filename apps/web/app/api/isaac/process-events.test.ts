import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { attachIsaacProcessLifecycle, type IsaacManagedProcess } from "./process-events";

class FakeChild extends EventEmitter {
  constructor(public pid: number | undefined) {
    super();
  }
}

function createEntry(pid?: number) {
  const resolvedPid = arguments.length === 0 ? 42 : pid;
  const child = new FakeChild(resolvedPid);
  const entry: IsaacManagedProcess = {
    child,
    exitCode: null,
    id: "renderer-1",
    logPath: "C:/tmp/renderer.log",
    stopping: false,
  };
  return { child, entry };
}

test("attachIsaacProcessLifecycle logs unexpected exits with the renderer log path", () => {
  const calls: string[] = [];
  const { child, entry } = createEntry();
  attachIsaacProcessLifecycle(
    { error: (message) => calls.push(message), log: () => undefined },
    entry,
  );

  child.emit("exit", 7);

  assert.equal(entry.exitCode, 7);
  assert.deepEqual(calls, [
    "IsaacSim quit unexpectedly, check C:/tmp/renderer.log (id=renderer-1 pid=42 exitCode=7)",
  ]);
});

test("attachIsaacProcessLifecycle logs spawn failures and records the synthetic exit code", () => {
  const calls: string[] = [];
  const { child, entry } = createEntry(undefined);
  attachIsaacProcessLifecycle(
    { error: (message) => calls.push(message), log: () => undefined },
    entry,
  );

  child.emit("error", new Error("spawn EPERM"));

  assert.equal(entry.exitCode, -1);
  assert.deepEqual(calls, [
    "IsaacSim failed to launch, check C:/tmp/renderer.log (id=renderer-1 pid=n/a error=Error: spawn EPERM)",
  ]);
});

test("attachIsaacProcessLifecycle treats requested shutdowns as stopped instead of crashes", () => {
  const calls: string[] = [];
  const { child, entry } = createEntry();
  entry.stopping = true;
  attachIsaacProcessLifecycle(
    { error: () => undefined, log: (message) => calls.push(message) },
    entry,
  );

  child.emit("exit", 1);

  assert.equal(entry.exitCode, 1);
  assert.deepEqual(calls, [
    "IsaacSim stopped, check C:/tmp/renderer.log (id=renderer-1 pid=42 exitCode=1)",
  ]);
});
