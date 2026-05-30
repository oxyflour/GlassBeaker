import type { ChildProcess } from "node:child_process";

import { logIsaacProcessEvent } from "./logging";

type IsaacLogger = Pick<Console, "error" | "log">;

type ProcessChild = {
  pid?: ChildProcess["pid"];
  once(event: "error", listener: (error: Error) => void): unknown;
  once(event: "exit", listener: (code: number | null) => void): unknown;
};

export type IsaacManagedProcess = {
  child: ProcessChild;
  exitCode: number | null;
  id: string;
  logPath: string;
  stopping: boolean;
};

export function attachIsaacProcessLifecycle(
  logger: IsaacLogger,
  entry: IsaacManagedProcess,
  callbacks?: {
    onError?: (message: string) => void;
    onSettled?: () => void;
  },
) {
  let settled = false;

  const settle = (handler: () => void) => {
    if (settled) {
      return false;
    }
    settled = true;
    handler();
    callbacks?.onSettled?.();
    return true;
  };

  entry.child.once("error", (error) => {
    settle(() => {
      entry.exitCode = -1;
      const message = String(error);
      callbacks?.onError?.(message);
      logIsaacProcessEvent(logger, "launch-error", {
        id: entry.id,
        pid: entry.child.pid ?? null,
        logPath: entry.logPath,
        error: message,
      });
    });
  });

  entry.child.once("exit", (code) => {
    settle(() => {
      entry.exitCode = code ?? -1;
      logIsaacProcessEvent(logger, entry.stopping ? "stopped" : "quit", {
        id: entry.id,
        pid: entry.child.pid ?? null,
        logPath: entry.logPath,
        exitCode: entry.exitCode,
      });
    });
  });
}
