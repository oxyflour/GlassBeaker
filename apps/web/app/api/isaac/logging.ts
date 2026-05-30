export type IsaacProcessEvent = "launch-error" | "quit" | "stopped";

export type IsaacProcessEventMeta = {
  id: string;
  pid: number | null;
  logPath: string;
  exitCode?: number | null;
  error?: string;
};

type IsaacLogger = Pick<Console, "error" | "log">;

function formatPid(pid: number | null) {
  return pid === null ? "n/a" : String(pid);
}

export function formatIsaacProcessMessage(
  event: IsaacProcessEvent,
  meta: IsaacProcessEventMeta,
) {
  const fields = [`id=${meta.id}`, `pid=${formatPid(meta.pid)}`];
  if (meta.exitCode !== undefined && meta.exitCode !== null) {
    fields.push(`exitCode=${meta.exitCode}`);
  }
  if (meta.error) {
    fields.push(`error=${meta.error}`);
  }
  const suffix = ` (${fields.join(" ")})`;
  if (event === "launch-error") {
    return `IsaacSim failed to launch, check ${meta.logPath}${suffix}`;
  }
  if (event === "quit") {
    return `IsaacSim quit unexpectedly, check ${meta.logPath}${suffix}`;
  }
  return `IsaacSim stopped, check ${meta.logPath}${suffix}`;
}

export function logIsaacProcessEvent(
  logger: IsaacLogger,
  event: IsaacProcessEvent,
  meta: IsaacProcessEventMeta,
) {
  const message = formatIsaacProcessMessage(event, meta);
  if (event === "stopped") {
    logger.log(message);
    return;
  }
  logger.error(message);
}
