import { spawn, type ChildProcess, type SpawnOptions } from "node:child_process";
import { createWriteStream, existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";

type StartBody = {
  id?: string;
  cmd?: unknown;
  cwd?: string;
  env?: Record<string, string>;
  logPath?: string;
};

type ManagedProcess = {
  child: ChildProcess;
  exitCode: number | null;
  logPath: string;
};

const workspaceRoot = resolveWorkspaceDir();
const globalState = globalThis as typeof globalThis & { __glassbeakerIsaac?: Map<string, ManagedProcess> };
const processes = globalState.__glassbeakerIsaac ?? (globalState.__glassbeakerIsaac = new Map<string, ManagedProcess>());

function resolveWorkspaceDir() {
  let current = process.cwd();
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
  return process.cwd();
}

function isInsideWorkspace(target: string) {
  const resolved = path.resolve(target);
  const relative = path.relative(workspaceRoot, resolved);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function waitForClose(child: ChildProcess, timeoutMs: number) {
  if (child.exitCode !== null) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    timer.unref();
    child.once("close", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function stopProcess(id: string, entry: ManagedProcess) {
  if (entry.exitCode === null) {
    if (process.platform === "win32" && entry.child.pid) {
      await new Promise<void>((resolve) => {
        const killer = spawn("taskkill", ["/PID", String(entry.child.pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
        killer.once("close", () => resolve());
        killer.once("error", () => {
          try {
            entry.child.kill();
          } catch {}
          resolve();
        });
      });
    } else {
      try {
        entry.child.kill("SIGTERM");
      } catch {}
    }
    await waitForClose(entry.child, 5000);
  }
  processes.delete(id);
}

export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id")?.trim();
  if (!id) {
    return Response.json({ error: "Missing id." }, { status: 400 });
  }
  const entry = processes.get(id);
  const running = entry?.exitCode === null;
  return Response.json({
    id,
    pid: entry?.child.pid ?? null,
    running: Boolean(running),
    exitCode: entry?.exitCode ?? null,
    logPath: entry?.logPath ?? null,
  });
}

export async function POST(request: Request) {
  let body: StartBody;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const id = body.id?.trim();
  const cwd = body.cwd?.trim();
  const logPath = body.logPath?.trim();
  const cmd = Array.isArray(body.cmd) ? body.cmd.filter((value): value is string => typeof value === "string" && value.length > 0) : [];
  if (!id || !cwd || !logPath || cmd.length === 0 || !body.env || typeof body.env !== "object") {
    return Response.json({ error: "Invalid Isaac launch payload." }, { status: 400 });
  }
  if (!isInsideWorkspace(cwd) || !isInsideWorkspace(logPath)) {
    return Response.json({ error: "Isaac launch path must stay inside workspace." }, { status: 400 });
  }

  const existing = processes.get(id);
  if (existing) {
    await stopProcess(id, existing);
  }

  await mkdir(path.dirname(logPath), { recursive: true });
  const logStream = createWriteStream(logPath, { flags: "w" });
  const options: SpawnOptions = {
    cwd,
    // @ts-ignore
    env: Object.fromEntries(Object.entries(body.env).map(([key, value]) => [key, String(value)])),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  };
  const child = spawn(cmd[0], cmd.slice(1), options);
  const entry: ManagedProcess = { child, exitCode: null, logPath };
  child.stdout?.pipe(logStream, { end: false });
  child.stderr?.pipe(logStream, { end: false });
  child.once("exit", (code) => {
    entry.exitCode = code ?? -1;
    logStream.end();
  });
  child.once("error", (error) => {
    entry.exitCode = -1;
    logStream.write(`${String(error)}\n`);
    logStream.end();
  });
  processes.set(id, entry);

  return Response.json({ id, pid: child.pid ?? null, running: true, logPath });
}

export async function DELETE(request: Request) {
  let id = new URL(request.url).searchParams.get("id")?.trim();
  if (!id) {
    try {
      const body = await request.json();
      id = typeof body?.id === "string" ? body.id.trim() : "";
    } catch {}
  }
  if (!id) {
    return Response.json({ error: "Missing id." }, { status: 400 });
  }
  const entry = processes.get(id);
  if (entry) {
    await stopProcess(id, entry);
  }
  return Response.json({ id, running: false });
}
