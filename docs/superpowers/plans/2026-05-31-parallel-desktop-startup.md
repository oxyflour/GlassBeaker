# Parallel Desktop Startup Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
**Goal:** Start the Electron-managed Python services and Next server concurrently so startup waits for the slower side instead of Python then Next.
**Architecture:** `apps/desktop/src/main.cjs` will spawn Python, ROS, and Hermes, then immediately fork Next with stable connection env. `apps/web/app/api/copilotkit/route.ts` will lazily fetch Python `/runtime` on first CopilotKit request and cache the resulting `CopilotRuntime`, removing the process-launch `API_RUNTIME` dependency.
**Tech Stack:** Electron main process CommonJS, Next.js App Router route handlers, CopilotKit runtime, Node test runner with `tsx`.
---

## File Structure
- Modify `apps/desktop/src/main.cjs`: reorder `startServer()` so Next forks before waiting for Python `/runtime`.
- Modify `apps/web/app/api/copilotkit/route.ts`: replace module-load `API_RUNTIME` parsing with request-time runtime creation.
- Modify `apps/web/components/agent/pi-provider.test.ts`: add source-level startup contract assertions.
No Python files are needed. `apps/python/app.py` already exposes `/runtime`.

### Task 1: Add Failing Startup Contract Tests
**Files:**
- Modify: `apps/web/components/agent/pi-provider.test.ts`
- [ ] **Step 1: Append tests**
```ts
test("desktop starts Next before waiting for the Python runtime", async () => {
  const source = await readFile(new URL("../../../desktop/src/main.cjs", import.meta.url), "utf8");
  const nextForkIndex = source.indexOf("utilityProcess.fork");
  const runtimeWaitIndex = source.indexOf("assertUrl(`http://127.0.0.1:${pythonPort}/runtime`)");

  assert.notEqual(nextForkIndex, -1);
  assert.notEqual(runtimeWaitIndex, -1);
  assert.ok(nextForkIndex < runtimeWaitIndex);
  assert.match(source, /Promise\.all\(\[/);
  assert.doesNotMatch(source, /API_RUNTIME:\s*apiRuntime/);
});

test("CopilotKit route loads the Python runtime after Next has started", async () => {
  const source = await readFile(new URL("../../app/api/copilotkit/route.ts", import.meta.url), "utf8");

  assert.match(source, /async function getRuntime/);
  assert.match(source, /fetch\(new URL\(["']runtime["'],/);
  assert.match(source, /status:\s*503/);
  assert.doesNotMatch(source, /JSON\.parse\(process\.env\.API_RUNTIME/);
});
```
- [ ] **Step 2: Verify failing behavior**
```powershell
pnpm --dir apps/web exec tsx --test components/agent/pi-provider.test.ts
```
Expected: FAIL. `main.cjs` still waits for `/runtime` before `utilityProcess.fork`, and `route.ts` still parses `process.env.API_RUNTIME` at module load.

### Task 2: Fork Next Before Python Runtime Readiness
**Files:**
- Modify: `apps/desktop/src/main.cjs`
- [ ] **Step 1: Replace the sequential wait/fork block**
Replace the block beginning with `const apiRuntime = await assertUrl` and ending with `return url` with:
```js
    const url = `http://localhost:${nextJsPort}`,
        nextjs = utilityProcess.fork(path.join(root, 'web/node_modules/next/dist/bin/next'), [
            '-p', `${nextJsPort}`
        ], {
            env: {
                ...process.env,
                API_REWRITE: `http://127.0.0.1:${pythonPort}/`,
                GLASSBEAKER_HERMES_PORT: `${hermesPort}`,
            },
            cwd: path.join(root, 'web'),
            stdio: "pipe"
        })
    // @ts-ignore
    watchProc('nextjs', nextjs)

    const [apiRuntime] = await Promise.all([
        assertUrl(`http://127.0.0.1:${pythonPort}/runtime`),
        assertUrl(url),
    ])
    console.log(`[main] RUNTIME: ${apiRuntime}`)
    return url
```
- [ ] **Step 2: Verify partial progress**
```powershell
pnpm --dir apps/web exec tsx --test components/agent/pi-provider.test.ts
```
Expected: FAIL. The desktop startup test passes; the CopilotKit lazy-runtime test still fails.

### Task 3: Create CopilotKit Runtime Lazily
**Files:**
- Modify: `apps/web/app/api/copilotkit/route.ts`
- [ ] **Step 1: Replace static `agents` and `runtime` construction**
Replace the current `const agents = ...` through `const runtime = ...` block with:
```ts
type PythonRuntime = {
  agents?: Array<{ path?: string; name?: string }>;
};

let runtimePromise: Promise<CopilotRuntime> | null = null;

async function fetchPythonRuntime(): Promise<PythonRuntime> {
  const response = await fetch(new URL("runtime", process.env.API_REWRITE || "http://localhost:13001/"));
  if (!response.ok) {
    throw new Error(`Python runtime returned ${response.status}: ${await response.text()}`);
  }
  return await response.json() as PythonRuntime;
}

function createRuntime(apiRuntime: PythonRuntime): CopilotRuntime {
  const agents = {} as Record<string, AbstractAgent>;
  for (const { path, name } of apiRuntime.agents || []) {
    if (path && name) {
      agents[name] = new LangGraphHttpAgent({
        url: `${process.env.API_REWRITE || "http://localhost:13001/"}${path.slice(1)}`,
      });
    }
  }
  agents.default = Object.values(agents)[0] || builtin;
  agents.builtin = builtin;
  return new CopilotRuntime({ agents });
}

async function getRuntime(): Promise<CopilotRuntime> {
  runtimePromise ??= fetchPythonRuntime()
    .then(createRuntime)
    .catch((error) => {
      runtimePromise = null;
      throw error;
    });
  return await runtimePromise;
}
```
- [ ] **Step 2: Use `getRuntime()` in `POST`**
Replace the existing endpoint setup in `POST` with:
```ts
  let runtime: CopilotRuntime;
  try {
    runtime = await getRuntime();
  } catch (error) {
    return Response.json(
      { error: `Python runtime is not ready: ${error instanceof Error ? error.message : String(error)}` },
      { status: 503 }
    );
  }

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    endpoint: request.nextUrl.pathname
  });
```
- [ ] **Step 3: Verify focused test**
```powershell
pnpm --dir apps/web exec tsx --test components/agent/pi-provider.test.ts
```
Expected: PASS.

### Task 4: Build and Diff Review
**Files:**
- Verify: `apps/desktop/src/main.cjs`
- Verify: `apps/web/app/api/copilotkit/route.ts`
- Verify: `apps/web/components/agent/pi-provider.test.ts`
- [ ] **Step 1: Run the web build**
```powershell
pnpm --filter glassbeaker-web build
```
Expected: PASS.
- [ ] **Step 2: Review only intended files**
```powershell
git diff -- apps/desktop/src/main.cjs apps/web/app/api/copilotkit/route.ts apps/web/components/agent/pi-provider.test.ts
```
Expected: Diff shows the parallel startup reorder, lazy CopilotKit runtime creation, and startup assertions only.

## Self-Review
Coverage: Tasks 2 and 3 remove the sequential startup dependency while preserving Python agent discovery. Placeholder scan passed. Type consistency is covered by defining `PythonRuntime`, `runtimePromise`, `fetchPythonRuntime`, `createRuntime`, and `getRuntime` before use.
