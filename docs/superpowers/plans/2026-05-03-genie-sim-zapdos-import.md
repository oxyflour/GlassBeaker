# Genie Sim Zapdos Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click handoff from generated `GenieSim` `scene.usda` files into `Zapdos`, and make `Zapdos` bootstrap the correct backend session for each imported scene.

**Architecture:** Keep the handoff path-based. `GenieSim` emits `sceneUsdaPath`, the web page builds a `/demo/zapdos?scene_usd=...` URL, and `Zapdos` derives both its init URL and local-storage session key from those query params. The backend keeps the existing `scene_usd` contract, but it must evict failed futures and emit readable `init` errors.

**Tech Stack:** Next.js App Router, React, FastAPI, `unittest`, `pnpm exec tsx --test`

---

## File Structure

- Create: `apps/web/components/zapdos/zapdos-import.ts`
- Create: `apps/web/components/zapdos/zapdos-import.test.ts`
- Create: `apps/web/components/genie-sim/OpenInZapdosLink.tsx`
- Create: `apps/web/components/genie-sim/open-in-zapdos-link.test.tsx`
- Create: `apps/python/tests/test_zapdos_import.py`
- Modify: `apps/web/app/demo/zapdos/page.tsx`
- Modify: `apps/web/app/demo/agent-genie-sim/page.tsx`
- Modify: `apps/web/components/genie-sim/index.ts`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`

### Task 1: Add Zapdos import helpers and helper tests

**Files:**
- Create: `apps/web/components/zapdos/zapdos-import.ts`
- Create: `apps/web/components/zapdos/zapdos-import.test.ts`

- [ ] **Step 1: Write the failing helper tests**

```ts
import assert from "node:assert/strict";
import test from "node:test";

import {
  buildZapdosInitStreamUrl,
  buildZapdosSessionStorageKey,
  parseZapdosInitEvent,
} from "./zapdos-import";

test("buildZapdosInitStreamUrl includes encoded scene and robot params", () => {
  const url = buildZapdosInitStreamUrl("sess-1", "C:/tmp/a scene.usda", "deps/galaxea/object/r1pro/r1pro.usda");
  assert.equal(url, "/python/zapdos/sess-1/init/start?scene_usd=C%3A%2Ftmp%2Fa+scene.usda&robot_usd=deps%2Fgalaxea%2Fobject%2Fr1pro%2Fr1pro.usda");
});

test("buildZapdosSessionStorageKey changes when scene changes", () => {
  assert.notEqual(
    buildZapdosSessionStorageKey("C:/tmp/scene-a.usda", null),
    buildZapdosSessionStorageKey("C:/tmp/scene-b.usda", null),
  );
});

test("parseZapdosInitEvent recognizes error payloads", () => {
  assert.deepEqual(parseZapdosInitEvent("error: scene_usd not found"), {
    phase: "error",
    message: "scene_usd not found",
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test components/zapdos/zapdos-import.test.ts`
Expected: FAIL with module-not-found for `./zapdos-import` or missing export errors.

- [ ] **Step 3: Write the minimal helper implementation**

```ts
export type ZapdosInitPhase = "loading" | "started" | "error";

export function buildZapdosInitStreamUrl(sess: string, sceneUsd: string | null, robotUsd: string | null) {
  const query = new URLSearchParams();
  if (sceneUsd?.trim()) query.set("scene_usd", sceneUsd.trim());
  if (robotUsd?.trim()) query.set("robot_usd", robotUsd.trim());
  const suffix = query.toString();
  return suffix ? `/python/zapdos/${sess}/init/start?${suffix}` : `/python/zapdos/${sess}/init/start`;
}

export function buildZapdosSessionStorageKey(sceneUsd: string | null, robotUsd: string | null) {
  return ["zapdos-session", sceneUsd?.trim() || "", robotUsd?.trim() || ""].join("|");
}

export function parseZapdosInitEvent(data: string): { phase: ZapdosInitPhase; message: string } {
  if (data === "started") return { phase: "started", message: "started" };
  if (data.startsWith("error:")) return { phase: "error", message: data.slice(6).trim() || "Session bootstrap failed" };
  return { phase: "loading", message: data || "loading" };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm exec tsx --test components/zapdos/zapdos-import.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/zapdos-import.ts apps/web/components/zapdos/zapdos-import.test.ts
git commit -m "feat: add zapdos import helpers"
```

### Task 2: Wire Zapdos bootstrap to query params and scene-specific sessions

**Files:**
- Modify: `apps/web/app/demo/zapdos/page.tsx`
- Test: `apps/web/components/zapdos/zapdos-import.test.ts`

- [ ] **Step 1: Extend the failing test with the no-query fallback**

```ts
test("buildZapdosInitStreamUrl omits query when no import params are present", () => {
  assert.equal(buildZapdosInitStreamUrl("sess-1", null, null), "/python/zapdos/sess-1/init/start");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test components/zapdos/zapdos-import.test.ts`
Expected: FAIL if the helper always appends `?` or mishandles nulls.

- [ ] **Step 3: Modify the Zapdos page to consume the helper**

```tsx
import { useSearchParams } from "next/navigation";
import { buildZapdosInitStreamUrl, buildZapdosSessionStorageKey, parseZapdosInitEvent } from "../../../components/zapdos/zapdos-import";

export default function ZapdosInit() {
  const searchParams = useSearchParams();
  const sceneUsd = searchParams.get("scene_usd");
  const robotUsd = searchParams.get("robot_usd");
  const sess = useLocalUUID(buildZapdosSessionStorageKey(sceneUsd, robotUsd));
  const [state, setState] = useState<{ phase: "loading" | "started" | "error"; message: string }>({
    phase: "loading",
    message: "loading",
  });

  useEffect(() => {
    const sse = new EventSource(buildZapdosInitStreamUrl(sess, sceneUsd, robotUsd));
    sse.onmessage = event => {
      const next = parseZapdosInitEvent(event.data);
      setState(next);
      if (next.phase !== "loading") sse.close();
    };
    sse.onerror = () => setState({ phase: "error", message: "Session bootstrap failed" });
    return () => sse.close();
  }, [robotUsd, sceneUsd, sess]);

  return state.phase === "started" ? <Zapdos sess={sess} /> : <div className="w-full h-full text-center">{state.message}</div>;
}
```

- [ ] **Step 4: Run the helper test and typecheck**

Run: `pnpm exec tsx --test components/zapdos/zapdos-import.test.ts && pnpm exec tsc --noEmit`
Expected: PASS and exit code `0`

- [ ] **Step 5: Commit**

```bash
git add apps/web/app/demo/zapdos/page.tsx apps/web/components/zapdos/zapdos-import.ts apps/web/components/zapdos/zapdos-import.test.ts
git commit -m "feat: wire zapdos scene import bootstrap"
```

### Task 3: Add the Genie Sim handoff link and test it

**Files:**
- Create: `apps/web/components/genie-sim/OpenInZapdosLink.tsx`
- Create: `apps/web/components/genie-sim/open-in-zapdos-link.test.tsx`
- Modify: `apps/web/components/genie-sim/index.ts`
- Modify: `apps/web/app/demo/agent-genie-sim/page.tsx`

- [ ] **Step 1: Write the failing handoff link test**

```tsx
import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { OpenInZapdosLink } from "./OpenInZapdosLink";

test("OpenInZapdosLink encodes the USDA path into the Zapdos href", () => {
  const html = renderToStaticMarkup(<OpenInZapdosLink sceneUsdaPath="C:/tmp/my scene.usda" />);
  assert.match(html, /Open in Zapdos/);
  assert.match(html, /\/demo\/zapdos\?scene_usd=C%3A%2Ftmp%2Fmy%20scene\.usda/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec tsx --test components/genie-sim/open-in-zapdos-link.test.tsx`
Expected: FAIL with module-not-found for `OpenInZapdosLink`.

- [ ] **Step 3: Add the link component and mount it on the preview stage**

```tsx
export function OpenInZapdosLink({ sceneUsdaPath }: { sceneUsdaPath: string }) {
  const href = `/demo/zapdos?scene_usd=${encodeURIComponent(sceneUsdaPath)}`;
  return (
    <a className="absolute top-3 right-3 z-10 rounded-full border border-slate-700 bg-slate-950/90 px-4 py-2 text-sm text-slate-100" href={href}>
      Open in Zapdos
    </a>
  );
}
```

```tsx
<div className="relative h-full min-h-0">
  <div className="absolute inset-0"><SceneViewer scene={scene} /></div>
  <OpenInZapdosLink sceneUsdaPath={scene.sceneUsdaPath} />
  <SceneUsdaBadge path={scene.sceneUsdaPath} />
</div>
```

- [ ] **Step 4: Run the new test plus the Genie Sim component suite**

Run: `pnpm exec tsx --test components/genie-sim/open-in-zapdos-link.test.tsx components/genie-sim/scene-state.test.tsx components/genie-sim/scene-usda-badge.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/genie-sim/OpenInZapdosLink.tsx apps/web/components/genie-sim/open-in-zapdos-link.test.tsx apps/web/components/genie-sim/index.ts apps/web/app/demo/agent-genie-sim/page.tsx
git commit -m "feat: add genie sim handoff to zapdos"
```

### Task 4: Harden Zapdos session bootstrap and add backend tests

**Files:**
- Create: `apps/python/tests/test_zapdos_import.py`
- Modify: `apps/python/api/zapdos/{session}/{action}.py`

- [ ] **Step 1: Write the failing backend tests**

```py
class ZapdosImportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        MODULE.sessions.clear()

    def make_request(self, query: str = ""):
        return Request({
            "type": "http",
            "method": "GET",
            "path_params": {"session": "sess-1", "action": "init", "name": "start"},
            "query_string": query.encode("utf-8"),
            "headers": [],
        })

    async def test_failed_bootstrap_is_evicted_and_can_retry(self):
        req = self.make_request()
        with mock.patch.object(MODULE.ZapdosSession, "create", new=mock.AsyncMock(side_effect=[RuntimeError("boom"), SimpleNamespace(camera_index={})])):
            future = MODULE._get_or_create_session_future(req, "sess-1")
            with self.assertRaises(RuntimeError):
                await MODULE._await_session_future("sess-1", future)
            retry = MODULE._get_or_create_session_future(req, "sess-1")
            self.assertIs(MODULE.sessions["sess-1"], retry)

    async def test_init_stream_emits_readable_error(self):
        future = asyncio.Future()
        future.set_exception(RuntimeError("scene_usd not found"))
        MODULE.sessions["sess-1"] = future
        events = [chunk async for chunk in MODULE._init_stream("sess-1", future)]
        self.assertEqual(events, ["data: loading\\n\\n", "data: error: scene_usd not found\\n\\n"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_zapdos_import`
Expected: FAIL with missing helper names such as `_get_or_create_session_future`.

- [ ] **Step 3: Implement bootstrap helpers and route wiring**

```py
def _get_or_create_session_future(req: Request, sess: str) -> asyncio.Future[ZapdosSession]:
    future = sessions.get(sess)
    if future is not None and future.done() and future.exception() is not None:
        sessions.pop(sess, None)
        future = None
    if future is None:
        robot_usd = _input_path(req, "robot_usd", DEFAULT_ROBOT_USD)
        scene_usd = _input_path(req, "scene_usd", DEFAULT_SCENE_USD)
        future = asyncio.create_task(ZapdosSession.create(sess, robot_usd, scene_usd))
        sessions[sess] = future
    return future

async def _await_session_future(sess: str, future):
    try:
        return await future
    except Exception:
        if sessions.get(sess) is future:
            sessions.pop(sess, None)
        raise

async def _init_stream(sess: str, future):
    yield "data: loading\\n\\n"
    try:
        await _await_session_future(sess, future)
    except Exception as exc:
        yield f"data: error: {exc}\\n\\n"
        return
    yield "data: started\\n\\n"
```

- [ ] **Step 4: Run the new backend tests and the existing camera-name regression**

Run: `uv run python -m unittest tests.test_zapdos_import tests.test_zapdos_render_camera`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/python/api/zapdos/{session}/{action}.py apps/python/tests/test_zapdos_import.py
git commit -m "fix: make zapdos scene imports retryable"
```

### Task 5: Final verification

**Files:**
- Verify only

- [ ] **Step 1: Run the targeted web suite**

Run: `pnpm exec tsx --test components/zapdos/zapdos-import.test.ts components/genie-sim/open-in-zapdos-link.test.tsx components/genie-sim/scene-state.test.tsx components/genie-sim/scene-usda-badge.test.tsx`
Expected: PASS

- [ ] **Step 2: Run web typecheck**

Run: `pnpm exec tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Run the targeted Python suite**

Run: `uv run python -m unittest discover -s tests -p "test_genie_sim*.py"`
Expected: PASS

Run: `uv run python -m unittest tests.test_zapdos_import tests.test_zapdos_render_camera`
Expected: PASS

- [ ] **Step 4: Manual verification**

Open `/demo/agent-genie-sim`, generate a scene, click `Open in Zapdos`, then confirm the URL contains `scene_usd=` and a second generated scene opens a different backend session instead of reusing the first.
