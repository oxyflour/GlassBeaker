# Zapdos Robot Model Select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a robot model selector to the Zapdos Config panel that switches between `r1pro` and `moz1`, persists the selected robot key in `localStorage`, and keeps Zapdos bootstrap/session behavior keyed by the effective `robot_usd`.

**Architecture:** Add one focused robot-model helper module for key/path/storage resolution, one small selector component for the Config popover, and wire the active robot key through the existing Zapdos page -> scene -> overlay path. Keep session bootstrapping unchanged except for resolving the effective `robot_usd` from URL first, persisted key second, and default robot last.

**Tech Stack:** Next.js 16 client components, React 19, TypeScript, `node:test`, `react-dom/server`, `tsx`

---

### Task 1: Robot Model Helper And Pure Tests

**Files:**
- Create: `apps/web/components/zapdos/robot-model.ts`
- Create: `apps/web/components/zapdos/robot-model.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_ROBOT_MODEL_KEY,
  getRobotModelKeyFromUsd,
  getRobotUsdForModel,
  readPersistedRobotModelKey,
  resolveEffectiveRobotUsd,
  ROBOT_MODEL_STORAGE_KEY,
  writePersistedRobotModelKey,
} from "./robot-model";

test("resolveEffectiveRobotUsd prefers a recognized URL robot over persisted state", () => {
  assert.equal(
    resolveEffectiveRobotUsd("deps/moz01/spirit01_model/urdf/moz1.urdf", "r1pro"),
    "deps/moz01/spirit01_model/urdf/moz1.urdf"
  );
});

test("resolveEffectiveRobotUsd falls back to persisted robot key when URL is absent", () => {
  assert.equal(
    resolveEffectiveRobotUsd(null, "moz1"),
    "deps/moz01/spirit01_model/urdf/moz1.urdf"
  );
});

test("resolveEffectiveRobotUsd falls back to the default robot for stale persisted values", () => {
  assert.equal(
    resolveEffectiveRobotUsd(null, "stale-value"),
    getRobotUsdForModel(DEFAULT_ROBOT_MODEL_KEY)
  );
});

test("getRobotModelKeyFromUsd only reverse-maps known robot paths", () => {
  assert.equal(getRobotModelKeyFromUsd("deps/galaxea/object/r1pro/r1pro.usda"), "r1pro");
  assert.equal(getRobotModelKeyFromUsd("deps/custom/custom.usd"), null);
});

test("robot model storage helpers persist the robot key instead of the USD path", () => {
  const memory = new Map<string, string>();
  const storage = {
    getItem(key: string) {
      return memory.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      memory.set(key, value);
    },
  };

  writePersistedRobotModelKey("moz1", storage);

  assert.equal(memory.get(ROBOT_MODEL_STORAGE_KEY), "moz1");
  assert.equal(readPersistedRobotModelKey(storage), "moz1");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test components/zapdos/robot-model.test.ts`
Expected: FAIL with module-not-found or missing-export errors for `./robot-model`

- [ ] **Step 3: Write minimal implementation**

```typescript
export const ROBOT_MODEL_STORAGE_KEY = "zapdos.robot-model";

export type RobotModelKey = "r1pro" | "moz1";

export const DEFAULT_ROBOT_MODEL_KEY: RobotModelKey = "r1pro";

const ROBOT_USD_BY_KEY: Record<RobotModelKey, string> = {
  r1pro: "deps/galaxea/object/r1pro/r1pro.usda",
  moz1: "deps/moz01/spirit01_model/urdf/moz1.urdf",
};

export function getRobotUsdForModel(key: RobotModelKey) {
  return ROBOT_USD_BY_KEY[key];
}

export function getRobotModelKeyFromUsd(robotUsd: string | null) {
  const entry = Object.entries(ROBOT_USD_BY_KEY).find(([, value]) => value === robotUsd);
  return entry ? entry[0] as RobotModelKey : null;
}

function coerceRobotModelKey(value: string | null | undefined) {
  return value === "moz1" || value === "r1pro" ? value : DEFAULT_ROBOT_MODEL_KEY;
}

export function readPersistedRobotModelKey(storage: Pick<Storage, "getItem"> | null = typeof window === "undefined" ? null : window.localStorage) {
  return coerceRobotModelKey(storage?.getItem(ROBOT_MODEL_STORAGE_KEY));
}

export function writePersistedRobotModelKey(key: RobotModelKey, storage: Pick<Storage, "setItem"> | null = typeof window === "undefined" ? null : window.localStorage) {
  storage?.setItem(ROBOT_MODEL_STORAGE_KEY, key);
}

export function resolveEffectiveRobotUsd(urlRobotUsd: string | null, persistedRobotKey: string | null) {
  return urlRobotUsd?.trim() || getRobotUsdForModel(coerceRobotModelKey(persistedRobotKey));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --dir apps/web test components/zapdos/robot-model.test.ts`
Expected: PASS with 5 tests and 0 failures

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/robot-model.ts apps/web/components/zapdos/robot-model.test.ts
git commit -m "feat: add zapdos robot model helpers"
```

### Task 2: Config Popover Robot Selector

**Files:**
- Create: `apps/web/components/zapdos/RobotModelSelect.tsx`
- Modify: `apps/web/components/zapdos/ZapdosTopOverlay.tsx`
- Modify: `apps/web/components/zapdos/zapdos-top-overlay.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
test("ZapdosTopOverlay renders the robot selector when settings are open", () => {
  const html = renderToStaticMarkup(
    <ZapdosTopOverlay
      activeRobotModelKey="moz1"
      defaultSettingsOpen
      onRobotModelChange={ () => undefined }
      sess="sess-1"
      sse={1} />
  );

  assert.match(html, /Robot model/);
  assert.match(html, /option selected="" value="moz1"/);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test components/zapdos/zapdos-top-overlay.test.tsx`
Expected: FAIL because `activeRobotModelKey` and `onRobotModelChange` props do not exist yet

- [ ] **Step 3: Write minimal implementation**

```typescript
export function RobotModelSelect({
  activeRobotModelKey,
  onChange,
}: {
  activeRobotModelKey: RobotModelKey | null;
  onChange: (key: RobotModelKey) => void;
}) {
  return <div className="rounded-md bg-black/60 px-3 py-2 text-white backdrop-blur-sm">
    <label className="mr-2 text-sm" htmlFor="robot-model">Robot model</label>
    <select
      id="robot-model"
      className="rounded border border-white/20 bg-black/40 px-2 py-1 text-sm"
      onChange={ event => onChange(event.target.value as RobotModelKey) }
      value={ activeRobotModelKey ?? "" }>
      <option disabled value="">Unknown URL model</option>
      <option value="r1pro">r1pro</option>
      <option value="moz1">moz1</option>
    </select>
  </div>;
}
```

```typescript
<ZapdosTopOverlay
  activeRobotModelKey={ activeRobotModelKey }
  onRobotModelChange={ onRobotModelChange }
  mode={ mode }
  selectedBody={ selectedBody }
  sess={ sess }
  sse={ sse } />
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --dir apps/web test components/zapdos/zapdos-top-overlay.test.tsx`
Expected: PASS with the new selector assertions and 0 failures

- [ ] **Step 5: Commit**

```bash
git add apps/web/components/zapdos/RobotModelSelect.tsx apps/web/components/zapdos/ZapdosTopOverlay.tsx apps/web/components/zapdos/zapdos-top-overlay.test.tsx
git commit -m "feat: add zapdos robot model selector"
```

### Task 3: Page Wiring For URL, Persistence, And Session Keys

**Files:**
- Modify: `apps/web/app/demo/zapdos/page.tsx`
- Modify: `apps/web/components/zapdos/ZapdosScene.tsx`
- Modify: `apps/web/components/zapdos/zapdos-import.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
test("buildRobotModelHref preserves unrelated query params and replaces robot_usd", () => {
  assert.equal(
    buildRobotModelHref(
      "/demo/zapdos",
      "scene_usd=C%3A%2Ftmp%2Fscene.usda&view=debug&robot_usd=old-value",
      "moz1"
    ),
    "/demo/zapdos?scene_usd=C%3A%2Ftmp%2Fscene.usda&view=debug&robot_usd=deps%2Fmoz01%2Fspirit01_model%2Furdf%2Fmoz1.urdf"
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --dir apps/web test components/zapdos/zapdos-import.test.ts components/zapdos/robot-model.test.ts`
Expected: FAIL because `buildRobotModelHref` does not exist yet

- [ ] **Step 3: Write minimal implementation**

```typescript
const urlRobotUsd = searchParams.get("robot_usd");
const effectiveRobotUsd = resolveEffectiveRobotUsd(
  urlRobotUsd,
  typeof window === "undefined" ? null : window.localStorage.getItem(ROBOT_MODEL_STORAGE_KEY)
);
const activeRobotModelKey = getRobotModelKeyFromUsd(effectiveRobotUsd);
const storageKey = buildZapdosSessionStorageKey(sceneUsd, effectiveRobotUsd);
```

```typescript
export function buildRobotModelHref(pathname: string, search: string, key: RobotModelKey) {
  const nextParams = new URLSearchParams(search);
  nextParams.set("robot_usd", getRobotUsdForModel(key));
  const suffix = nextParams.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}
```

```typescript
function handleRobotModelChange(next: RobotModelKey) {
  writePersistedRobotModelKey(next);
  router.replace(buildRobotModelHref(pathname, searchParams.toString(), next));
}
```

```typescript
<ZapdosScene
  activeRobotModelKey={ activeRobotModelKey }
  onRobotModelChange={ handleRobotModelChange }
  onRuntimeError={ message => setState({ phase: "error", message }) }
  sess={ sess } />
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm --dir apps/web test components/zapdos/robot-model.test.ts components/zapdos/zapdos-import.test.ts components/zapdos/zapdos-top-overlay.test.tsx`
Expected: PASS with 0 failures

- [ ] **Step 5: Commit**

```bash
git add apps/web/app/demo/zapdos/page.tsx apps/web/components/zapdos/ZapdosScene.tsx apps/web/components/zapdos/zapdos-import.test.ts
git commit -m "feat: persist zapdos robot model selection"
```
