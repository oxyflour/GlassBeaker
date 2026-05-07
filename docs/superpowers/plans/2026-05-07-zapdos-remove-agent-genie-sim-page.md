# Remove Standalone Agent Genie Sim Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the standalone `/demo/agent-genie-sim` web page and its homepage entry while keeping shared `genie-sim` components and Python APIs untouched.

**Architecture:** This change stays entirely in the Next.js app surface. The homepage loses the public entry point, and the dedicated route file is deleted so the standalone demo is no longer routable. Shared `genie-sim` modules remain in place because `Zapdos` still reuses them.

**Tech Stack:** Next.js App Router, React 19, TypeScript, `node:test`, `react-dom/server`, pnpm workspace scripts

---

## File Structure

- Modify: `apps/web/app/page.tsx`
  - Responsibility: homepage demo link list
- Create: `apps/web/app/page.test.tsx`
  - Responsibility: assert the homepage no longer renders the removed demo link
- Delete: `apps/web/app/demo/agent-genie-sim/page.tsx`
  - Responsibility removed: standalone Agent Genie Sim route entry
- Create: `apps/web/app/demo/agent-genie-sim.test.ts`
  - Responsibility: assert the standalone route file does not exist anymore

### Task 1: Remove the homepage entry

**Files:**
- Create: `apps/web/app/page.test.tsx`
- Modify: `apps/web/app/page.tsx`

- [ ] **Step 1: Write the failing homepage test**

```tsx
import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import HomePage from "./page";

test("homepage does not list the removed Agent Genie Sim demo", () => {
  const html = renderToStaticMarkup(<HomePage />);

  assert.match(html, /Agent CopilotKit/);
  assert.match(html, /Robotic rendering/);
  assert.doesNotMatch(html, /Agent Genie Sim/);
  assert.doesNotMatch(html, /\/demo\/agent-genie-sim/);
});
```

- [ ] **Step 2: Run the homepage test to verify it fails**

Run:

```bash
pnpm exec tsx --test apps/web/app/page.test.tsx
```

Expected: FAIL because `HomePage` still renders `Agent Genie Sim` and `/demo/agent-genie-sim`.

- [ ] **Step 3: Remove the homepage link with the minimal edit**

Replace `apps/web/app/page.tsx` with:

```tsx
"use client";

export default function HomePage() {
    return <div>
        Demo Links
        <ul>
            <li>
                <a href="/demo/agent-cpk">Agent CopilotKit</a>
            </li>
            <li>
                <a href="/demo/agent-pi-web">Agent Pi Web</a>
            </li>
            <li>
                <a href="/demo/chinatsu">Circuit design</a>
            </li>
            <li>
                <a href="/demo/nijika">Antenna design</a>
            </li>
            <li>
                <a href="/demo/zapdos">Robotic rendering</a>
            </li>
        </ul>
    </div>
}
```

- [ ] **Step 4: Run the homepage test to verify it passes**

Run:

```bash
pnpm exec tsx --test apps/web/app/page.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the homepage cleanup**

```bash
git add apps/web/app/page.tsx apps/web/app/page.test.tsx
git commit -m "chore: remove homepage link for agent genie sim"
```

### Task 2: Remove the standalone route file

**Files:**
- Delete: `apps/web/app/demo/agent-genie-sim/page.tsx`
- Create: `apps/web/app/demo/agent-genie-sim.test.ts`

- [ ] **Step 1: Write the failing route-removal test**

Create `apps/web/app/demo/agent-genie-sim.test.ts` with:

```ts
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

test("standalone Agent Genie Sim page route has been removed", () => {
  const routePath = fileURLToPath(new URL("./agent-genie-sim/page.tsx", import.meta.url));

  assert.equal(existsSync(routePath), false);
});
```

- [ ] **Step 2: Run the route-removal test to verify it fails**

Run:

```bash
pnpm exec tsx --test apps/web/app/demo/agent-genie-sim.test.ts
```

Expected: FAIL because `apps/web/app/demo/agent-genie-sim/page.tsx` still exists.

- [ ] **Step 3: Delete the standalone page file**

Run:

```bash
git rm apps/web/app/demo/agent-genie-sim/page.tsx
```

- [ ] **Step 4: Run the route-removal test to verify it passes**

Run:

```bash
pnpm exec tsx --test apps/web/app/demo/agent-genie-sim.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run the web build as the integration check**

Run from the repo root:

```bash
pnpm build
```

Expected: exit code `0`; the Next.js web build and `prepare-standalone.ts` step complete without route-import errors.

- [ ] **Step 6: Commit the route removal**

```bash
git add apps/web/app/demo/agent-genie-sim.test.ts
git commit -m "chore: remove standalone agent genie sim page"
```

## Notes

- Do not delete anything under `apps/web/components/genie-sim`.
- Do not change `apps/web/components/zapdos/useZapdosAgentTools.ts`.
- Do not change `apps/python/api/genie_sim.py` or `apps/python/utils/genie_sim_runtime.py`.
- Leave historical docs and tests for shared `genie-sim` modules untouched in this task.
