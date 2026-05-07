# Remove Standalone Agent Genie Sim Page

**Goal:** Remove the standalone `/demo/agent-genie-sim` page from the web app while keeping the shared `genie-sim` web components and Python APIs available for future integration into `Zapdos`.

## Decisions

- Delete the standalone page file at `apps/web/app/demo/agent-genie-sim/page.tsx`.
- Remove the homepage link that points to `/demo/agent-genie-sim`.
- Keep `apps/web/components/genie-sim` unchanged.
- Keep `/python/genie_sim/*` unchanged.
- Keep `Zapdos` calls to shared `genie-sim` helpers unchanged.

## Why This Shape

- The user only wants to remove the independent demo surface, not the scene-generation foundation.
- `Zapdos` still reuses `genie-sim` asset-search behavior through shared web and Python code.
- Deleting only the page and entry link is the smallest change that removes the exposed demo without breaking future work on scene generation inside `Zapdos`.

## Current Constraints

- The homepage still exposes `Agent Genie Sim` through `apps/web/app/page.tsx`.
- The standalone demo page lives at `apps/web/app/demo/agent-genie-sim/page.tsx`.
- `Zapdos` still imports shared `genie-sim` helpers such as asset-search descriptions and tool client utilities.
- Python-side `genie_sim` routes remain part of the current asset and scene-generation stack.

## Change Plan

### Web routing

- Remove `apps/web/app/demo/agent-genie-sim/page.tsx`.
- Do not add a redirect or replacement page in this task.

### Homepage

- Remove the `Agent Genie Sim` link from `apps/web/app/page.tsx`.
- Leave the rest of the demo links unchanged.

### Shared modules

- Do not remove or rename anything under `apps/web/components/genie-sim`.
- Do not change any `Zapdos` imports that currently depend on those shared modules.
- Do not change `apps/python/api/genie_sim.py` or the runtime helpers behind it.

## Testing

### Web

- Update or add lightweight checks so the homepage no longer renders the removed link.
- Confirm the app still builds after deleting the route file.

### Manual verification

1. Open the homepage and confirm `Agent Genie Sim` is no longer listed.
2. Confirm `/demo/zapdos` still loads normally.
3. Confirm no unrelated `genie-sim` imports were removed from `Zapdos`.

## Non-Goals

- Moving scene generation into `Zapdos`
- Removing `apps/web/components/genie-sim`
- Removing `/python/genie_sim` APIs
- Cleaning up old `genie-sim` design docs, tests, or runtime helpers
