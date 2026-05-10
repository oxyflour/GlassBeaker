# Zapdos Robot Model Select Design

**Goal:** Add a robot model selector to the Zapdos `Config` panel so the user can switch between `r1pro` and `moz1`, persist the choice in `localStorage`, and have the current page/session follow that selection.

## Decisions

- The selector lives in the existing `Config` popover in `ZapdosTopOverlay`.
- The user-facing values are `r1pro` and `moz1`.
- The persisted value in `localStorage` is the robot key, not the full USD path.
- The page still treats `robot_usd` in the URL as the runtime source of truth for session bootstrapping and session-key derivation.
- When the URL does not provide `robot_usd`, the page falls back to the persisted robot key and resolves it to a known USD path.
- The default robot is `r1pro`.

## Frontend Structure

- Add a small helper module for robot model config:
  - robot key union type
  - robot key -> `robot_usd` mapping
  - `robot_usd` -> robot key reverse lookup
  - `localStorage` read/write helpers
- Add a `RobotModelSelect` component alongside the existing `SpaceMouseModeSelect`.
- Pass the currently active robot key into `ZapdosTopOverlay` so the selector reflects the active runtime model.

## Navigation And Session Flow

- `app/demo/zapdos/page.tsx` computes the effective robot path in this order:
  - `searchParams.get("robot_usd")`
  - persisted robot key from `localStorage`, resolved to a known path
  - default `r1pro` path
- `buildZapdosSessionStorageKey()` continues to use the effective `robot_usd`, so changing models creates a different session key.
- Changing the selector updates `localStorage` and then `router.replace()` updates the current page query string while preserving other query parameters.
- Because the page already keys bootstrap work off `robot_usd`, the session naturally reinitializes on model change without extra ad hoc reset logic.

## Error Handling

- Unknown or stale `localStorage` values fall back to `r1pro`.
- Unknown `robot_usd` values in the URL do not break bootstrap; the selector can show no match or fall back to the reverse-mapped known option only when the path is recognized.
- The selector remains a frontend-only preference layer; backend API shape stays unchanged.

## Testing

- Add pure tests for:
  - key/path mapping
  - reverse lookup
  - URL/session key behavior with `r1pro` vs `moz1`
- Add UI tests for:
  - `ZapdosTopOverlay` rendering the robot selector
  - selector showing the active model
- Keep the implementation split into a helper module and a small UI component so existing files stay bounded.
