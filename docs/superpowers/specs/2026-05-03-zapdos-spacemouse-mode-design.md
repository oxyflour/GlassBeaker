# Zapdos SpaceMouse Mode Design

**Goal:** Make the SpaceMouse backend start with the Python service and let the Zapdos page control teleop with a three-state selector: `off`, `left`, `right`.

## Decisions

- The SpaceMouse manager starts automatically when the `apps/python` service starts.
- Autostart does not imply motion. The manager starts in `off` mode.
- `off` keeps the teleop thread, device polling, and ROS polling alive, but suppresses joint command publishing and ignores motion input.
- `left` and `right` enable command publishing for the corresponding arm and snap control to that arm.

## Backend

- Extend `SpaceMouseManager` with explicit `mode` state: `off | left | right`.
- Keep `active_arm` as the currently selected arm for IK target management.
- Poll the device before the first joint state is available so `device_connected` can become true even while ROS is still warming up.
- Add a high-level API operation that sets the mode instead of forcing the frontend to orchestrate `start`, `stop`, and `set_active_arm`.
- Include `mode` in the status payload. `running` means the worker thread is alive; `mode` controls whether teleop is armed.
- Register a router startup hook that calls `manager.start()` once when the Python service starts.

## Frontend

- Add a compact selector to the Zapdos page with options `关闭`, `左臂`, and `右臂`.
- Read `/python/teleop/spacemouse/status` on mount to initialize the selector from backend state.
- On selector change, call the new mode endpoint and keep the UI state in sync with the returned status.
- Keep the selector UI in its own component so `apps/web/app/demo/zapdos/page.tsx` does not grow further.

## Testing

- Backend tests cover:
  - startup autostart behavior
  - `set_mode("off" | "left" | "right")`
  - no joint command publishing while mode is `off`
  - device polling before joint state arrival
- Frontend tests cover:
  - status-to-mode mapping helper
  - request payload generation for selector actions
