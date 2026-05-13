# Zapdos Multi-Robot Scene Model Design

**Goal:** Capture the key architecture decisions from the `overlay` / `runtime` discussion and define the correct direction for future multi-robot support.

## Decisions

- Keep the current single-robot behavior working without changing session bootstrap semantics right now.
- Treat `runtime` and persisted scene edits as different layers.
- Keep robot whole-body pose edits in persisted pose overrides for the current single-robot model.
- Do not model additional robots as today's `OverlayInstance`.
- For real multi-robot support, replace the top-level `scene_usd + robot_usd + overlay` mental model with a scene composition model that can represent multiple robot and object actors as peers.

## Current Layer Meanings

### Runtime

`runtime` means live execution state owned by the current `ZapdosSession`.

Examples:

- MuJoCo model and data
- renderer process and frame buffers
- camera index and last-frame bookkeeping
- rebuild jobs, queues, and executor state

This state is recreated when a session restarts and should not be treated as the long-term source of truth.

### Overlay

`overlay` currently means session-local persisted scene edits on top of immutable bootstrap inputs.

Today it stores:

- `instances`: extra scene assets inserted after session start
- `pose_overrides`: post-spawn body transform edits that should survive rebuilds

This is already broader than "extra assets only" because robot-root movement now also persists through `pose_overrides`.

## Why `overlay` Feels Wrong

The discomfort is valid. The current architecture is not just using a narrow name; it is built around a single-robot world model.

Verified single-robot assumptions in the current code:

- Session bootstrap takes one `robot_usd` and one `scene_usd`.
- Render stages hardcode the robot namespace as `/MyRobot`.
- Physics capability detection identifies robot bodies by `render_path.startswith("MyRobot/")`.
- Frontend session identity and robot switching logic are based on a single `robot_usd`.

Because of those assumptions, the current model can express:

- one immutable base scene
- one immutable primary robot
- many extra scene objects
- pose edits on movable bodies

It cannot cleanly express:

- 10 peer robots in one scene
- robot-specific cameras and controls per instance
- multiple robot namespaces without path collisions
- robot instances that are added and removed the same way as other actors

## Current Rule For Robot Movement

For the current single-robot design:

- the robot itself is not an `OverlayInstance`
- robot whole-body movement should still persist in `overlay_state.pose_overrides`

This is correct because a pose override is an edit to an existing body, not the creation of a new scene actor.

## Correct Future Model For Multi-Robot

If Zapdos must support many robots as first-class scene entities, the top-level model should become a scene composition model.

Suggested shape:

```text
SceneState
|- base_scene_usd
|- actors
|  |- RobotInstance
|  |  |- id
|  |  |- robot_asset or robot_usd
|  |  |- placement
|  |  |- motion or control policy
|  |  `- camera or capability config
|  `- ObjectInstance
|     |- id
|     |- asset_id or url
|     |- motion
|     `- placement
`- pose_overrides
```

In that model:

- robots and objects are peer actor types
- `robot_usd` is no longer the only robot input to the session
- each robot instance gets its own namespace, for example `/Robots/<id>/...`
- body capability logic keys off instance ownership instead of a hardcoded `/MyRobot` prefix
- camera overrides and selection mapping can be scoped per robot instance

## Migration Guidance

Short term:

- keep the current single-robot bootstrap contract
- keep robot-root pose persistence in `pose_overrides`
- avoid expanding `OverlayInstance` to fake robot instances

Medium term:

- rename the persisted scene-edit layer from `overlay` to something broader such as `SceneState`, `SceneComposition`, or `SceneDelta`
- replace the single `robot_usd` bootstrap model with actor-based scene composition
- replace `/MyRobot` hardcoding with per-instance namespaces

## Non-Goals For The Current Refactor

- implementing multi-robot support immediately
- making `OverlayInstance` represent robots
- keeping the name `overlay` if the data model grows into full scene composition
