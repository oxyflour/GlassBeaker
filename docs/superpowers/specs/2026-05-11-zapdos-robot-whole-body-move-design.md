# Zapdos Robot Whole-Body Move Design

**Goal:** Let the user click either `r1pro` or `moz1` anywhere in the scene, select the robot as one object, and move the whole robot with the existing Zapdos transform controls.

## Decisions

- Do not hardcode per-model selection logic in the frontend.
- Treat "selectable" and "movable" as different concepts.
- Keep scene assets movable as they are today.
- Make robot whole-body movement work by routing any robot hit to that robot's root body.
- Discover robot root bodies dynamically from `body_map` and MuJoCo parent links so the behavior works for both `r1pro` and `moz1`.

## Verified Runtime Facts

- `r1pro` page input uses `deps/galaxea/object/r1pro/r1pro.usda`.
- Its robot root body is `Root_r1_pro_with_gripper_base_link`.
- `moz1` page input uses `deps/spirit01_model/USD/Moz1_robot_only.usda`.
- Its robot root body is `Root_base_link`.
- Moving either root body in MuJoCo moves the whole robot subtree.

## Problem In Current Flow

- `get_visual()` exposes bodies with only `name`, `label`, `editable`, and `matrix`.
- Frontend picking currently knows which body mesh was hit, but it does not know whether that body should select itself or a different owning body.
- Robot bodies are deliberately excluded from `editable_body_names`, so the frontend cannot use the existing "editable means draggable" rule to support whole-robot movement.
- If we simply mark all robot links editable, the user would drag whichever link was hit instead of moving the robot as one object.

## Target Data Contract

Extend each body returned by `get_visual()` with:

- `selectable: boolean`
- `movable: boolean`
- `selectionBody: string | null`

Rules:

- Scene object body:
  - `selectable = true`
  - `movable = true`
  - `selectionBody = body.name`
- Robot root body:
  - `selectable = true`
  - `movable = true`
  - `selectionBody = body.name`
- Non-root robot body:
  - `selectable = true`
  - `movable = false`
  - `selectionBody = robot root body name`
- Static world geometry without a body stays non-selectable through the existing `null` body behavior.

## Backend Changes

### Body capability sets

In `MujocoPhysics`, derive three sets:

- `editable_body_names`: existing scene-edit bodies
- `robot_body_names`: all bodies whose render path starts with `MyRobot/`
- `movable_body_names`: `editable_body_names + robot_root_body_names`

Add a helper that finds robot root bodies by:

1. taking all `robot_body_names`
2. resolving each MuJoCo parent body
3. keeping bodies whose parent is not also a robot body

This makes the result model-agnostic and covers both `r1pro` and `moz1`.

### Selection mapping

Add a helper that maps any body name to its effective selection body:

- editable scene body -> itself
- robot body -> owning robot root body
- everything else -> `null`

`get_visual()` should serialize that mapping into `selectionBody`, plus `selectable` and `movable`.

### Pose editing

Update `set_body_pose()` to allow bodies in `movable_body_names`, not only `editable_body_names`.

This keeps scene assets movable and additionally allows robot root bodies to move.

### Pose override persistence

Keep persisting pose overrides for any body accepted by `set_body_pose()`.

During runtime bundle swap, replay overrides for `movable_body_names` so robot moves survive rebuilds the same way scene-body moves do.

## Frontend Changes

### Picking

Replace the current "pick body name directly from hit" behavior with:

- read `zapdosSelectionBody`
- fall back to `zapdosBody` only if no explicit selection mapping is available

This lets a hit on any robot link select the root body.

### Transform controls

Show `TransformControls` when the selected object is `movable`, not only when it is `editable`.

Scene assets remain draggable.
Robot roots become draggable.
Non-root robot links never get their own transform gizmo.

### Runtime object metadata

When loading bodies into Three.js objects, attach:

- `zapdosBody`
- `zapdosEditable`
- `zapdosMovable`
- `zapdosSelectionBody`

Mesh descendants should inherit the same selection metadata from their owning body group.

## Testing

### Python

- root-body detection returns the correct movable robot root for a robot-only body map
- `get_visual()` exposes `selectionBody`, `selectable`, and `movable`
- non-root robot bodies map to the robot root selection body
- `set_body_pose()` accepts robot root bodies and still rejects non-root robot bodies
- runtime swap reapplies pose overrides for movable robot roots

### Web

- pick helper returns the mapped `selectionBody`
- a robot-link hit selects the robot root body
- transform controls render for movable robot roots
- transform controls do not render for non-movable bodies

## Non-Goals

- per-link robot dragging
- separate robot manipulation UI
- model-name conditionals for `r1pro` versus `moz1`
