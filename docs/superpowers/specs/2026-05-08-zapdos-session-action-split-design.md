# Zapdos Session Action Split Design

**Goal:** Split `apps/python/api/zapdos/{session}/{action}.py` so the route file only handles HTTP/session bootstrapping, while scene rebuild logic and `ZapdosSession` runtime move into focused backend modules without changing external behavior.

## Problem

The current action file mixes five concerns:

- FastAPI route dispatch
- session bootstrap and registry access
- MuJoCo and Isaac runtime orchestration
- overlay scene operation state and subprocess rebuild flow
- SSE and MJPEG streaming helpers

That makes the route file hard to scan, hard to test in isolation, and prone to receiving more runtime logic because the class already lives there.

## Constraints

- Keep all existing routes and tool names unchanged.
- Keep `ZapdosSession`, `_stream_scene_operation`, and related symbols importable from the action module for compatibility with current tests.
- Do not change the request or response shape of `set_scene_assets`, `remove_asset_from_scene`, render streaming, or init SSE.
- Avoid touching unrelated dirty files in the worktree.

## Chosen Shape

Split the current file into three layers:

### 1. Route layer

`apps/python/api/zapdos/{session}/{action}.py`

Responsibilities:

- request path/query parsing
- session registry access
- init/start stream
- route dispatch for `stream`, `op`, `ros`, `call`, `render`, and `asset`
- compatibility re-exports for tests

### 2. Session runtime layer

`apps/python/utils/zapdos/zapdos_session.py`

Responsibilities:

- define `ZapdosSession`
- own MuJoCo/Isaac runtime lifecycle
- own pose updates, ROS publishing, MJPEG rendering, and teardown
- delegate overlay scene operation internals to the scene-operations module

### 3. Scene operation layer

`apps/python/utils/zapdos/zapdos_scene_operations.py`

Responsibilities:

- define `PreparedOverlayRebuild`, `OverlayRebuildCompletion`, and `SceneOperation`
- build replacement overlay payloads for `set_scene_assets` and `remove_asset_from_scene`
- manage operation futures and completion queues
- run overlay rebuild subprocess preparation
- expose `_stream_scene_operation` as a reusable SSE helper

## Why This Split

- It moves the heaviest state machine out of the route file first, which is the highest-value split.
- It keeps external behavior stable because the action module can re-export moved names.
- It keeps test churn low: existing tests can continue importing through the action module while asserting the implementation now lives elsewhere.
- It creates a clear landing zone for future splits if `ZapdosSession` still grows.

## Non-Goals

- No frontend API changes
- No semantic changes to overlay rebuild behavior
- No new persistence model
- No broad refactor of other Python APIs

## Validation

- Add a regression test that requires `ZapdosSession` and `_stream_scene_operation` to come from the new modules.
- Re-run focused Python tests covering action import, overlay rebuild subprocess, session registry, and render camera validation.
