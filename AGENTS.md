# General
- All files are in utf8-encoded. note windows shell does not support utf8
- Ask me if you have to modify files outside current workspace, except for that worktree root.
- Use branch `nijika-dev` for Nijika work unless I ask for a different branch.
- Use `uv` Python environment in `apps/python`
- Attention: codex will inject ELECTRON_RUN_AS_NODE and will break debugging
- **DO NOT KEEP ASKING ME QUESTIONS AFTER THINKING**. Prepare your questions and let me answer them at once.
- use zod to define parameters in CopilotKit
- Ask me before you switch branch
- DO NOT write tests for unnecessary feature changes

# Coding styles
- Try to limit content of each file in 200 lines，except for documents or plans
- Try to limit public member of each class in 10
- Avoid single-use wrappers and helpers. Keep simple control flow inline unless extraction removes real complexity or creates a reusable pure contract.

# Refactoring discipline
- Prefer inline code for simple one-off orchestration. Do not extract a helper for logic under about 10-15 lines unless it is reused by 2+ call sites or isolates a pure transformation with a stable contract.
- Do not introduce a new function only to make a path easier to unit test. First prefer testing the original failing layer.
- For protocol or boundary bugs (serialization, websocket, IPC, subprocess, cross-thread, event-loop handoff), add a regression test at the failing boundary. Helper-level tests are supplementary, not a replacement.
- If a refactor adds an extra jump but does not remove meaningful branching, shared logic, or side effects, keep the logic inline.
- When introducing a single-call-site helper anyway, explicitly justify it in the response with one of: reused soon, pure codec/validation, or side-effect isolation.

