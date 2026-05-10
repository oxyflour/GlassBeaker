# Zapdos IsaacSim Crash Status

## Summary

- `ZapdosSession` 本身不是先死掉的组件。
- 当前 `IsaacRenderer` 被 `DELETE /api/isaac` 清掉，是因为 IsaacSim 子进程先退出，Python 侧在 `wait_ready()` 或进程状态刷新里发现它已经不在了，随后执行清理。
- Next.js 终端里只看到连续的 `GET /api/isaac` 和最后一条 `DELETE /api/isaac`，这只是后果，不是根因。

## Observed Symptom

- 前端和 Next.js 侧能看到类似下面的请求日志：
  - `GET /api/isaac?id=glassbeaker_...`
  - `DELETE /api/isaac`
- 但这些日志没有说明 IsaacSim 为什么退出。
- 真实根因需要看 `apps/python/tmp/renderer_*.log`。

## Confirmed Findings

### 1. Session timeout was not the root cause

- 代码里 session 默认有 120 秒 idle timeout。
- 但这次问题里，用户确认 session 仍然活着，后续日志也证明 renderer 是子进程先崩。
- 因此 “session 过期导致 renderer 被销毁” 不是这次主因。

### 2. First confirmed crash: stale flat import path

- 旧版 `apps/isaac/rl_renderer_entry.py` 里仍然引用了已经移除的平铺模块：
  - `from utils.zapdos.isaac_renderer_reload import ...`
  - `from utils.zapdos.renderer_ipc import ...`
- 对应 renderer 日志里出现过：
  - `ModuleNotFoundError: No module named 'utils.zapdos.isaac_renderer_reload'`
- 这会导致 IsaacSim 在启动后不久直接退出。

### 3. Second confirmed crash: import chain accidentally pulled in `mujoco`

- 修完旧路径后，新的崩溃点是：
  - `from utils.zapdos.renderer.control_channel import request_path, response_path`
- 这条 import 先执行了 `utils.zapdos.renderer.__init__`。
- `utils.zapdos.renderer.__init__` 又导入 `.base`。
- `apps/python/utils/zapdos/renderer/base.py` 在运行时导入了 `utils.zapdos.bundle.RenderBundle`。
- 这条链进一步导入了 `apps/python/utils/zapdos/bundle/camera_specs.py`。
- `camera_specs.py` 顶层依赖 `mujoco`。
- IsaacSim 使用的是 `apps/isaac/.venv`，其中没有安装 `mujoco`，因此 renderer 日志里出现：
  - `ModuleNotFoundError: No module named 'mujoco'`

### 4. Why Next.js only showed GET/DELETE

- `/api/isaac` 只是一个轻量进程管理层。
- Python 侧会不断 `GET /api/isaac?id=...` 轮询子进程状态。
- 一旦 IsaacSim 子进程已经退出，Python 侧的 `IsaacRenderer.wait_ready()` 或 `_refresh_process_state()` 会发现 `running = false`。
- 随后 `IsaacRenderer.close()` 会调用 `DELETE /api/isaac` 做清理。
- 所以 Next.js 终端里只看到 GET/DELETE，不会自动带出真正的 Python 异常堆栈。

## Fixes Applied

### 1. Renderer entry import paths

- 已将 `apps/isaac/rl_renderer_entry.py` 改为使用当前包路径：
  - `utils.zapdos.renderer.isaac_renderer_reload`
  - `utils.zapdos.renderer.control_channel`

### 2. Remove runtime dependency on `RenderBundle` inside renderer protocol

- 已将 `apps/python/utils/zapdos/renderer/base.py` 中的 `RenderBundle` 改成 `TYPE_CHECKING` 条件导入。
- 这样 `RendererBackend` 作为协议类型不会在运行时把整条 `bundle -> camera_specs -> mujoco` 链拖进 IsaacSim 子进程。

### 3. Regression coverage

- 已新增测试覆盖以下约束：
  - `rl_renderer_entry.py` 不再使用旧的平铺导入路径。
  - `import utils.zapdos.renderer.control_channel` 在 `mujoco` 不可导入时仍然应当成功。

## Verification

已运行：

```powershell
uv run python -m unittest tests.test_camera_math tests.test_renderer_reload
```

结果：

```text
Ran 7 tests in 0.268s
OK
```

## Follow-up Update

- The remaining `/api/isaac` diagnostics gap is now closed in code:
  - Next.js process management now logs `launch-error`, unexpected `quit`, and intentional `stopped` events with `id`, `pid`, `exitCode`, and `logPath`.
  - The frontend runtime error pass-through has regression coverage so messages like `IsaacSim quit unexpectedly, check ...` keep reaching the user unchanged.
- Fresh verification for this follow-up:
  - `pnpm test app/api/isaac/logging.test.ts app/api/isaac/process-events.test.ts components/zapdos/zapdos-runtime.test.ts`
  - `pnpm exec tsc --noEmit --pretty false`
  - `uv run python -m unittest tests.test_sim_env_renderer tests.test_zapdos_camera_streaming tests.test_camera_math`

## Current Status

- 已经确认并修掉两条会导致 IsaacSim 子进程早期退出的导入问题。
- 相关单元测试已通过。
- 还没有重新做一遍完整的 Zapdos 端到端启动验证。
- 还没有实现“IsaacSim quit, check xxx.log”这类更明确的服务端/前端日志提示。

## Remaining Gap

当前缺少明确的退出诊断机制：

- Next.js `/api/isaac` 子进程 `exit/error` 时没有稳定打印 `id`、`pid`、`exitCode`、`logPath`。
- Python 侧 `wait_ready()` / control request 失败时，没有统一输出：
  - `IsaacSim failed to start, check ...`
  - `IsaacSim quit unexpectedly, check ...`

这会让排查过程仍然依赖手动翻 `apps/python/tmp/renderer_*.log`。

## Recommended Next Step

按下面顺序继续：

1. 先重新验证一遍当前 Zapdos 启动路径，确认 renderer 不再因导入链崩溃。
2. 在 `apps/web/app/api/isaac/route.ts` 增加稳定的子进程退出日志。
3. 在 `apps/python/utils/zapdos/renderer/isaac_renderer.py` 里把启动失败和运行时退出转换成明确错误消息，并带上 `log_path`。
4. 让前端 runtime error 文案直接提示用户检查对应 renderer log。
