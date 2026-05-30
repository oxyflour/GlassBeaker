# Zapdos 接入 BlenderRenderer 计划

## Summary
- 在现有 `zapdos` 会话流程里把 renderer 做成可选后端，默认仍是 `IsaacRenderer`。
- 新增 `BlenderRenderer`，目标能力与 `IsaacRenderer` 对等：实时读帧、`reload_scene`、`snapshot_cameras`、状态查询、关闭。
- `RenderBundle` 继续作为场景与相机的单一输入源；这一轮不改 overlay/physics 主流程。

## Key Changes
- Renderer 抽象层：
  - 保留 `RendererBackend` 现有方法集，不再让 session 直接依赖 `IsaacRenderer` 具体类。
  - 在 renderer 包内补一个共享的 `CameraSnapshot` 类型，统一 `snapshot_cameras()` 返回结构，兼容当前 `save_camera_overrides()` 需要的字段。
  - 统一 `status()` 的最小公共字段：`backend`、`running`、`ready`，其余字段可后端自扩展。

- Renderer 选择与创建：
  - 给 zapdos session 初始化增加可选参数 `renderer=isaac|blender`，默认 `isaac`。
  - 在 renderer 包内增加 factory/config，负责按 `renderer_kind` 创建 `IsaacRenderer` 或 `BlenderRenderer`。
  - `ZapdosSession.__init__` 和 runtime bundle swap 过程都改成走 factory，去掉对 `IsaacRenderer` 的硬编码。

- BlenderRenderer 实现：
  - 新增 `BlenderRenderer` 类，作为 `RendererBackend` 的 drop-in 实现。
  - 新增 Blender worker 启动脚本，用 `blender --background --python ...` 起常驻子进程。
  - 输入直接使用 `bundle.render_scene_usda` 和 `bundle.cameras`；Blender 侧负责 USD 导入、相机绑定、渲染输出。
  - 复用现有 `SharedFrameBuffer` 作为帧传输层，保持 `read(camera_name) -> (frame_index, np.ndarray)` 契约不变。
  - 复用现有文件式 control channel 思路处理 `snapshot_cameras` 和 `reload_scene`，这样 session/streaming 层基本不用分叉。
  - `snapshot_cameras()` 返回的数据结构必须与当前 camera override 存储格式兼容：`parent_prim`、`name`、`pos`、`quat`、`fovy`、`horizontal_aperture`、`vertical_aperture`、`clipping_range`。

- Session / rebuild 适配：
  - `ZapdosSession` 保存选中的 `renderer_kind`，overlay rebuild 之后继续使用同一种 backend。
  - `_swap_runtime_bundle()` 不再直接 new `IsaacRenderer`；优先尝试当前 backend 的 `reload_scene()`，失败则按同一 `renderer_kind` 重建新实例。
  - `SessionStreamingMixin` 继续只依赖协议方法 `wait_ready/read/close`，不引入 Blender 分支逻辑。

## Test Plan
- 为 `BlenderRenderer` 补一套与现有 Isaac renderer 等价的契约测试：`wait_ready`、`read`、`snapshot_cameras`、`reload_scene`、`close`、异常路径。
- 增加 session 初始化测试：默认 `isaac`、显式 `renderer=blender`、非法值拒绝。
- 增加 overlay rebuild 测试：会话选择 Blender 后，runtime swap 仍然保留 Blender，不回退成 Isaac 硬编码。
- 复用现有 camera streaming / camera override 测试，确认后端切换不影响会话层行为。
- 加一个 smoke test：mock 掉 Blender 进程边界，验证 `renderer=blender` 的会话能拿到第一帧并可保存 camera override。

## Assumptions
- 这一轮的 BlenderRenderer 是 `IsaacRenderer` 的平替后端，不是离线渲染工具。
- 帧读取继续走共享内存，避免改动当前 `SessionStreamingMixin` 的拉帧逻辑。
- `RenderBundle.render_scene_usda` 与 `RenderCamera` 已足够作为 Blender 输入；如果后续需要 `.blend` 模板，放到 renderer config 或环境变量，不放进 `RenderBundle`。
- 默认后端保持 `isaac`，以保证现有调用方和测试默认行为不变。
