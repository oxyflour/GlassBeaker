# Zapdos IsaacRenderer 进程复用方案

## Summary
选择的方案是：只针对 Zapdos 同一 session 的 overlay 重建做进程复用，在现有 renderer IPC 上新增 `reload_scene`，让同一个 IsaacSim 进程内部替换 scene 并刷新绑定状态；如果热重载失败，自动回退到当前的“重启 renderer”路径。

不选 `/api/isaac` 层进程池/进程复用。那一层只管理子进程生命周期，不持有 stage、camera annotator、body map 或 TF 缓存，不能解决真正昂贵的 IsaacSim 初始化成本。

## Key Changes
### 1. Python 侧 `IsaacRenderer` 控制面
- 在 `apps/python/utils/sim_env.py` 为 renderer IPC 增加统一的同步控制请求 helper。
- 所有基于 `request.json` / `response.json` 的控制请求都走同一个互斥锁，避免 `snapshot_cameras` 与 `reload_scene` 并发踩同一对文件。
- 新增 `IsaacRenderer.reload_scene(bundle, timeout=...) -> None`：
  - 发送 `reload_scene` 请求，负载至少包含新的 `scene_usd` 和 `cameras`。
  - 在成功返回前，不更新本地 `bundle` / `camera_index`。
  - 成功后更新 `bundle`、`camera_index`，保留原有 `shm` 绑定。
  - 失败时抛错，由上层决定回退到重启路径。

### 2. IsaacSim 进程内热重载
- 在 `apps/isaac/rl_renderer_entry.py` 扩展控制 op：
  - 保留 `snapshot_cameras`
  - 新增 `reload_scene`
- `reload_scene` 的执行步骤固定为：
  - 校验新 camera 拓扑与当前 renderer 兼容：同一 `num_envs`、同一 camera 数量、同一 camera 名称顺序、同一分辨率。
  - 进入 reload 临界区，停止使用旧 annotator。
  - detach 旧 annotator，destroy 旧 render products。
  - 替换 `/World/envs` 场景子树并重新构建 env clone。
  - 更新 `self.args.scene_usd` 与 `self._camera_list`。
  - 基于新 `scene_usd` 对应的 body map 重新计算 `body_name_map`。
  - 对每个 `EnvTFSubscriber` 刷新 `body_name_map`，清空 `_attr_cache` 和 `_ordered_attrs`。
  - 调用 `world.reset()`，重新挂接 camera annotator。
  - 成功后返回 `ok: true`；任一步失败返回 `ok: false`，不杀进程。
- 不重建：
  - ROS executor / subscription
  - shared memory 名称与布局
  - render callback
  - IsaacSim 进程本身

### 3. Zapdos overlay swap 路径
- 在 `apps/python/api/zapdos/{session}/{action}.py` 调整 `_swap_runtime_bundle()`：
  - 先构建 `new_physics`
  - 优先调用现有 `self.renderer.reload_scene(bundle)`
  - reload 成功时：
    - 复用同一个 renderer 对象
    - 更新 `self.bundle`、`self.physics`、`camera_index`、`last_frame_index`
    - 关闭旧 physics，不关闭 renderer
  - reload 失败时：
    - 自动回退到创建全新 `IsaacRenderer` 的旧路径
    - 新 renderer 启动成功后再替换 session 引用
    - 然后关闭旧 renderer / 旧 physics
- `_rebuild_overlay_runtime()` 的对外语义保持不变：
  - 成功返回新 `scene_revision`
  - 热重载和重启都失败时，回滚 overlay state 和 revision

## Public Interfaces
- 新增 `IsaacRenderer.reload_scene(bundle, timeout=...) -> None`
- renderer IPC 新增请求：
  - `{"id": "...", "op": "reload_scene", "scene_usd": "...", "cameras": [...]}`
- 不新增前端 API；`add_asset_to_scene` / `remove_asset_from_scene` 继续走原路由

## Test Plan
- `apps/python/tests/test_sim_env_renderer.py`
  - `reload_scene` 成功时发送正确 IPC，更新本地 bundle/camera index。
  - `reload_scene` 失败时抛错，不污染现有 renderer 状态。
  - `snapshot_cameras` 与 `reload_scene` 共享控制锁。
- `apps/python/tests/test_zapdos_import.py`
  - `_swap_runtime_bundle()` 在 reload 成功时复用同一个 renderer，不再构造新 `IsaacRenderer`。
  - `_swap_runtime_bundle()` 在 reload 失败时自动回退到重启路径。
  - 回退成功时只关闭旧 renderer；回退失败时保持旧 session 状态不被破坏。
- renderer 侧单元测试可落在新的小 helper 上，而不是直接起 IsaacSim：
  - subscriber cache reset
  - camera topology compatibility check
  - annotator detach/destroy/recreate 顺序
- 手工验收
  - 记录 `/api/isaac?id=<renderer_id>` 的 `pid`
  - 调用 `add_asset_to_scene`
  - 验证 `pid` 不变、画面恢复、日志非空且没有新进程启动痕迹
  - 再人为制造 reload 失败，验证自动回退后功能仍成功

## Assumptions
- 这次只覆盖 overlay-only 场景：`add_asset_to_scene` / `remove_asset_from_scene` 引发的同 session scene 重建。
- 允许 reload 期间短暂卡顿或返回占位帧，不要求无缝连续帧。
- 若 camera 拓扑、分辨率、`num_envs`、robot 结构发生变化，直接判为不兼容并回退到完整重启。
- 默认不加 feature flag；先以内建“复用优先，失败自动重启”方式落地。
