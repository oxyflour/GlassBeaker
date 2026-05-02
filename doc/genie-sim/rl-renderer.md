# RL Renderer 当前状态

更新时间：2026-05-02

## 目标

当前方案的目标是把：

- MuJoCo 作为机器人动力学与控制执行端
- Isaac RL renderer 作为图像渲染端
- ROS2 作为两者之间的消息边界

串成一条最小可维护链路，并且不修改 `deps/genie_sim` 上游源码。

## 当前调用链

### 1. 前端入口

`apps/web/app/demo/zapdos/page.tsx`

- 用 `GET /python/zapdos/{session}/call/start` 建立 SSE
- 用 `POST /python/zapdos/{session}/call/get_visual` 拉取 MuJoCo 几何
- 用 `GET /python/zapdos/{session}/render/{camera_name}` 读取 Isaac MJPEG
  - 例如：`/python/zapdos/{session}/render/head_camera`
  - `camera_name` 必须精确等于 MuJoCo 里的真实 camera 名；不存在时返回 `404`

### 2. Python API 入口

`apps/python/api/zapdos/{session}/{action}.py`

会话首次创建时：

1. 解析 `robot_usd` / `scene_usd`
2. 调用 `ensure_render_bundle(...)`
3. 用 bundle 里的 `sim_scene.xml` 启动 MuJoCo
4. 启动 ROS worker
5. 启动 Isaac renderer
6. 启动后台循环，持续发布：
   - `/env_0/joint_states`
   - `/env_0/tf_render`
   - `/env_0/<camera_name>/image_raw`

控制输入边界固定为：

- `/env_0/joint_command`
- `sensor_msgs/msg/JointState`

### 3. Bundle 编译层

`apps/python/utils/rl_bundle.py`
`apps/python/utils/rl_bundle_stage.py`

输入：

- `deps/galaxea/object/r1pro/r1pro.usda`
- 某个静态 `scene.usda`

输出到 `apps/python/tmp/rl_bundles/<hash>/`：

- `sim_scene.xml`
- `sim_scene.usda`
- `scene_render.usda`
- `robot_wrapper.usda`
- `render_scene.usda`
- `render_scene_body_map.json`
- `render_scene_body_map.jsona`
- `manifest.json`

说明：

- `sim_scene.xml` 给 MuJoCo 用
- `render_scene.usda` 给 Isaac RL renderer 用
- `robot_wrapper.usda` 是扁平 per-link wrapper
- `body_map.json` / `jsona` 是给上游 body map 推导缺陷做兼容

### 4. Isaac 进程编排层

`apps/python/utils/sim_env.py`

职责：

- 通过 `apps/web/app/api/isaac/route.ts` 管理 Isaac 子进程
- 配置 Isaac Python 环境变量
- 绑定 SHM
- 读取 Isaac 输出帧

注意：

- `apps/python` 侧现在不直接 import `isaacsim`
- 真正需要 Isaac 环境执行的本地入口已经移到 `apps/isaac`

### 5. Isaac 本地入口

`apps/isaac/rl_renderer_entry.py`

职责：

- 在 `apps/isaac/.venv/Scripts/python.exe` 下执行
- 导入上游 `geniesim.rl.renderer.rl_renderer`
- 做本地运行时适配，不改上游文件

当前适配点：

- 把 `SingleThreadedExecutor.spin` 改成 no-op
- 在主循环中显式 `spin_once(timeout_sec=0.0)`
- 优先通过 `--cameras-json` 配置多 camera；只有显式传 `/default_viz_camera` 时才复用现有默认 camera

### 6. ROS bridge

Python 侧 websocket bridge：

- `apps/python/api/ros.py`
- `apps/python/utils/ros_bridge.py`

ROS worker 侧：

- `apps/python/utils/ros_worker.py`
- `apps/ros/app.py`

职责：

- Python API 进程通过 websocket 调 ROS worker
- ROS worker 执行真实 ROS2 publish / subscribe

## 当前目录职责边界

### `apps/python`

保留：

- FastAPI 路由
- MuJoCo 会话与控制
- bundle 编译
- ROS bridge 编排
- Isaac 进程编排

不应放：

- 必须在 Isaac Python 环境下运行的入口脚本

### `apps/isaac`

保留：

- IsaacSim 运行环境
- Isaac 本地入口脚本
- 未来任何必须 `import isaacsim` / `SimulationApp` 的本地代码

## 已完成修复

### 1. 机器人姿态错误

根因已经确认并修复。

问题来源不是 TF 本身，而是 bundle 生成的几个 USDA stage 没有显式继承 stage metadata，导致：

- 新建 stage 默认回退成 `Y-up`
- `usd_to_mjcf.py` 读取后额外插入 `usd_stage_root` 旋转补偿
- MuJoCo 与 Isaac wrapper 的世界位姿整体错轴

修复方式：

- 在 `apps/python/utils/rl_bundle_stage.py` 中统一写入 scene 的：
  - `upAxis`
  - `metersPerUnit`
- bump `BUNDLE_VERSION` 强制重编 bundle

结果：

- Isaac 画面里的 R1Pro 已恢复直立
- `apps/python/tests/test_rl_bundle.py` 已覆盖这个回归点

### 2. Isaac 运行时入口迁移

之前本地 renderer 入口放在 `apps/python/scripts/rl_renderer_entry.py`。

现状：

- 已迁到 `apps/isaac/rl_renderer_entry.py`
- `apps/python/utils/sim_env.py` 已改成从 `apps/isaac` 启动

### 3. 默认场景已清空

`apps/python/assets/default_scene.usda` 当前只保留空的 `World` 根节点。

之前用于调试的两个 cube 已删除：

- `Floor`
- `MarkerCube`

因此当前默认场景不再自带地面或 marker。

## 当前接口与约束

### HTTP

- `GET/POST /python/zapdos/{session}/{action}/{name}`
- `GET /api/isaac?id=...`
- `POST /api/isaac`
- `DELETE /api/isaac`

### ROS

- 输入：`/env_0/joint_command` `sensor_msgs/msg/JointState`
- 输出：`/env_0/joint_states` `sensor_msgs/msg/JointState`
- 输出：`/env_0/<camera_name>/image_raw` `sensor_msgs/msg/Image`
- 内部：`/env_0/tf_render` `tf2_msgs/msg/TFMessage`

### 运行约束

- v1 只支持静态 scene
- 动态同步只覆盖机器人 body，不覆盖 scene object
- 不修改 `deps/genie_sim`
- `apps/python/tmp/rl_bundles` 视为缓存，不手改

## 当前验证状态

已验证：

- `ensure_render_bundle(...)` 能产出完整 bundle
- `apps/python/tests/test_rl_bundle.py` 通过
- MuJoCo -> TF -> Isaac 的姿态同步已恢复正确
- Isaac MJPEG 能经 `render/head_camera` 等真实 camera 名输出

调试阶段生成过的对照图位于：

- `apps/python/tmp/rl_debug/mujoco_pose_matched.png`
- `apps/python/tmp/rl_debug/isaac_pose.png`

这些文件是调试产物，不属于正式接口。

## 已知限制 / 技术债

- `ZapdosSession.get_camera()` 当前前端没有消费
- `call/subscribe` 路径当前 demo 没有实际用例
- SSE 中的 `camera` 字段当前前端没有实际使用
- MJPEG URL 与 ROS image topic 都要求精确 camera 名，当前不提供别名或兼容 `render/main`
- `apps/python/utils/mujoco_tools.py` 下半部分仍有旧链路残留 helper，可后续拆理
- `apps/python/tmp/**` 下有大量调试脚本、旧 bundle、截图、日志，适合后续清理
- `deps/galaxea/object/r1pro/r1pro.xml` 和 `out.xml` 看起来已不在当前主链上

## 下一轮迭代建议

优先级建议：

1. 清理不再使用的调试产物与旧辅助代码
2. 收紧 `zapdos` API，只保留当前前端真实使用的接口
3. 视需要补 `camera_info` / depth / compressed image
4. 视需要扩展 scene object 的动态同步，而不只是机器人本体
5. 把更多 Isaac-only 代码继续约束在 `apps/isaac`

## 后续修改时的判断标准

如果某段代码满足以下任一条件，应优先放在 `apps/isaac`：

- 直接 `import isaacsim`
- 依赖 `SimulationApp`
- 必须用 `apps/isaac/.venv` 执行
- 本质上是 Isaac renderer 的本地运行时适配

否则优先放在 `apps/python`：

- bundle 编译
- MuJoCo 编排
- FastAPI 路由
- ROS bridge 编排
- 进程管理与 SHM 管理
