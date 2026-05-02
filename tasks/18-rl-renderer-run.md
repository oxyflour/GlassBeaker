# genie_sim 集成方案（R1Pro + MuJoCo + Isaac RL Renderer）

## Summary

- 采用 `genie_sim` 的 RL renderer 轻量链路，不走 `genie_sim app` 的 articulation 主链路。
- 输入固定为两个 USDA：机器人本体 `deps/galaxea/object/r1pro/r1pro.usda` 和一个静态场景 `scene.usda`。
- 启动前先编译两套产物：
  - 仿真产物：机器人与场景组合后的 MJCF，供 MuJoCo 仿真。
  - 渲染产物：机器人扁平 wrapper USDA、静态场景 USDA、组合渲染场景 USDA、`body_map.json` 与 `body_map.jsona`，供 Isaac RL renderer。
- 3D 鼠标的正式输入边界固定为 ROS2 `sensor_msgs/msg/JointState` `/env_0/joint_command`。
- 该方案下，关节状态和渲染图像都发布到 ROS：`/env_0/joint_states` 与 `/env_0/main_camera/image_raw`；`/env_0/tf_render` 保留为 renderer 消费的内部/调试 topic。

## Implementation Changes

- 在 `apps/python/utils` 新增一层 bundle 编译器，负责把 `robot_usda + scene_usda` 编译成运行清单。清单至少包含 `mjcf`、`render_scene_usd[a]`、`robot_wrapper_usd[a]`、`body_map_json`、`body_map_jsona`、默认相机 prim。
- 复用现有 `usd_to_mjcf.py` 生成 MuJoCo 侧 XML，但输入改为机器人和静态场景的组合 stage，不再只把原始 `r1pro.usda` 单独转 MJCF。
- 机器人渲染 wrapper 必须是扁平 per-link 结构：每个 MuJoCo body 对应一个可直接写 `xformOp:translate/orient` 的 prim，命名直接对齐 body 名，例如 `Root_r1_pro_with_gripper_<body>`；wrapper 内不保留 articulation、joint、physics 求解依赖。
- wrapper 的每个 link prim 只引用原始 `r1pro.usda` 对应 link 的 `visuals`，并禁用会干扰默认视角的嵌入相机；材质保持 best-effort，允许继续出现当前日志里的 scope 外 material warning，不阻塞 v1。
- 场景渲染 USDA 只保留静态可视内容、灯光和选定主相机；v1 不为场景中的物体建立动态 body map，也不让 Isaac 侧参与任何场景物理。
- `apps/python/utils/sim_env.py` 的 `IsaacRenderer` 改为启动本地 renderer 入口脚本，而不是直接把原始 `r1pro.usda` 喂给上游 `rl_renderer.py`。
- 本地 renderer 入口脚本放在受版本控制的 `apps/python` 侧，用 `apps/isaac/.venv/Scripts/python.exe` 执行；它导入上游 `geniesim.rl.renderer.rl_renderer`，只做运行时适配，不修改 `deps/genie_sim` 文件。
- 运行时适配固定包含两点：ROS executor 在主循环中显式 `spin_once`；生成 `body_map.json` 和 `body_map.jsona` 两份 sidecar，兼容上游对 `.usda` 的路径推导缺陷。
- `apps/python/api/zapdos/{session}/{action}.py` 改为从 bundle 清单启动整条链路：编译资产、启动 MuJoCo、启动 Isaac renderer、启动 ROS 图像桥，并在 session 销毁时统一清理子进程和 SHM。
- 现有 3D 鼠标或控制端不再直接碰 Isaac；它只需要发布 `/env_0/joint_command`，MuJoCo 更新后由 `/env_0/tf_render` 驱动 Isaac 画面同步。

## Public Interfaces

- 新的运行输入从单个 robot USD 改为 `robot_usd + scene_usd + runtime options`。
- 新的 bundle 清单是显式接口，后续所有运行入口都只消费清单，不再各自推导路径。
- ROS topic 契约固定为：
  - 输入：`/env_0/joint_command` `sensor_msgs/msg/JointState`
  - 输出：`/env_0/joint_states` `sensor_msgs/msg/JointState`
  - 输出：`/env_0/main_camera/image_raw` `sensor_msgs/msg/Image`
  - 内部/调试：`/env_0/tf_render` `tf2_msgs/msg/TFMessage`
- `camera_info`、depth、compressed image 不进 v1 接口；需要时再在图像桥上追加。

## Test Plan

- 编译测试：给定 `r1pro.usda + scene.usda` 能稳定生成 MJCF、wrapper USDA、组合场景 USDA、`body_map.json`、`body_map.jsona`，且 body map 覆盖 MuJoCo 全部机器人 body，除 `world`。
- 运行测试：headless 启动整条链路后，MuJoCo 正常接收 `/env_0/joint_command`，`/env_0/joint_states` 中被控关节出现随时间变化的 position。
- 渲染同步测试：在持续关节动作下，renderer SHM 连续两帧和 `/env_0/main_camera/image_raw` 连续两帧都出现非噪声级差异；接受标准沿用当前经验阈值即可，不要求像素级一致。
- ROS 发布测试：`/env_0/joint_states` 和 `/env_0/main_camera/image_raw` 都可被外部 ROS 节点稳定订阅；图像发布频率接近当前 `render_hz`。
- 清理测试：session 结束或程序退出后，Isaac renderer 子进程、MuJoCo 进程、SHM、ROS 图像桥都被释放，不残留僵尸进程。
- 约束测试：`git -C deps/genie_sim status --short` 为空，确认全方案不依赖修改上游源码。

## Assumptions

- v1 只支持静态场景；动态同步仅覆盖机器人本体。
- 3D 鼠标最终通过 ROS `JointState` 驱动，不在仿真栈内做设备专用适配。
- 新的可维护源码放在 `apps/python`，`apps/isaac` 只作为 Isaac 运行环境和工作目录。
- 继续使用现有 `apps/python` 的 `uv` 环境做编译/编排，继续使用 `apps/isaac` 的 `uv` 环境执行 IsaacSim。
- 不修改 `deps/genie_sim`；如上游缺陷仍存在，全部通过本地入口脚本和外层生成资产规避。
