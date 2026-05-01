# 任务

把 `deps\galaxea\object\r1pro\r1pro.usda` 转成 `genie_sim` 的 `rl_renderer.py` 可用的机器人 USDA，让 RL 渲染链路里机器人在 Isaac 画面和 ROS 图像里能明显动起来。

## 当前结论

- `ros_image_publisher.py` 不是问题。它只是把 RL renderer 的 SHM 帧转发成 ROS `sensor_msgs/Image`。
- 关节命令已经到达 MuJoCo，`joint_states` 也确实在变化。
- 但 RL renderer 产出的图像几乎不动，说明问题在 `rl_renderer + robot_usd` 这层，而不是 ROS 发图层。
- `genie_sim` 官方“能 work”的链路并不是任意原始机器人 USD 都能直接喂给 RL renderer。
- 官方 robot config 明显依赖 wrapper/fix 资产，例如 `robot/G2_omnipicker/robot_fix.usda`，而不是原始机器人 USDA。

## 已验证事实

### 1. MuJoCo 侧命令是通的

- `apps/isaac/ros_joint_command_publisher.py` 会往 `/env_0/joint_command` 发 `JointState`
- `deps/genie_sim/source/geniesim/rl/mujoco/mujoco_ros_node.py` 里加了日志
- MuJoCo 日志已确认收到命令，例如：
  - `joint_command[1] ... pos=[0.347..., 0.0]`
  - `joint_command[2] ... pos=[0.348..., 0.0]`
- 订阅 `/env_0/joint_states` 时，`left_arm_joint1` 幅度实测为：
  - `min=-0.35`
  - `max=0.3819`

### 2. ROS 图像和 SHM 原始帧都几乎不动

已保存图像到 `build/genie_r1pro_demo/image_check/`

旧视角：
- `ros_0.png`, `ros_1.png`, `ros_2.png`
- `shm_0.png`, `shm_1.png`, `shm_2.png`

新视角 / 更大动作：
- `ros_new_0.png`, `ros_new_1.png`, `ros_new_2.png`
- `ros_fix_0.png`, `ros_fix_1.png`, `ros_fix_2.png`

结论：
- 图像哈希和像素差证明它们不是完全相同
- 但肉眼看几乎没有明显姿态变化
- 差异更像低幅渲染噪声，而不是明显的关节运动

### 3. body map 不是主要问题

- `apps/isaac/r1pro_demo_scene_body_map.json` 是从 `r1pro.usda` 自动生成的
- MuJoCo body 名与 body map 基本对上：
  - `nbody=37`
  - `matched=36`
  - `unmatched=['world']`

### 4. 我已经试过的方向

- 改 ROS 图像桥：无效，图像问题不在这层
- 改相机视角：无明显改善
- 改 ROS 关节动作幅度：无明显改善
- 改 `rl_renderer.py`，把 world pose 转 parent-local pose 再写 `xformOp`：仍未解决肉眼不动问题

## 对 genie_sim 官方实现的确认

### A. 官方“主链路”是完整 Isaac articulation 链路

关键文件：
- `deps/genie_sim/source/geniesim/app/controllers/api_core.py`
- `deps/genie_sim/source/geniesim/app/ros_publisher/robot_interface.py`

特点：
- 会把机器人初始化成 `SingleArticulation`
- 相机直接挂在 Isaac 里的机器人 / 场景上
- ROS 图像由 Isaac 直接发布

这条链路本身应该是 work 的。

### B. `GenieSimVectorEnv` 走的是另一条“RL 轻量链路”

关键文件：
- `deps/genie_sim/source/geniesim/rl/renderer/rl_renderer.py`
- `deps/genie_sim/source/geniesim/rl/mujoco/mujoco_ros_node.py`

特点：
- Isaac 端不做 articulation 控制
- MuJoCo 发布 `/env_i/tf_render`
- `rl_renderer.py` 订阅后，直接把 body pose 写回 USD prim 的 xform

这条链路默认假设：`robot_usd` 是适合被 per-link xform 直接驱动的 render wrapper。

### C. 官方配置明确偏向 wrapper/fix 资产

关键配置：
- `deps/genie_sim/source/geniesim/app/robot_cfg/G2_omnipicker.json`
- `deps/genie_sim/source/data_collection/config/robot_cfg/G2_omnipicker_fixed_dual.json`

结论：
- 官方不是直接拿原始机器人 USDA 喂 RL renderer
- 官方用的是 `robot_fix.usda` 这类 wrapper 资产

## 当前本地文件

### 新增文件

- `apps/isaac/genie_r1pro_demo.py`
- `apps/isaac/genie_r1pro_demo_support.py`
- `apps/isaac/mujoco_ros_entry.py`
- `apps/isaac/r1pro_demo_scene.usda`
- `apps/isaac/r1pro_demo_scene_body_map.json`
- `apps/isaac/ros_image_publisher.py`
- `apps/isaac/ros_joint_command_publisher.py`

### 当前仍有本地改动的 submodule 文件

- `deps/genie_sim/source/geniesim/rl/mujoco/mujoco_ros_node.py`
  - 加了 `joint_command` 收包调试日志
- `deps/genie_sim/source/geniesim/rl/renderer/rl_renderer.py`
  - 改过 TF 应用逻辑
  - 加过 viewport camera 相关逻辑
  - 这些都是探索性 patch，尚未证明是最终方案

## 运行入口

### 1. SHM 驱动 demo

`apps/python/.venv/Scripts/python.exe apps/isaac/genie_r1pro_demo.py --headless --steps 90`

### 2. ROS joint 驱动 + ROS 图像

`apps/python/.venv/Scripts/python.exe apps/isaac/genie_r1pro_demo.py --drive-mode ros_joint --publish-ros-image --steps 600`

## 下一步建议

不要再花时间怀疑 `ros_image_publisher.py`。

应该把精力放在“从 `deps\galaxea\object\r1pro\r1pro.usda` 生成一个 `rl_renderer` 可用 wrapper USDA”：

1. 对照 `genie_sim` 官方 `robot_fix.usda` / wrapper 资产的结构
2. 为 R1 生成一个 render wrapper：
   - prim 层级尽量与 MuJoCo body map 对齐
   - 每个可见 link 应该能被直接写 `xformOp:translate/orient`
   - 不依赖原始 articulation 在 Isaac 里自行求解关节
3. 然后再让 `rl_renderer.py` 继续走现有 `tf_render -> prim xform` 链路

如果 wrapper 做对了，ROS 图像应该会自然跟着动，不需要再改 ROS 图像桥。


任务说明已经把方向收窄了：重点不在 ROS 发图，而在给 rl_renderer 提供一个能被逐 link xform 驱动的 wrapper USDA。接下来我对照现有 demo 支撑代码和 genie_sim 官方 robot_fix.usda 资产，确认现在缺的是“生成 wrapper”的哪一段。

现在要确认两件事：官方 wrapper 到底长什么样，以及 r1pro.usda 的层级里哪些 prim/mesh 绑定在 rigid body 下。这个对不齐，rl_renderer 就算写对 TF 也只会动“空壳”或局部变换被上层 articulation 抵消。

## 2026-05-01 继续推进后的新结论

上面最后那句“最小实验里，apps/python 这边发 TFMessage，Isaac 的 rclpy 环境一条都收不到”已经被证伪。

### 1. DDS / Isaac rclpy 基础互通不是根因

- 用最小实验验证过两种情况都能收到：
  - `apps/python + pixi ROS` 发布，普通 Isaac Python 订阅
  - `apps/python + pixi ROS` 发布，带 `SimulationApp` 的 Isaac 进程订阅
- `std_msgs/String` 和 `tf2_msgs/TFMessage` 都能收
- 结论：不是“Isaac 自带 rclpy 完全收不到外部 TFMessage”

### 2. 真正卡住 RL renderer 的是两个 `rl_renderer.py` 内部问题

#### A. Windows 下后台 ROS executor 线程会被 `world.step(render=True)` 饿死

- 现象：`MuJoCoRosNode` 单独探测时，`/env_0/tf_render` 每秒稳定发布；但 `renderer.log` 里原来没有任何 `[RLRenderer] tf_msg`
- 修复：把 ROS executor 从后台线程改成主循环里显式 `spin_once(timeout_sec=0.0)`
- 修完后，`renderer.log` 已出现：
  - `[RLRenderer] tf_msg env=0 count=36 ...`

#### B. `scene_usd -> body_map_json` 的 `.usda` 路径拼接有 bug

- 原代码对 `scene_usd` 先 `.replace(".usd", "_body_map.json")`，再 `.replace(".usda", "_body_map.json")`
- 对 `r1pro_demo_scene.usda` 会错误生成 `r1pro_demo_scene_body_map.jsona`
- 结果：`apps/isaac/r1pro_demo_scene_body_map.json` 根本没有被加载
- 后果：renderer 把 MuJoCo body 名原样当成 prim suffix 去找：
  - `/World/envs/env_0/Root_r1_pro_with_gripper_base_link`
  - `/World/envs/env_0/Root_r1_pro_with_gripper_right_gripper_link`
  - 这些 prim 都不存在，所以 `resolved=0/36`
- 修复：改成 `os.path.splitext(self.args.scene_usd)` 再拼 `_body_map.json`
- 修完后，`renderer.log` 已出现：
  - `[RLRenderer] tf_apply env=0 body=Root_r1_pro_with_gripper_right_gripper_link prim=/World/envs/env_0/MyRobot/r1_pro_with_gripper/right_gripper_link ...`

### 3. 当前状态

- `wrapper USDA + body_map sidecar + tf_render -> prim xform` 这条链路现在已经通了
- 不是只有 MuJoCo 在动；renderer 端已经确实收到并写回 prim xform
- 直接读取 renderer SHM 做前后帧对比（counter 11 vs 56）结果：
  - `mean_abs_diff = 1.3898`
  - `max_diff = 103`
  - `changed_pixels_gt10 = 1429`
- 说明渲染帧已经出现非纯噪声级别的可见变化

### 4. 现在更合理的下一步

- `ros_image_publisher.py` 仍然不是重点，不要回头怀疑它
- 如需让“肉眼明显动起来”更显著，优先考虑：
  1. 调更激进的动作幅度 / 持续时间
  2. 调默认视角，让右臂/夹爪占画面更大比例
  3. 清理 `rl_renderer.py` 里现有探索性 debug patch，只保留被证明有效的修复

## 2026-05-02 约束调整后的落地方案

新增约束：

- 不修改 `deps/genie_sim` 源代码
- 应该由工作区外层代码和资产去适配 `genie_sim`

据此，方案改为：

### 1. `genie_sim` 源码已恢复原状

- `deps/genie_sim/source/geniesim/rl/renderer/rl_renderer.py`
- `deps/genie_sim/source/geniesim/rl/mujoco/mujoco_ros_node.py`

当前做法不再依赖改动这两个文件。

### 2. 外层适配改到 `apps/isaac`

关键文件：

- `apps/isaac/r1pro_rl_wrapper.py`
- `apps/isaac/rl_renderer_entry.py`
- `apps/isaac/genie_r1pro_demo_support.py`

具体策略：

1. wrapper USDA 改成扁平 per-link prim
   - 不再保留 `r1_pro_with_gripper/<link>` 这种层级驱动结构
   - 每个刚体 link 都直接变成 `/MyRobot/Root_r1_pro_with_gripper_<link>`
   - 这样可以适配上游 `rl_renderer.py` 的“直接把 TF world pose 写到 prim xform”假设
2. body map 兼容上游 `.usda -> _body_map.jsona` 路径 bug
   - 除了正常生成 `r1pro_demo_scene_body_map.json`
   - 还额外生成 `r1pro_demo_scene_body_map.jsona`
   - 让未改动的上游 `rl_renderer.py` 也能读到外层提供的 map
3. renderer 启动改走本地入口
   - `apps/isaac/rl_renderer_entry.py` 导入上游模块
   - 只在运行时把 ROS executor 改成主循环 `spin_once`
   - 不改上游文件内容

### 3. 当前验证结果

- `git -C deps/genie_sim status --short` 已为空
- `r1pro_rl.usda` 已变成扁平 prim 结构
- `apps/isaac/r1pro_demo_scene_body_map.jsona` 已生成
- 直接读 renderer SHM 的前后帧对比结果：
  - `early_counter = 5`
  - `late_counter = 50`
  - `mean_abs_diff = 4.0361`
  - `max_diff = 96`
  - `changed_pixels_gt10 = 3950`

结论：

- 现在已经满足“外层代码适配 `genie_sim`，不改 `genie_sim` 源码”的约束
- 渲染帧也已经确认发生了明显变化
