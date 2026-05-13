# 给 `deps/genie_sim` 增加 `r1` 机器人

## 目标

- 在 `deps/genie_sim` 中新增机器人类型 `r1`
- 使用 `deps/galaxea/object/r1pro/r1_pro_with_gripper.urdf`
- 使用 `deps/galaxea/object/r1pro/r1pro_backup.usda`

## 进度

- [x] 定位 `deps/genie_sim` 中机器人类型与配置入口
- [x] 梳理 `r1` 需要补齐的资源与代码路径
- [x] 实现 `r1` 配置与注册
- [x] 验证配置可被加载

## 过程记录

- 2026-04-23 23:56: 初步确认 `teleop/config/robot_interface.py`、`teleop/utils/ros_nodes.py`、`geniesim/app/robot_cfg/*` 是主要接入点。
- 2026-04-23 23:56: 确认任务文件原本为空，开始在此处持续记录进度。
- 2026-04-24 00:02: 确认 `geniesim/app/controllers/api_core.py` 与 `geniesim/app/controllers/kinematics_solver.py` 当前只支持从 `genie_sim` 自身目录拼接资产路径，无法直接加载 `deps/galaxea` 下的绝对路径资源。
- 2026-04-24 00:04: 确认 `benchmark/pi_env/abs_pose_env` 还依赖 `IK-SDK` 的 G1/G2 模型，`r1` 若要进入策略控制链，需要额外补齐 IK 模型，不属于本次最小接入闭环。
- 2026-04-24 00:05: 当前实现目标收敛为：让 `genie_sim` 主程序可以通过新增 `R1` robot config 加载 `deps/galaxea` 中的 URDF/USD 资产，同时把相关硬编码分支改到不阻塞 `R1` 加载。
- 2026-04-24 00:12: 在 `geniesim/utils/system_utils.py` 增加通用路径解析函数，让 robot config 可以同时兼容 `genie_sim` 内部资产路径和外部相对路径。
- 2026-04-24 00:14: 在 `geniesim/app/utils/robot.py` 中放开 `R1` 机器人名校验；在 `kinematics_solver.py` 与 `api_core.py` 中切换到基于 config 目录的路径解析。
- 2026-04-24 00:16: 新增 `geniesim/app/robot_cfg/R1.json`，URDF 使用 `deps/galaxea/object/r1pro/r1_pro_with_gripper.urdf`，运动学描述使用 `deps/galaxea/config/r1pro_rmpflow/r1pro_descriptor.yaml`。
- 2026-04-24 00:18: 发现 `r1pro_backup.usda` 的 `defaultPrim` 为 `Root`，直接挂载会多一层 prim，因此新增 `geniesim/app/robot_cfg/R1.usda` 作为薄包装，把 `</Root/r1_pro_with_gripper>` 映射为 `/R1`。
- 2026-04-24 00:20: 调整 `geniesim/app/ros_publisher/robot_interface.py`，TF 构建改为优先使用 articulation 自身的 DOF 名判定动态关节，并跳过不存在的 G2 专用 prim。
- 2026-04-24 00:22: 轻量验证通过：`R1.json` 中引用的 URDF、descriptor、wrapper USDA 均可解析到真实文件；`uv run python -m py_compile` 已通过；`RobotCfg(R1.json)` 已可实例化并识别为 `R1`。

## 当前限制

- 本次没有补齐 `benchmark/pi_env/abs_pose_env` 所需的 `R1` IK 模型，原因是现有 `IK-SDK` 只内置 G1/G2 资源。
- `R1` 当前接入重点是 `genie_sim app` 资产加载链，不等同于完整 benchmark/策略控制闭环。
