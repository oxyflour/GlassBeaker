根据 `tasks/19-train-nijika.md` 继续迭代，实时记录本轮新的增量进展。

Progress update 2026-05-02 (staged-loss topology recheck started)

- 当前主 baseline:
  - run: `tmp/nijika-staged-loss-6600`
  - model: `structured_pair_spectral_head`
  - best epoch: `268`
  - validation `dB MAE`: `4.49499 dB`
- 本轮假设：
  - `pair_topology` 之前在旧 loss 下直接注入 pair head 没有收益
  - 但现在 staged loss 已经把优化目标对齐到 coupling / deep-notch `dB` 误差，拓扑特征有可能只在这个新目标下才体现价值
- 本轮最小实验：
  - 先不发散到新 decoder 设计，直接复用现有 `structured_pair_topology_spectral_head`
  - 只改训练配方，给它套用和当前最佳 run 相同的 staged loss schedule
  - 先跑 `6600` 样本 `50` epoch 公平 pilot，再决定是否值得上更大预算
- gate:
  - 主要比较对象：staged-loss baseline @ epoch `50` 的 `5.17867 dB`
  - 如果 topology 分支在同预算下没有优于这个数字，就停止这条路线
- running command:
  - `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-staged-loss-pair-topo-6600-pilot50 --model-kind structured_pair_topology_spectral_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --db-weight-final 0.1 --coupling-weight-final 1.5 --notch-weight-final 2.0 --notch-threshold-db -10 --loss-ramp-start-epoch 10 --loss-ramp-end-epoch 35`
- 状态：
  - `started`

Progress update 2026-05-02 (staged-loss pole/residue pilot nearly matched the gate)

- `6600` 样本上的 `50` epoch staged-loss pole/residue pilot 已完成：
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-staged-loss-pole-6600-pilot50 --model-kind structured_pair_pole_residue_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --num-poles 12 --db-weight-final 0.1 --coupling-weight-final 1.5 --notch-weight-final 2.0 --notch-threshold-db -10 --loss-ramp-start-epoch 10 --loss-ramp-end-epoch 35`
  - best epoch: `49`
  - validation RMSE: `0.12766`
  - validation `dB MAE`: `5.23270 dB`
  - validation `dB RMSE`: `8.17741 dB`
- 和 staged-loss spectral baseline @ epoch `50`（`5.17867 dB`）对比：
  - pole/residue 更差 `0.05403 dB`
  - 相对回退约 `1.04%`
  - 结论：没有正式过 gate，但已经非常接近，不像 topology 注入那样明显失败

Progress update 2026-05-02 (independent analysis for staged-loss pole/residue pilot)

- 独立 analysis：
  - command: `uv run --project apps/python python packages/nijika/analyze_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --model-path tmp/nijika-staged-loss-pole-6600-pilot50/baseline_model.pt --split val --batch-size 256 --output-dir tmp/nijika-staged-loss-pole-6600-pilot50-analysis`
  - artifacts:
    - summary: `tmp/nijika-staged-loss-pole-6600-pilot50-analysis/val_summary.json`
    - analysis: `tmp/nijika-staged-loss-pole-6600-pilot50-analysis/analysis.json`
- 相对 staged-loss spectral baseline pilot（`tmp/nijika-staged-loss-6600-pilot50-analysis`）的关键对比：
  - sample mean `dB MAE`: `5.17867` -> `5.23270 dB`
  - sample median `dB MAE`: `4.79105` -> `4.85248 dB`
  - sample p90 `dB MAE`: `6.66116` -> `6.92151 dB`
  - worst sample `dB MAE`: `17.34091` -> `16.77218 dB`
  - reflection `dB MAE`:
    - `S11`: `1.0385` -> `1.0480 dB`
    - `S22`: `1.0400` -> `1.0491 dB`
    - `S33`: `1.0301` -> `1.0392 dB`
  - coupling `dB MAE`:
    - `S12`: `7.1797` -> `7.2739 dB`
    - `S13`: `7.4090` -> `7.4366 dB`
    - `S23`: `7.1609` -> `7.2684 dB`
  - notch 区域：
    - truth `< -10 dB`: `7.3443` -> `7.4168 dB`
    - truth `< -20 dB`: `7.0736` -> `7.1167 dB`
- 分桶观察：
  - pole/residue 在 `1320` 个 validation 样本里有 `551` 个样本优于 baseline（`41.74%`）
  - 它不是像 topology 一样“大面积回退”，而是更接近一种高方差近似持平：
    - `1` cut: `+0.1036 dB`
    - `2` cuts: `+0.0330 dB`
    - `3` cuts: `+0.0973 dB`
    - `4` cuts: `-0.0146 dB`
  - 说明它在高复杂度样本上有轻微潜力，但当前实现整体仍略输给 spectral baseline

Current conclusion 2026-05-02 (pole/residue line)

- `structured_pair_pole_residue_head` 在 staged loss 下比它历史旧成绩强很多，说明这条线不是死路
- 但“原样复活”版本仍未超过 staged-loss spectral baseline，所以现在还不适合升为主 baseline
- 和 topology 注入不同，这条线保留一个很窄的后续入口：
  - 因为它和 gate 的差距只有 `0.054 dB`
  - 且在 `4` cuts` 桶已经接近持平
- 如果继续，下一步应该只做一次受控小改动再 gate，而不是直接上大训练：
  - 优先候选 1：给 shared poles 加 `pair-specific pole offset`
  - 优先候选 2：把一阶 complex pole 改成更贴近 notch 形状的 second-order resonator
  - 两者都应继续沿用当前 staged loss，并先做 `50` epoch pilot

Progress update 2026-05-02 (staged-loss pair-topology pilot failed gate)

- `6600` 样本上的 `50` epoch staged-loss topology pilot 已完成：
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-staged-loss-pair-topo-6600-pilot50 --model-kind structured_pair_topology_spectral_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --db-weight-final 0.1 --coupling-weight-final 1.5 --notch-weight-final 2.0 --notch-threshold-db -10 --loss-ramp-start-epoch 10 --loss-ramp-end-epoch 35`
  - best epoch: `48`
  - validation RMSE: `0.14600`
  - validation `dB MAE`: `5.75789 dB`
  - validation `dB RMSE`: `8.94129 dB`
- 和 staged-loss baseline @ epoch `50`（`5.17867 dB`）对比：
  - topology pilot 更差 `0.57923 dB`
  - 相对回退约 `11.18%`
  - 结论：这条 topology 注入路线在 staged loss 下仍然过不了公平 gate，不值得继续上 `300` epoch

Progress update 2026-05-02 (analysis path fixed for graph/topology-aware models)

- 发现一个分析工具缺口：
  - `packages/nijika/baseline/analyze.py` 之前直接调用 `model(...)`
  - 对 `structured_pair_topology_spectral_head` / graph 类模型不会自动传 `pair_topology` 或其他 graph tensor
  - 这会让独立 analysis 无法覆盖这些实验分支
- 修复：
  - `packages/nijika/baseline/analyze.py`
    - 改为复用 `baseline.training_utils.forward_model()`
    - 当模型声明 `uses_graph_features=True` 时，自动按需传入 graph/topology tensor
  - 新增回归测试：`apps/python/tests/test_nijika_analyze.py`
- 验证：
  - command: `uv run --project apps/python python -m unittest apps.python.tests.test_nijika_analyze`
  - 结果：`OK`

Progress update 2026-05-02 (independent analysis confirms topology branch is broadly worse)

- 独立 analysis：
  - command: `uv run --project apps/python python packages/nijika/analyze_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --model-path tmp/nijika-staged-loss-pair-topo-6600-pilot50/baseline_model.pt --split val --batch-size 256 --output-dir tmp/nijika-staged-loss-pair-topo-6600-pilot50-analysis`
  - artifacts:
    - summary: `tmp/nijika-staged-loss-pair-topo-6600-pilot50-analysis/val_summary.json`
    - analysis: `tmp/nijika-staged-loss-pair-topo-6600-pilot50-analysis/analysis.json`
- 相对 staged-loss baseline pilot（`tmp/nijika-staged-loss-6600-pilot50-analysis`）的关键对比：
  - sample mean `dB MAE`: `5.17867` -> `5.75789 dB`
  - sample median `dB MAE`: `4.79105` -> `5.10805 dB`
  - sample p90 `dB MAE`: `6.66116` -> `8.45781 dB`
  - worst sample `dB MAE`: `17.34091` -> `15.63461 dB`
  - reflection `dB MAE`:
    - `S11`: `1.0385` -> `1.1854 dB`
    - `S22`: `1.0400` -> `1.1885 dB`
    - `S33`: `1.0301` -> `1.1868 dB`
  - coupling `dB MAE`:
    - `S12`: `7.1797` -> `7.9772 dB`
    - `S13`: `7.4090` -> `8.1682 dB`
    - `S23`: `7.1609` -> `7.9848 dB`
  - notch 区域：
    - truth `< -10 dB`: `7.3443` -> `8.0580 dB`
    - truth `< -20 dB`: `7.0736` -> `7.5829 dB`
- 分桶观察：
  - topology 分支在所有 cut-count 桶都更差，只是复杂样本上的回退略小：
    - `1` cut: `+0.7962 dB`
    - `2` cuts: `+0.6587 dB`
    - `3` cuts: `+0.5080 dB`
    - `4` cuts: `+0.3672 dB`
  - 在 `1320` 个 validation 样本里，topology 只在 `266` 个样本上优于 baseline（`20.15%`）
  - 说明它不是“整体持平、只被少数坏点拖累”，而是大范围回退，只在少量局部样本上有收益

Current conclusion 2026-05-02

- staged loss 仍然是当前最核心的增益来源
- 把 `pair_topology` 直接注入现有 pair head，即使放到 staged-loss baseline 上，也仍然不成立
- topology 分支唯一留下来的信号是：
  - 最坏样本尾部略好
  - 高 cut-count 样本上的回退略小
- 更合理的下一步优先级应改成：
  - 不再继续这条“直接 topology 注入”的线
  - 优先做更细的 error slicing / sample clustering，找出那 `20%` 真正受益的子类是否具有稳定结构特征
  - 或者回到 decoder 侧，但要避免大改 encoder，而是围绕 staged-loss baseline 做更受控的局部解码增强

Progress update 2026-05-02 (staged-loss pole/residue revival started)

- 背景：
  - 历史上最好的 `pole/residue` 配置是 `num_poles=12`
  - 它在旧 baseline 上曾经略优于 spectral head 的 `RMSE / dB RMSE`，但输在主指标 `dB MAE`
  - 这条线还没有在当前 staged-loss 目标下复测过
- 本轮最小实验：
  - 不先改 `structured_pair_pole_residue_head` 实现
  - 直接复用当前架构，套用和主 baseline 相同的 staged loss schedule
  - 先跑 `6600` 样本 `50` epoch pilot
- gate:
  - 主要比较对象：staged-loss spectral baseline @ epoch `50` 的 `5.17867 dB`
  - 如果 `pole/residue` 在同预算下不能优于这个数字，就先停止这条“原样复活”路线
- running command:
  - `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-staged-loss-pole-6600-pilot50 --model-kind structured_pair_pole_residue_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --num-poles 12 --db-weight-final 0.1 --coupling-weight-final 1.5 --notch-weight-final 2.0 --notch-threshold-db -10 --loss-ramp-start-epoch 10 --loss-ramp-end-epoch 35`
- 状态：
  - `started`

Progress update 2026-05-04 (pair-specific pole offset pilot implemented and evaluated)

- 先做了比直接开新大实验更便宜的一步：
  - 基于前面的 slicing，判断 `structured_pair_pole_residue_head` 的主要问题更像是 “shared poles 不够灵活”，而不是整条 pole/residue 参数化方向错误
  - 因此没有改成更重的 resonator 结构，而是先加一个零初始化的 `pair-specific pole offset`
- 代码改动：
  - `packages/nijika/baseline/structured_pole_model.py`
    - 新增 `use_pair_pole_offsets`
    - 新增 `pair_pole_offset_head`
    - 默认零初始化，初始行为严格退化回 shared poles
    - 每个 pair 只允许对 shared pole 的 damping / omega 做小幅受限偏移
  - `packages/nijika/baseline/model.py`
    - 新增 `model_kind=structured_pair_pole_offset_residue_head`
  - `packages/nijika/baseline/train.py`
    - train CLI 接受新的 `model_kind`
  - `apps/python/tests/test_nijika_pole_model.py`
    - 回归测试：
      - 新 model kind 初始时等价于 shared poles
      - 不同 pair latent 可以只移动对应 pair 的 poles，而不扰动未激活的 pair
- 验证：
  - unit tests:
    - `uv run --project apps/python python -m unittest apps.python.tests.test_nijika_pole_model apps.python.tests.test_nijika_analyze`
    - 结果：`OK`
  - smoke train:
    - `uv run --project apps/python python packages/nijika/run_baseline.py --dataset-root C:\Projects\GlassBeaker\tmp\antenna-dataset-smoke --output-dir tmp\nijika-pole-offset-smoke --model-kind structured_pair_pole_offset_residue_head --epochs 1 --batch-size 8 --hidden-dim 64 --num-poles 12`
  - smoke predict:
    - `uv run --project apps/python python packages/nijika/predict_baseline.py --dataset-root C:\Projects\GlassBeaker\tmp\antenna-dataset-smoke --model-path tmp\nijika-pole-offset-smoke\baseline_model.pt --sample-name antenna_000 --output-dir tmp\nijika-pole-offset-predict-smoke`
- `6600` 样本上的 `50` epoch staged-loss pilot：
  - command:
    - `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root C:\Projects\GlassBeaker\tmp\antenna-dataset-2400-v2 --output-dir tmp\nijika-staged-loss-pole-offset-6600-pilot50 --model-kind structured_pair_pole_offset_residue_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10 --num-poles 12 --db-weight-final 0.1 --coupling-weight-final 1.5 --notch-weight-final 2.0 --notch-threshold-db -10 --loss-ramp-start-epoch 10 --loss-ramp-end-epoch 35`
  - best epoch: `49`
  - validation RMSE: `0.12795`
  - validation `dB MAE`: `5.21926 dB`
  - validation `dB RMSE`: `8.17522 dB`
- 和 staged-loss spectral baseline @ epoch `50`（`5.17867 dB`）对比：
  - pair-pole-offset 更差 `0.04060 dB`
  - 相对回退约 `0.78%`
  - 结论：仍然没有正式过 gate，但已经把原始 pole/residue 的差距从 `0.05403 dB` 收窄到了 `0.04060 dB`
- 和原始 staged-loss pole/residue pilot（`5.23270 dB`）对比：
  - pair-pole-offset 改善 `0.01343 dB`
  - 在 `1320` 个 validation 样本里，pair-pole-offset 有 `662` 个样本优于旧 pole 版本（`50.15%`）
- 独立 analysis：
  - command:
    - `uv run --project apps/python python packages/nijika/analyze_baseline.py --dataset-root C:\Projects\GlassBeaker\tmp\antenna-dataset-2400-v2 --model-path tmp\nijika-staged-loss-pole-offset-6600-pilot50\baseline_model.pt --split val --batch-size 256 --output-dir tmp\nijika-staged-loss-pole-offset-6600-pilot50-analysis`
  - 相对 staged-loss spectral baseline pilot 的关键对比：
    - sample mean `dB MAE`: `5.17867` -> `5.21926 dB`
    - sample median `dB MAE`: `4.79105` -> `4.80554 dB`
    - sample p90 `dB MAE`: `6.66116` -> `6.68828 dB`
    - worst sample `dB MAE`: `17.34091` -> `19.84004 dB`
    - reflection `dB MAE`:
      - `S11`: `1.0385` -> `1.0493 dB`
      - `S22`: `1.0400` -> `1.0525 dB`
      - `S33`: `1.0301` -> `1.0342 dB`
    - coupling `dB MAE`:
      - `S12`: `7.1797` -> `7.2612 dB`
      - `S13`: `7.4090` -> `7.4094 dB`
      - `S23`: `7.1609` -> `7.2481 dB`
    - notch 区域：
      - truth `< -10 dB`: `7.3443` -> `7.3933 dB`
      - truth `< -20 dB`: `7.0736` -> `7.1191 dB`
  - 相对原始 staged-loss pole/residue pilot：
    - mean / median / p90 都有小幅改善
    - 但 worst tail 明显变差，说明 pair offset 虽然提升了中位样本和多数样本的拟合灵活度，也更容易放大少数坏点
  - 相对 baseline 的分桶观察：
    - `1` cut: `+0.1264 dB`
    - `2` cuts: `+0.0051 dB`
    - `3` cuts: `+0.0211 dB`
    - `4` cuts: `+0.0097 dB`
    - 说明这次改动把旧 pole 版本的 gap 普遍收窄了，但还没有像最初预期那样在高 cut-count 桶里稳定翻正

Current conclusion 2026-05-04 (pair-specific pole offset)

- 这条线比“原样复活 pole/residue”又向前推了一小步，但还没有过 staged-loss spectral baseline 的公平 gate
- `pair-specific pole offset` 这个方向本身不是坏信号：
  - 它确实把总体 gap 从 `0.054 dB` 缩到 `0.041 dB`
  - 并且相对旧 pole 版本，已经是 `50%+` 样本受益
- 但当前最明显的新问题是 tail 变坏：
  - worst sample 从 `16.77 dB` 放大到 `19.84 dB`
  - 说明现在的 offset 自由度还缺少约束，不适合直接继续放大
- 如果继续，这条线的下一步应该优先做“约束更强”的小改动，而不是更大的 decoder 改写：
  - 优先候选 1：只允许 coupling pairs 使用 pole offset，reflection pairs 保持 shared poles
  - 优先候选 2：给 pole offset 加显式 regularization / scale schedule，先压 tail 再看是否能过 gate
  - 这两条都应该继续沿用当前 staged loss，并先做 `50` epoch pilot
