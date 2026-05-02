看 `tasks/17-nijika-train.md` 确定下一步优化方向，是继续生成更多数据还是优化模型。实时更新进度到这个文档用于下一次迭代。

Progress update 2026-05-02 (pair-topology hybrid attempt)

- 按 `17` 里的结论，继续尝试“保留 `structured_pair_spectral_head` 主体，只把预计算的 `pair_topology` 注入 pair head”的混合模型路线。
- 代码改动：
  - `packages/nijika/baseline/structured_spectral_model.py`
    - 为 `StructuredSpectralPredictor` 增加 `use_pair_topology` 开关
    - 新增 `model_kind=structured_pair_topology_spectral_head`
    - 模型声明只消费 `pair_topology`，不依赖整套 graph node / edge 特征
  - `packages/nijika/baseline/training_utils.py`
    - `forward_model()` 现在会按模型声明过滤 graph tensor，只把需要的键传进 forward
  - `packages/nijika/baseline/model.py`
    - 暴露 `structured_pair_topology_spectral_head`
  - `packages/nijika/baseline/train.py`
    - train CLI 接受新的 model kind
    - 是否携带 graph 特征改为根据模型能力自动判断，而不是只特判 graph 分支
- 验证：
  - compile: `uv run --project apps/python python -m compileall packages/nijika/baseline packages/nijika/run_baseline.py packages/nijika/predict_baseline.py packages/nijika/analyze_baseline.py`
  - smoke train: `uv run --project apps/python python packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-smoke --output-dir tmp/nijika-pair-topo-smoke-v2 --model-kind structured_pair_topology_spectral_head --epochs 1 --batch-size 8 --hidden-dim 64`
  - smoke predict: `uv run --project apps/python python packages/nijika/predict_baseline.py --dataset-root tmp/antenna-dataset-smoke --model-path tmp/nijika-pair-topo-smoke/baseline_model.pt --sample-name antenna_000 --output-dir tmp/nijika-pair-topo-predict-smoke`
  - 结果：train / predict 路径都正常，checkpoint 读回也正常
- 在 `6600` 样本数据集上的第一次公平 pilot（直接拼接 `pair_topology` 到 pair token）：
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-pair-topo-6600-pilot50 --model-kind structured_pair_topology_spectral_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - best epoch: `37`
  - validation RMSE: `0.14137`
  - validation `dB MAE`: `7.60709 dB`
  - 结论：明显差于现有 `structured_pair_spectral_head` 的 50 epoch 表现（`7.1605 dB`），说明“直接拼接 topology 特征”会破坏原本较强的 structured encoder
- 第二次 pilot（改成只作用于 coupling pair 的 residual topology adapter，且 adapter 零初始化，尽量保持 baseline 主干不受扰动）：
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-pair-topo-6600-pilot50-v2 --model-kind structured_pair_topology_spectral_head --epochs 50 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - best epoch: `39`
  - validation RMSE: `0.14438`
  - validation `dB MAE`: `7.59816 dB`
  - 相比第一次只改善了 `0.12%`，仍然显著差于：
    - spectral baseline @ epoch 50: `7.1605 dB`
    - pure graph branch @ epoch 50: `6.8171 dB`
- 当前判断：
  - `pair_topology` 这个特征本身不一定没价值，但“直接往现有 pair head 里注入”的两种简单方式都没有带来收益
  - 这条路线目前不值得直接上 `300` epoch 全量训练
  - 更合理的下一步应转向：
    - coupling / deep-notch 定向 loss 或 decoder 设计
    - 或者先做更细的 error slicing，确认 topology 信息究竟对哪些 coupling 子类样本有帮助，再决定是否保留这条分支

Plan update 2026-05-02 (next experiment queue)

- Step 1: 先尝试新的 `structured_pair_coupling_freq_head`
  - 保留当前 `structured_pair_spectral_head` 的几何 token encoder、token mixer、pair latent 构造方式
  - reflection (`S11/S22/S33`) 继续用当前“整条频谱一次性回归”的简单头
  - coupling (`S12/S13/S21/S23/S31/S32`) 改成显式 frequency-conditioned decoder，让每个 coupling pair latent 和频率 embedding 逐频交互
  - 目标：优先提升 deep-notch / coupling 区域拟合，而不是重做整个 encoder
- Step 2: 先做低成本验证
  - compile
  - smoke train / smoke predict
  - `6600` 样本数据集上的 `50` epoch 公平 pilot
- Step 3: 以当前已知基线作为 gate
  - 主要比较对象：`structured_pair_spectral_head` 在 epoch `50` 的 `7.1605 dB`
  - 如果新头在 `50` epoch 已经没有优势，就停止这条分支，不上 `300` epoch
  - 如果新头在 `50` epoch 有明显优势，再继续跑 `300` epoch + validation analysis
- Step 4: 如果 Step 1 失败，下一条路线改为 loss 侧
  - 在现有 spectral baseline 上做 staged `db_weight / notch_weight / coupling_weight`
  - 优先让训练目标更对齐最终关注的 coupling / deep-notch dB 误差
