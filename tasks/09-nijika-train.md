训练一个可以预测天线 S 参数的 AI 模型
- `npm run build:nijika-dataset` 执行数据生成
- 在 `packages/nijika` 里添加模型和训练代码。AI 模型输入是 `tmp/antenna-dataset` 里的 json 信息（包含几何和端口），输出是 S 参数
- 具体使用的 AI 模型和架构，参考 `packages\nijika\doc\em_sparam_model_plan.md`
- 把计划和进展实时更新到这个文档里
Progress update 2026-04-12

- Implemented a minimal end-to-end baseline in `packages/nijika/baseline`.
- Architecture: PointNet-lite geometry encoder + flattened port features + global MLP spectral head.
- Training entry: `uv run --project apps/python python packages/nijika/run_baseline.py`
- Prediction entry: `uv run --project apps/python python packages/nijika/predict_baseline.py --model-path tmp/nijika-baseline/baseline_model.pt --sample-name antenna_001`
- Current baseline resamples each spectrum to 201 frequency points and predicts the full 3x3 complex S-matrix in one forward pass.
- Latest training run used 100 complete samples from `tmp/antenna-dataset` with an 80/20 split.
- Latest example metrics on `antenna_001`: real/imag RMSE `0.06196`, magnitude dB MAE `4.90 dB`.
- Latest artifacts:
  - model: `tmp/nijika-baseline/baseline_model.pt`
  - metrics: `tmp/nijika-baseline/metrics.json`
  - validation plot: `tmp/nijika-baseline/antenna_001_matrix_db.png`

Progress update 2026-04-12 (optimization round 2)

- Upgraded the baseline to a stronger structured model in `packages/nijika/baseline/model.py`:
  - normalize point cloud and port coordinates by geometry center/size
  - encode per-port tokens and run a lightweight Transformer over ports
  - decode only the symmetric upper triangle of the S-matrix and mirror it back, matching the reciprocal dataset
  - use frequency-conditioned decoding with Fourier features instead of one global full-spectrum head
- Upgraded training/evaluation in `packages/nijika/baseline/train.py`:
  - loss = real/imag MSE + magnitude L1 + first-difference spectral loss
  - metrics now report full validation-set RMSE / dB MAE / dB RMSE, not just one example sample
- Best current run:
  - command: `uv run --project apps/python python packages/nijika/run_baseline.py --epochs 180 --batch-size 16 --hidden-dim 128 --lr 1e-3`
  - dataset: 100 complete samples from `tmp/antenna-dataset`, 80/20 split, seed `7`
- Current best metrics:
  - validation RMSE: `0.05622` (previous `0.06473`, improved `13.14%`)
  - validation magnitude dB MAE: `2.92 dB` (previous `3.71 dB`, improved `21.24%`)
  - validation magnitude dB RMSE: `5.92 dB`
  - example sample `antenna_001` RMSE: `0.03942` (previous `0.06196`, improved `36.38%`)
  - example sample `antenna_001` magnitude dB MAE: `3.23 dB` (previous `4.90 dB`, improved `33.97%`)
- Extra experiments that did not beat the best run:
  - `hidden_dim=160`, stronger magnitude/smooth loss: validation dB MAE `3.15 dB`
  - `batch_size=8` with the same best-run architecture: validation dB MAE `2.94 dB`
- Latest artifacts:
  - model: `tmp/nijika-baseline/baseline_model.pt`
  - metrics: `tmp/nijika-baseline/metrics.json`
  - train log: `tmp/nijika-baseline/latest_train.log`
  - validation plot: `tmp/nijika-baseline/antenna_001_matrix_db.png`
- Prediction entry is unchanged, and now also reports RMSE / dB MAE when ground truth exists:
  - `uv run --project apps/python python packages/nijika/predict_baseline.py --model-path tmp/nijika-baseline/baseline_model.pt --sample-name antenna_001`
