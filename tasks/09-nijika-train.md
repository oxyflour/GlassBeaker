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
