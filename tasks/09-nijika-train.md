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

Progress update 2026-04-12 (dataset fix + retrain)

- Found a major dataset bug in `apps/web/components/nijika/antenna-builder.ts`:
  - old `generateRandomAntennaOptions()` only randomized `frameWidth` and `gap`
  - `cuts` and `nibs` were effectively hard-coded, so the old dataset had `1` unique cut pattern and `1` unique nib pattern
- Fixed dataset generation:
  - randomize nib `distance` and `width`
  - randomize cut count / side / `distance` / `width`
  - keep simple spacing constraints to avoid strong overlap on the same edge
  - clear old `antenna_*.json` and sample folders before rebuild in `apps/web/scripts/batch-antenna.ts`
- Rebuilt dataset with `npm run build:nijika-dataset`:
  - `100/100` simulations succeeded
  - new dataset has `100` unique cut patterns and `100` unique nib patterns
  - cut-count distribution: `{1: 27, 2: 26, 3: 26, 4: 21}`
- Retrained the current baseline without changing the model:
  - command: `uv run --project apps/python python packages/nijika/run_baseline.py --epochs 180 --batch-size 16 --hidden-dim 128 --lr 1e-3`
  - validation RMSE: `0.18954`
  - validation magnitude dB MAE: `9.47 dB`
  - validation magnitude dB RMSE: `13.66 dB`
  - example sample `antenna_001` RMSE: `0.20138`
  - example sample `antenna_001` magnitude dB MAE: `7.57 dB`
- Interpretation:
  - old baseline gains were inflated by the low-diversity dataset
  - after fixing the data generator, the current global baseline is clearly too weak and underfits the task
  - best validation result happened at epoch `1`, which suggests architecture limits are now the main bottleneck rather than overfitting
- Validation-set prediction plots for manual inspection:
  - directory: `tmp/nijika-baseline-val-predict`
  - summary: `tmp/nijika-baseline-val-predict/val_summary.json`

Progress update 2026-04-12 (structured spectral head)

- Added explicit antenna-configuration features in `packages/nijika/baseline/antenna_features.py`:
  - encode `frameWidth` / `gap`
  - encode up to `4` cut tokens with side / distance / width / active flag
  - encode up to `4` nib tokens with side / distance / width / thickness / active flag
- Added two new model branches and kept the old ones for comparison:
  - `structured_token_decoder`: structure-aware version of the old frequency-conditioned decoder
  - `structured_pair_spectral_head`: structure-aware encoder + pair-wise full-spectrum head
- New best structure is `structured_pair_spectral_head` in `packages/nijika/baseline/structured_spectral_model.py`
  - rationale: this dataset is generated from low-dimensional `antennaConfig`, so learning directly from `frame/cuts/nibs` is easier than recovering them from sparse point samples
  - rationale: frequency sampling is fixed at `201` bins, so decoding the whole spectrum per port-pair works better than independent per-frequency decoding
- Training / inference plumbing now passes the structured features end-to-end:
  - dataset loader: `packages/nijika/baseline/data.py`
  - model factory default: `packages/nijika/baseline/model.py`
  - training default: `packages/nijika/baseline/train.py`
  - prediction path: `packages/nijika/baseline/predict.py`
- Best current run:
  - command: `uv run --project apps/python python packages/nijika/run_baseline.py --model-kind structured_pair_spectral_head --epochs 180 --batch-size 16 --hidden-dim 160 --lr 1e-3`
  - dataset: 100 complete samples from `tmp/antenna-dataset`, 80/20 split, seed `7`
- Current best metrics:
  - validation RMSE: `0.16548` (previous `0.18954`, improved `12.69%`)
  - validation magnitude dB MAE: `8.91 dB` (previous `9.47 dB`, improved `5.92%`)
  - validation magnitude dB RMSE: `13.76 dB`
  - example sample `antenna_001` RMSE: `0.16748` (previous `0.20138`, improved `16.83%`)
  - example sample `antenna_001` magnitude dB MAE: `7.98 dB`
- Extra experiments that did not beat the best run:
  - `structured_token_decoder`, `hidden_dim=160`, `batch_size=16`: validation dB MAE `9.53 dB`
  - `structured_pair_spectral_head`, `hidden_dim=128`, `batch_size=16`: validation dB MAE `9.12 dB`
  - `structured_pair_spectral_head`, `hidden_dim=160`, `batch_size=8`: validation dB MAE `9.80 dB`
- Latest artifacts:
  - model: `tmp/nijika-baseline/baseline_model.pt`
  - metrics: `tmp/nijika-baseline/metrics.json`
  - train log: `tmp/nijika-baseline/latest_train.log`
  - validation plots: `tmp/nijika-baseline-val-predict`
  - validation summary: `tmp/nijika-baseline-val-predict/val_summary.json`
- Prediction entry is unchanged:
  - `uv run --project apps/python python packages/nijika/predict_baseline.py --model-path tmp/nijika-baseline/baseline_model.pt --sample-name antenna_001`

Progress update 2026-04-12 (pole/residue head)

- Added a new structured rational-spectrum branch in `packages/nijika/baseline/structured_pole_model.py`:
  - model kind: `structured_pair_pole_residue_head`
  - keep the same structure-aware encoder as the current structured baseline
  - predict sample-level shared stable poles from the global latent
  - predict pair-level complex residues plus direct / linear terms from each port-pair latent
  - reconstruct the full complex spectrum with a conjugate-pole rational form instead of direct per-bin regression
- Wired the new branch into the baseline entry points:
  - model factory default is now `structured_pair_pole_residue_head`
  - training CLI adds `--num-poles`
  - checkpoint metadata stores `num_poles`, and prediction loads the new model kind without extra changes
- Smoke-test status:
  - train command: `uv run --project apps/python python packages/nijika/run_baseline.py --epochs 1 --batch-size 8 --hidden-dim 128 --num-poles 10 --output-dir tmp/nijika-pole-smoke`
  - predict command: `uv run --project apps/python python packages/nijika/predict_baseline.py --model-path tmp/nijika-pole-smoke/baseline_model.pt --sample-name antenna_001 --output-dir tmp/nijika-pole-predict-smoke`
  - smoke-test completed successfully on `cuda`
  - 1-epoch validation RMSE: `0.18998`
  - 1-epoch validation magnitude dB MAE: `9.42 dB`
  - sample `antenna_001` prediction RMSE: `0.20241`
  - sample `antenna_001` prediction magnitude dB MAE: `6.92 dB`
- Current interpretation:
  - implementation path is stable enough for longer training runs and ablation
  - these smoke-test metrics are only for correctness validation, not a fair comparison against the current best `structured_pair_spectral_head`
  - next useful sweep is `num_poles in {8, 12, 16}` with the same 180-epoch budget used by the current best model

Progress update 2026-04-12 (pole-count sweep)

- Ran a fair comparison sweep for `structured_pair_pole_residue_head` with the same budget as the current best spectral head:
  - common command shape: `uv run --project apps/python python packages/nijika/run_baseline.py --model-kind structured_pair_pole_residue_head --epochs 180 --batch-size 16 --hidden-dim 160 --lr 1e-3 --num-poles N`
  - dataset: 100 complete samples from `tmp/antenna-dataset`, 80/20 split, seed `7`
- Sweep results:
  - `num_poles=8`: best epoch `75`, validation RMSE `0.17362`, validation magnitude dB MAE `9.96 dB`, validation magnitude dB RMSE `15.21 dB`
  - `num_poles=12`: best epoch `102`, validation RMSE `0.16481`, validation magnitude dB MAE `9.13 dB`, validation magnitude dB RMSE `13.61 dB`
  - `num_poles=16`: best epoch `39`, validation RMSE `0.17814`, validation magnitude dB MAE `9.24 dB`, validation magnitude dB RMSE `14.62 dB`
- Comparison against the current best `structured_pair_spectral_head`:
  - best spectral-head validation RMSE: `0.16548`
  - best spectral-head validation magnitude dB MAE: `8.91 dB`
  - best spectral-head validation magnitude dB RMSE: `13.76 dB`
  - conclusion: `num_poles=12` is the strongest pole/residue setting so far
  - conclusion: pole/residue is slightly better on RMSE and dB RMSE, but still worse on the main selection metric `dB MAE`
  - conclusion: keep `structured_pair_spectral_head` as the default training model for now, and keep `structured_pair_pole_residue_head` as an explicit experimental option
- Artifacts:
  - `tmp/nijika-pole-8`
  - `tmp/nijika-pole-12`
  - `tmp/nijika-pole-16`
- Next useful improvements for the pole/residue line:
  - add explicit pole regularization to reduce unstable overfitting at larger pole counts
  - try second-order resonator parameterization instead of first-order complex poles
  - weight the loss more heavily around deep notches / resonance neighborhoods so the rational head optimizes the metric we care about

Progress update 2026-04-12 (dataset scaling tooling)

- Upgraded `apps/web/scripts/batch-antenna.ts` for long dataset-generation runs:
  - add CLI options `--samples`, `--nibs`, `--append`, `--keep-intermediates`, `--output-dir`
  - default behavior now prunes successful sample folders down to only `S*.cst.txt`, which are the files required by `packages/nijika/baseline/data.py`
  - `--append` resumes from the next `antenna_XXX` index instead of clearing the dataset root
  - `dataset.json` is rebuilt from disk after each run, so append mode keeps metadata consistent
- Smoke tests completed successfully:
  - fresh run: `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 1 --output-dir ../../tmp/antenna-dataset-smoke`
  - append run: `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 1 --append --output-dir ../../tmp/antenna-dataset-smoke`
  - both runs succeeded, and successful sample folders were reduced to the `9` `S*.cst.txt` files only
- Recommended large-run command for the next dataset expansion:
  - `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 2400 --output-dir ../../tmp/antenna-dataset-2400`
