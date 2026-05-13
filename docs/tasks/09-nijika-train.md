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

Progress update 2026-04-13 (2400-sample training started)

- Started a new baseline training run on the expanded dataset:
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --output-dir tmp/nijika-baseline-2400 --model-kind structured_pair_spectral_head --epochs 180 --batch-size 32 --hidden-dim 160 --lr 1e-3`
  - log: `tmp/nijika-baseline-2400.log`
  - error log: `tmp/nijika-baseline-2400.err.log`
- Current status:
  - training is running on `cuda`
  - epoch `1` metrics: train loss `1.0224`, validation RMSE `0.1785`, validation magnitude dB MAE `10.9906 dB`

Progress update 2026-04-13 (2400-sample result + validation analysis)

- The 2400-sample training run completed successfully:
  - metrics: `tmp/nijika-baseline-2400/metrics.json`
  - model: `tmp/nijika-baseline-2400/baseline_model.pt`
  - summary metrics:
    - train / val samples: `1920 / 480`
    - best epoch: `140`
    - validation RMSE: `0.12526`
    - validation magnitude dB MAE: `6.49544 dB`
    - validation magnitude dB RMSE: `10.37770 dB`
- Compared with the previous best `100`-sample structured spectral run:
  - old validation RMSE: `0.16548` -> new `0.12526` (`24.3%` better)
  - old validation dB MAE: `8.91 dB` -> new `6.50 dB` (`27.1%` better)
  - old validation dB RMSE: `13.76 dB` -> new `10.38 dB` (`24.6%` better)
- Ran full validation-set prediction analysis for the 2400-sample model:
  - command: `uv run --project apps/python python packages/nijika/predict_baseline.py --dataset-root tmp/antenna-dataset-2400 --model-path tmp/nijika-baseline-2400/baseline_model.pt --split val --output-dir tmp/nijika-baseline-2400-val-predict`
  - prediction summary: `tmp/nijika-baseline-2400-val-predict/val_summary.json`
  - analysis summary: `tmp/nijika-baseline-2400-val-predict/analysis.json`
- Main findings from the validation analysis:
  - sample-level `dB MAE` mean / median / p90 / max = `6.50 / 6.26 / 9.10 / 13.90 dB`
  - reflection terms are already much easier than coupling terms:
    - average reflection `dB MAE` (`S11/S22/S33`): `1.12 dB`
    - average coupling `dB MAE` (`S12/S13/S21/S23/S31/S32`): `9.18 dB`
  - the model is accurate in high-magnitude regions but struggles badly in deep-notch regions:
    - truth `>= -10 dB`: `1.53 dB` MAE
    - truth `< -10 dB`: `9.51 dB` MAE
    - truth `< -20 dB`: `9.75 dB` MAE
  - validation error rises with structural complexity:
    - `1` cut: mean `5.09 dB`
    - `2` cuts: mean `6.69 dB`
    - `3` cuts: mean `6.85 dB`
    - `4` cuts: mean `7.38 dB`
  - a dataset limitation is still present even after scaling:
    - validation set has only `1` nib-side pattern: `['top', 'bottom', 'left']`
    - so current data diversity increase is mostly in continuous geometry parameters and cut patterns, not in feed-side topology
- Interpretation:
  - increasing dataset size was clearly the right move and produced a large real gain
  - current bottleneck is no longer just sample count; it is concentrated in off-diagonal coupling prediction and deep resonant / low-magnitude regions
  - the next most valuable work is likely notch-aware / coupling-aware loss design, followed by more topology-diverse data generation

Progress update 2026-04-13 (nib-side randomization + time-domain probe)

- Updated `apps/web/components/nijika/antenna-builder.ts` so `nib` side patterns are no longer fixed:
  - previous behavior for `numNibs=3` always produced `['top', 'bottom', 'left']`
  - new behavior randomizes nib sides by sampling a shuffled subset of edges when `numNibs <= 4`
  - same-edge spacing constraints are still enforced through the existing occupied-offset logic
- Quick topology probe after the change:
  - generated `4` probe samples in `tmp/antenna-dataset-sideprobe`
  - observed nib side patterns:
    - `antenna_000`: `['bottom', 'top', 'left']`
    - `antenna_001`: `['top', 'bottom', 'right']`
    - `antenna_002`: `['top', 'left', 'right']`
    - `antenna_003`: `['top', 'bottom', 'right']`
- Time-domain probe:
  - generated `1` sample with preserved intermediates in `tmp/antenna-dataset-timeprobe`
  - confirmed that `i*.txt`, `o*,*.txt`, and `Port *[*].txt` are plain 2-column time-series files
  - example files:
    - `tmp/antenna-dataset-timeprobe/antenna_000/i1.txt`
    - `tmp/antenna-dataset-timeprobe/antenna_000/o1,1.txt`
    - `tmp/antenna-dataset-timeprobe/antenna_000/Port 1 [1].txt`
  - this makes a compact “first 100 time steps” auxiliary feature path feasible without keeping all raw intermediate files forever

Progress update 2026-04-13 (loss reweighting ablation)

- Added configurable loss-shaping controls in `packages/nijika/baseline/train.py` and `packages/nijika/baseline/training_utils.py`:
  - `--db-weight`: add a denormalized magnitude-dB L1 term
  - `--coupling-weight`: upweight off-diagonal S-parameters in the real/imag, magnitude, and smoothness losses
  - `--notch-weight` + `--notch-threshold-db`: ramp extra dB loss on deep-notch regions
  - checkpoints and `metrics.json` now store `loss_config`, so future sweeps do not need more code changes
- Regression check for the default loss path:
  - command: `uv run --project apps/python python packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --epochs 1 --batch-size 32 --hidden-dim 160 --lr 1e-3 --output-dir tmp/nijika-loss-default-smoke2400`
  - epoch `1` validation dB MAE: `10.9998 dB`
  - this matches the previous baseline epoch-1 result `10.9906 dB`, so the default training behavior is unchanged
- Smoke test for the new loss path:
  - command: `uv run --project apps/python python packages/nijika/run_baseline.py --epochs 1 --batch-size 8 --hidden-dim 128 --db-weight 0.05 --coupling-weight 1.5 --notch-weight 1.0 --notch-threshold-db -20 --output-dir tmp/nijika-loss-smoke`
  - completed successfully on `cuda`
- 2400-sample ablation A (coupling + notch-aware dB loss):
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --output-dir tmp/nijika-baseline-2400-cpl-notch --model-kind structured_pair_spectral_head --epochs 180 --batch-size 32 --hidden-dim 160 --lr 1e-3 --db-weight 0.05 --coupling-weight 1.5 --notch-weight 1.0 --notch-threshold-db -20`
  - best epoch: `1`
  - validation RMSE: `0.18724`
  - validation magnitude dB MAE: `8.76461 dB`
  - validation magnitude dB RMSE: `13.27436 dB`
- 2400-sample ablation B (coupling-only weighting):
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --output-dir tmp/nijika-baseline-2400-coupling15 --model-kind structured_pair_spectral_head --epochs 180 --batch-size 32 --hidden-dim 160 --lr 1e-3 --coupling-weight 1.5`
  - best epoch: `7`
  - validation RMSE: `0.16664`
  - validation magnitude dB MAE: `10.00057 dB`
  - validation magnitude dB RMSE: `15.61193 dB`
- Comparison against the current best 2400-sample baseline:
  - baseline validation RMSE: `0.12526`
  - baseline validation magnitude dB MAE: `6.49544 dB`
  - baseline validation magnitude dB RMSE: `10.37770 dB`
  - both new loss-weighting runs are clearly worse than the unchanged baseline
- Interpretation:
  - naive global reweighting toward coupling terms and deep-notch regions does not improve the current `structured_pair_spectral_head`
  - the explicit dB/notch term destabilized training most strongly and pushed the best checkpoint back to the first epoch
  - even simple off-diagonal upweighting hurts overall calibration, despite the baseline analysis showing coupling is the dominant error source
  - keep the original loss as the default for now; future loss work should use gentler schedules or more localized weighting instead of fixed global multipliers
- Artifacts:
  - default regression smoke: `tmp/nijika-loss-default-smoke2400`
  - new-loss smoke: `tmp/nijika-loss-smoke`
  - ablation A: `tmp/nijika-baseline-2400-cpl-notch`
  - ablation B: `tmp/nijika-baseline-2400-coupling15`

Progress update 2026-04-14 (deeper architecture + warmup)

- Architecture improvements to `structured_pair_spectral_head` in `packages/nijika/baseline/structured_spectral_model.py`:
  - pair MLP: added one hidden layer (2 → 3 layers), now `hidden_dim*5 → hidden_dim*2 → hidden_dim*2 → hidden_dim`
  - spectral decoder: added one hidden layer (2 → 3 layers), now `hidden_dim → hidden_dim*2 → hidden_dim*2 → freq_bins*2`
  - rationale: more capacity for coupling prediction and complex spectral shapes
- Training improvements in `packages/nijika/baseline/train.py`:
  - added `--warmup-epochs` (default `10`) for linear LR warmup
  - replaced `CosineAnnealingLR` with manual warmup + cosine decay schedule
  - deeper model should benefit from warmup for stable early training
- Regenerating dataset with nib-side randomization:
  - previous 2400-sample dataset had fixed nib-side pattern `['top', 'bottom', 'left']`
  - new dataset generating into `tmp/antenna-dataset-2400-v2` with randomized nib sides
- Ablation: training deeper model on existing 2400 dataset to isolate architecture effect:
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --output-dir tmp/nijika-deeper-2400 --epochs 300 --batch-size 32 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - status: in progress
- Ablation result (deeper model on old dataset):
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400 --output-dir tmp/nijika-deeper-2400 --epochs 300 --batch-size 32 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - best epoch: `241`
  - validation RMSE: `0.13008`
  - validation magnitude dB MAE: `6.54320 dB`
  - validation magnitude dB RMSE: `10.44783 dB`
  - interpretation: deeper model is slightly worse than old 2-layer model (`6.50` dB) on the same low-diversity dataset
  - the deeper model converges slower (best at epoch 241 vs 140) but does not reach a better minimum
  - confirms that dataset diversity (nib-side patterns) is the current bottleneck, not model capacity
- Artifacts:
  - deeper model on old dataset: `tmp/nijika-deeper-2400`
  - new dataset regenerating: `tmp/antenna-dataset-2400-v2` (in progress)
- Split decoder ablation (`structured_pair_split_decoder`):
  - separate diagonal decoder (2 layers) and coupling decoder (4 layers)
  - command: same budget, 300 epochs with warmup
  - best epoch: `8`, validation dB MAE: `9.68 dB`
  - interpretation: separate decoders overfit severely; the coupling decoder is too deep for this dataset size
  - the shared decoder is better because gradients from all pairs help regularize the encoder
  - keep `structured_pair_split_decoder` as an experimental option but not the default
- Architecture revert:
  - reverted `pair_mlp` and `spectral_decoder` back to 2-layer MLPs (matching old best model)
  - kept LR warmup improvement (linear warmup + cosine decay)
  - conclusion: on the current 2400-sample dataset, model capacity is NOT the bottleneck
  - the most impactful improvement is the dataset regeneration with nib-side randomization
- Artifacts:
  - deeper model on old dataset: `tmp/nijika-deeper-2400`
  - split decoder model: `tmp/nijika-split-2400`
  - new dataset regenerating: `tmp/antenna-dataset-2400-v2` (in progress)

Progress update 2026-04-14 (nib-randomized 2400-sample result)

- New dataset `tmp/antenna-dataset-2400-v2` completed: `2400/2400` samples, `0` failures, `24` unique nib-side patterns
- Trained `structured_pair_spectral_head` (2-layer MLP, reverted from deeper experiment) + LR warmup:
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-v2-2400 --epochs 300 --batch-size 32 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - best epoch: `223`
  - validation RMSE: `0.13878`
  - validation magnitude dB MAE: `7.09 dB`
  - validation magnitude dB RMSE: `11.06 dB`
- Compared with previous best (old non-randomized 2400 dataset):
  - old dB MAE: `6.50 dB` → new `7.09 dB` (`9.1%` worse)
  - old RMSE: `0.12526` → new `0.13878` (`10.8%` worse)
  - old dB RMSE: `10.38 dB` → new `11.06 dB` (`6.5%` worse)
- Interpretation:
  - nib-side randomization increases task difficulty: `24` unique patterns vs `1`
  - the old model's `6.50 dB` was inflated by the low-diversity validation set (all same nib pattern)
  - the new `7.09 dB` is a more honest measure of true generalization performance
  - with `2400` samples / `24` patterns ≈ `100` samples per pattern, the model has insufficient per-pattern coverage
  - the model was still improving at epoch `300`, suggesting longer training or more data would help
- Next most valuable improvement: scale the dataset to `5000+` samples to increase per-pattern coverage
- Artifacts:
  - model: `tmp/nijika-v2-2400/baseline_model.pt`
  - metrics: `tmp/nijika-v2-2400/metrics.json`
  - validation plots: `tmp/nijika-v2-2400-val-predict`
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-v2-222 --epochs 180 --batch-size 16 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - best epoch: `165`, validation dB MAE: `9.93 dB`
  - compared with old non-randomized 100-sample: `8.91 dB`
  - interpretation: nib-side randomization makes the task harder; 222 samples is not enough data for the increased diversity
  - the model needs the full 2400 samples to benefit from the additional diversity
- Summary of architecture experiments (all on old 2400-sample dataset, no nib randomization):
  - 2-layer MLP (old best): dB MAE `6.50 dB` ← best so far
  - 3-layer MLP (deeper): dB MAE `6.54 dB` ← slightly worse, overfits marginally
  - Split decoder (diag 2-layer + coupling 4-layer): dB MAE `9.68 dB` ← severely overfits
  - conclusion: model capacity is NOT the bottleneck on the current dataset; data diversity is
- Next steps:
  - wait for `tmp/antenna-dataset-2400-v2` to complete (2400 samples with nib-side randomization)
  - retrain with `structured_pair_spectral_head` (2-layer) + LR warmup on the new dataset
  - if results improve, try deeper model on new dataset to see if capacity helps with more diverse data
  - consider generating 5000+ samples if the improvement trend continues
- Artifacts:
  - deeper model on old dataset: `tmp/nijika-deeper-2400`
  - split decoder model: `tmp/nijika-split-2400`
  - 222-sample nib-randomized: `tmp/nijika-v2-222`
  - new dataset regenerating: `tmp/antenna-dataset-2400-v2` (in progress)

Progress update 2026-04-15 (12-hour data-generation run)

- New constraint: there is now a dedicated `12`-hour budget for dataset generation, so the highest-value move is to spend that budget on more topology-diverse data instead of more model ablations first.
- Throughput estimate from the completed long run in `tmp/antenna-dataset-2400.log`:
  - `2400` samples finished in `06:27:57`
  - average speed ≈ `371 samples/hour` ≈ `9.70 s/sample`
  - a full fresh `7200`-sample rebuild would exceed the current `12`-hour budget, so reusing the existing `2400`-sample v2 dataset is the only practical way to get to `5000+` total samples within this window
- Long-run generator safety fix:
  - restored default pruning in `apps/web/scripts/batch-antenna.ts` so successful samples keep only `S*.cst.txt`
  - added explicit `--keep-intermediates` for debugging / time-domain probes
  - rationale: without pruning, a `12`-hour run would waste significant disk on `i*.txt`, `o*.txt`, `Port *[*].txt`, `grid.json`, and `run.log`, none of which are required by `packages/nijika/baseline/data.py`
- Smoke test status after the fix:
  - fresh run: `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 1 --output-dir ../../tmp/antenna-dataset-12h-smoke`
  - append run: `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 1 --append --output-dir ../../tmp/antenna-dataset-12h-smoke`
  - both runs succeeded, and each sample directory was reduced back to exactly the `9` required `S*.cst.txt` files
- Started the 12-hour append run:
  - command: `pnpm --filter glassbeaker-web exec tsx scripts/batch-antenna.ts --samples 4200 --append --output-dir ../../tmp/antenna-dataset-2400-v2`
  - log: `tmp/antenna-dataset-2400-v2-append4200.log`
  - error log: `tmp/antenna-dataset-2400-v2-append4200.err.log`
  - start index: `2400`
  - target if the run completes: existing `2400` + new `4200` = `6600` total samples in `tmp/antenna-dataset-2400-v2`
  - note: the directory name keeps the historical `2400-v2` label, but the actual dataset size will grow beyond `2400`; reusing the existing directory is the only budget-feasible choice for this run
- Why `4200` instead of `5000` new samples:
  - historical estimate for `4200` new samples is about `11.32` hours, which leaves some buffer inside the `12`-hour window
  - `5000` new samples would project to about `13.47` hours at the measured historical speed, so it is too risky for the current budget
- Next step after generation completes:
  - train `structured_pair_spectral_head` on the expanded nib-randomized dataset and compare against the current `2400`-sample v2 result (`7.09 dB` validation magnitude dB MAE)

Progress update 2026-04-15 (6600-sample training started)

- The `12`-hour append generation run completed successfully:
  - added samples: `4200/4200`
  - failed simulations: `0`
  - final dataset: `tmp/antenna-dataset-2400-v2`
  - final sample count: `6600`
  - generation log: `tmp/antenna-dataset-2400-v2-append4200.log`
  - error log: `tmp/antenna-dataset-2400-v2-append4200.err.log` (`0` bytes)
- Started training on the expanded dataset:
  - command: `uv run --project apps/python python -u packages/nijika/run_baseline.py --dataset-root tmp/antenna-dataset-2400-v2 --output-dir tmp/nijika-v2-6600 --model-kind structured_pair_spectral_head --epochs 300 --batch-size 64 --hidden-dim 160 --lr 1e-3 --warmup-epochs 10`
  - log: `tmp/nijika-v2-6600.log`
  - error log: `tmp/nijika-v2-6600.err.log`
  - output directory: `tmp/nijika-v2-6600`
- Training configuration notes:
  - kept the current strongest model family: `structured_pair_spectral_head`
  - kept `hidden_dim=160`, `lr=1e-3`, and `warmup_epochs=10` for continuity with the 2400-sample v2 run
  - increased batch size from `32` to `64` because the dataset is now `6600` samples and the model is small enough for the available GPU memory
- Initial training status:
  - device: `cuda`
  - epoch `1`: train loss `1.1121`, validation RMSE `0.1814`, validation magnitude dB MAE `10.7427 dB`
  - this is only a warmup epoch and is not yet comparable with the previous best 2400-sample v2 checkpoint
