# Nijika

Nijika is the surrogate training and analysis package for antenna S-parameter prediction.

## Entrypoints

Run all commands from the repo root with the Python environment in `apps/python`:

```powershell
uv run --project apps/python python packages/nijika/run_baseline.py ...
uv run --project apps/python python packages/nijika/predict_baseline.py ...
uv run --project apps/python python packages/nijika/analyze_baseline.py ...
```

## Dataset Layout

The training dataset root is expected to look like this:

```text
tmp/antenna-dataset-2400-v2/
  antenna_000.json
  antenna_001.json
  ...
  antenna_000/
    S1,1.cst.txt
    S1,2.cst.txt
    ...
```

Each `antenna_XXX.json` contains the geometry and antenna config. Each sample directory contains the complex S-parameter text files used by `baseline.data`.

## Current Recommended Training Command

The current strongest forward surrogate is the staged-loss spectral baseline:

```powershell
uv run --project apps/python python -u packages/nijika/run_baseline.py `
  --dataset-root tmp/antenna-dataset-2400-v2 `
  --output-dir tmp/nijika-staged-loss-6600 `
  --model-kind structured_pair_spectral_head `
  --epochs 300 `
  --batch-size 64 `
  --hidden-dim 160 `
  --lr 1e-3 `
  --warmup-epochs 10 `
  --db-weight-final 0.1 `
  --coupling-weight-final 1.5 `
  --notch-weight-final 2.0 `
  --notch-threshold-db -10 `
  --loss-ramp-start-epoch 10 `
  --loss-ramp-end-epoch 35
```

Known result on the `6600`-sample dataset:

- Best epoch: `268`
- Validation `dB MAE`: `4.49499 dB`
- Output artifacts: `baseline_model.pt`, `metrics.json`, and one validation plot

## Prediction

Predict one sample from a trained checkpoint:

```powershell
uv run --project apps/python python packages/nijika/predict_baseline.py `
  --dataset-root tmp/antenna-dataset-2400-v2 `
  --model-path tmp/nijika-staged-loss-6600/baseline_model.pt `
  --sample-name antenna_000 `
  --output-dir tmp/nijika-predict-one
```

Predict an entire split:

```powershell
uv run --project apps/python python packages/nijika/predict_baseline.py `
  --dataset-root tmp/antenna-dataset-2400-v2 `
  --model-path tmp/nijika-staged-loss-6600/baseline_model.pt `
  --split val `
  --output-dir tmp/nijika-predict-val
```

Prediction artifacts include:

- `<sample>_matrix_db.png`
- `<sample>_prediction.npz`
- `<sample>_prediction.json`
- `<split>_summary.json` for split mode

## Analysis

Analyze a checkpoint on a dataset split:

```powershell
uv run --project apps/python python packages/nijika/analyze_baseline.py `
  --dataset-root tmp/antenna-dataset-2400-v2 `
  --model-path tmp/nijika-staged-loss-6600/baseline_model.pt `
  --split val `
  --batch-size 256 `
  --output-dir tmp/nijika-staged-loss-6600-analysis
```

The analysis output contains:

- `val_summary.json`: aggregate metrics and per-sample `db_mae` / `rmse`
- `analysis.json`: pair-wise errors, notch-region metrics, cut-count buckets, and best/worst samples

## Useful Model Kinds

- `structured_pair_spectral_head`
  Current default and recommended baseline.
- `structured_pair_pole_residue_head`
  Experimental pole/residue decoder with shared poles.
- `structured_pair_pole_offset_residue_head`
  Experimental variant that keeps shared poles but allows bounded pair-specific pole offsets.

Example pilot for the pair-pole-offset experiment:

```powershell
uv run --project apps/python python -u packages/nijika/run_baseline.py `
  --dataset-root tmp/antenna-dataset-2400-v2 `
  --output-dir tmp/nijika-staged-loss-pole-offset-6600-pilot50 `
  --model-kind structured_pair_pole_offset_residue_head `
  --epochs 50 `
  --batch-size 64 `
  --hidden-dim 160 `
  --lr 1e-3 `
  --warmup-epochs 10 `
  --num-poles 12 `
  --db-weight-final 0.1 `
  --coupling-weight-final 1.5 `
  --notch-weight-final 2.0 `
  --notch-threshold-db -10 `
  --loss-ramp-start-epoch 10 `
  --loss-ramp-end-epoch 35
```

Known result for that pilot:

- Best epoch: `49`
- Validation `dB MAE`: `5.21926 dB`
- This is better than the original staged-loss pole/residue pilot, but still behind the staged-loss spectral baseline

## Smoke Commands

Quick correctness checks:

```powershell
uv run --project apps/python python -m unittest `
  apps.python.tests.test_nijika_pole_model `
  apps.python.tests.test_nijika_analyze

uv run --project apps/python python packages/nijika/run_baseline.py `
  --dataset-root tmp/antenna-dataset-smoke `
  --output-dir tmp/nijika-smoke `
  --model-kind structured_pair_spectral_head `
  --epochs 1 `
  --batch-size 8 `
  --hidden-dim 64
```
