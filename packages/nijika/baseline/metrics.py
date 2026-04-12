from __future__ import annotations

import numpy as np
import torch


def magnitude_db(matrix: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.square(matrix[..., 0]) + np.square(matrix[..., 1]))
    return 20.0 * np.log10(np.clip(mag, 1e-6, None))


def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def summarize_prediction_metrics(
    pred: torch.Tensor | np.ndarray,
    truth: torch.Tensor | np.ndarray,
    port_count: int,
) -> dict[str, float | list[float]]:
    pred_np = _to_numpy(pred)
    truth_np = _to_numpy(truth)
    pred_matrix = pred_np.reshape(pred_np.shape[0], pred_np.shape[1], port_count, port_count, 2)
    truth_matrix = truth_np.reshape(truth_np.shape[0], truth_np.shape[1], port_count, port_count, 2)
    pred_db = magnitude_db(pred_matrix)
    truth_db = magnitude_db(truth_matrix)
    sample_rmse = np.sqrt(np.mean(np.square(pred_np - truth_np), axis=(1, 2)))
    sample_db_mae = np.mean(np.abs(pred_db - truth_db), axis=(1, 2, 3))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(pred_np - truth_np)))),
        "db_mae": float(np.mean(np.abs(pred_db - truth_db))),
        "db_rmse": float(np.sqrt(np.mean(np.square(pred_db - truth_db)))),
        "sample_rmse": sample_rmse.tolist(),
        "sample_db_mae": sample_db_mae.tolist(),
    }
