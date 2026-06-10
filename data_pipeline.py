"""
data_pipeline.py

Phase 1 dataset utilities for the UMich quadruped CSV format.

Expected columns:
- time
- pos_joint_0 ... pos_joint_11
- vel_joint_0 ... vel_joint_11
- Vx, Vy, omega_z, roll, pitch, yaw_rate

This module prepares sliding windows of raw proprioception and
supervised auxiliary targets for the Layer 1 hybrid attention model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


JOINT_POS_COLUMNS = [f"pos_joint_{index}" for index in range(12)]
JOINT_VEL_COLUMNS = [f"vel_joint_{index}" for index in range(12)]
RAW_FEATURE_COLUMNS = [*JOINT_POS_COLUMNS, *JOINT_VEL_COLUMNS]
PHYS_COLUMNS = ["Vx", "Vy", "omega_z", "roll", "pitch", "yaw_rate"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _to_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _safe_gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.zeros_like(values)

    safe_time = np.asarray(time_s, dtype=np.float64).copy()
    if safe_time.shape != values.shape:
        raise ValueError("time_s and values must have the same shape for gradient computation.")

    # Enforce strict monotonicity so repeated timestamps do not create divide-by-zero
    # inside np.gradient. This preserves ordering while keeping time perturbations
    # negligible relative to the original sampling period.
    epsilon = 1e-9
    for index in range(1, len(safe_time)):
        if not np.isfinite(safe_time[index]):
            safe_time[index] = safe_time[index - 1] + epsilon
        elif safe_time[index] <= safe_time[index - 1]:
            safe_time[index] = safe_time[index - 1] + epsilon

    return np.gradient(values, safe_time, edge_order=1)


def load_umich_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the single CSV dataset and validate the expected schema.
    """
    frame = pd.read_csv(csv_path)
    _require_columns(frame, ["time", *RAW_FEATURE_COLUMNS, *PHYS_COLUMNS], "umich_quadruped_dataset.csv")

    frame = frame.copy()
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("The `time` column contains non-numeric values.")

    frame[RAW_FEATURE_COLUMNS + PHYS_COLUMNS] = frame[RAW_FEATURE_COLUMNS + PHYS_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if frame[RAW_FEATURE_COLUMNS + PHYS_COLUMNS].isna().any().any():
        raise ValueError("The dataset contains non-numeric or missing values in required columns.")

    # Remove repeated timestamps by averaging rows that landed on the same clock tick.
    # This is the cleanest way to avoid singular derivatives later in the pipeline.
    aggregation_columns = RAW_FEATURE_COLUMNS + PHYS_COLUMNS
    frame = (
        frame.groupby("time", as_index=False)[aggregation_columns]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Drop any rows that still contain non-finite values after aggregation.
    finite_mask = np.isfinite(frame[["time", *RAW_FEATURE_COLUMNS, *PHYS_COLUMNS]].to_numpy(dtype=np.float64)).all(axis=1)
    frame = frame.loc[finite_mask].reset_index(drop=True)

    return frame


def build_phase1_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived training targets used to supervise the Layer 1 backbone.

    The core state is already provided by the dataset, so we derive:
    - body speed magnitude
    - joint activity magnitude
    - traction/slip proxy
    - one-step future residual targets for the physical state vector
    """
    frame = frame.copy()
    time_s = _to_float_array(frame["time"])

    vx = _to_float_array(frame["Vx"])
    vy = _to_float_array(frame["Vy"])
    omega_z = _to_float_array(frame["omega_z"])

    body_speed = np.sqrt(vx * vx + vy * vy)
    joint_speed = np.sqrt(np.mean(np.square(_to_float_array(frame[JOINT_VEL_COLUMNS])), axis=1))
    joint_activity = np.mean(np.abs(_to_float_array(frame[JOINT_VEL_COLUMNS])), axis=1)

    slip_index = np.clip(
        np.abs(joint_speed - body_speed) / (joint_speed + body_speed + 1e-6),
        0.0,
        1.0,
    )

    frame["body_speed"] = body_speed.astype(np.float32)
    frame["joint_speed"] = joint_speed.astype(np.float32)
    frame["joint_activity"] = joint_activity.astype(np.float32)
    frame["slip_index"] = slip_index.astype(np.float32)

    frame["ax"] = _safe_gradient(vx, time_s).astype(np.float32)
    frame["ay"] = _safe_gradient(vy, time_s).astype(np.float32)
    frame["yaw_accel"] = _safe_gradient(omega_z, time_s).astype(np.float32)
    frame["speed_accel"] = _safe_gradient(body_speed, time_s).astype(np.float32)

    return frame


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return values * self.std + self.mean


@dataclass
class Phase1Normalization:
    raw: Standardizer
    phys: Standardizer
    aux: Standardizer


def fit_normalization(frame: pd.DataFrame, train_end: int) -> Phase1Normalization:
    train_frame = frame.iloc[:train_end].copy()
    raw = Standardizer.fit(train_frame[RAW_FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    phys = Standardizer.fit(train_frame[PHYS_COLUMNS].to_numpy(dtype=np.float32))
    phys_values = train_frame[PHYS_COLUMNS].to_numpy(dtype=np.float32)
    delta_values = np.roll(phys_values, -1, axis=0) - phys_values
    delta_values[-1] = 0.0
    aux = Standardizer.fit(
        np.concatenate(
            [
                delta_values,
                train_frame[["slip_index"]].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
    )
    return Phase1Normalization(raw=raw, phys=phys, aux=aux)


class Phase1WindowDataset(Dataset):
    """
    Sliding windows for Layer 1.

    Each item returns:
    - x_raw: [sequence_length, 24]
    - x_phys: [6]
    - x_phys_future: [6]
    - aux_target: [7]  -> [delta_Vx, delta_Vy, delta_omega_z, delta_roll, delta_pitch, delta_yaw_rate, slip_index]
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        sequence_length: int = 50,
        prediction_horizon: int = 5,
        start_index: int = 0,
        end_index: int | None = None,
        normalization: Phase1Normalization | None = None,
    ) -> None:
        super().__init__()
        self.frame = frame.reset_index(drop=True).copy()
        self.sequence_length = int(sequence_length)
        self.prediction_horizon = int(prediction_horizon)
        self.start_index = int(start_index)
        self.end_index = len(self.frame) if end_index is None else int(end_index)
        self.normalization = normalization

        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        if self.end_index > len(self.frame):
            raise ValueError("end_index is out of bounds")
        if self.start_index < 0 or self.start_index >= self.end_index:
            raise ValueError("start_index/end_index define an empty range")

        self._max_start = self.end_index - self.sequence_length - self.prediction_horizon + 1
        self._length = max(0, self._max_start - self.start_index)

        self.raw = np.asarray(self.frame[RAW_FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=np.float32).copy()
        self.phys_raw = np.asarray(self.frame[PHYS_COLUMNS].to_numpy(dtype=np.float32), dtype=np.float32).copy()
        self.slip = np.asarray(self.frame["slip_index"].to_numpy(dtype=np.float32), dtype=np.float32).copy()

        if self.normalization is not None:
            self.raw = self.normalization.raw.transform(self.raw)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.start_index + index
        window_end = start + self.sequence_length
        target_index = window_end + self.prediction_horizon - 1

        x_raw = torch.tensor(self.raw[start:window_end], dtype=torch.float32)
        x_phys = torch.tensor(self.phys_raw[window_end - 1], dtype=torch.float32)
        x_phys_future = torch.tensor(self.phys_raw[target_index], dtype=torch.float32)
        delta_phys = x_phys_future - x_phys
        slip_target = torch.tensor([self.slip[window_end - 1]], dtype=torch.float32)
        aux_target = torch.cat([delta_phys, slip_target], dim=0)

        return {
            "x_raw": x_raw,
            "x_phys": x_phys,
            "x_phys_future": x_phys_future,
            "aux_target": aux_target,
        }


def build_dataloaders(
    csv_path: str | Path,
    sequence_length: int = 50,
    prediction_horizon: int = 5,
    batch_size: int = 32,
    train_fraction: float = 0.8,
    max_rows: int | None = None,
    num_workers: int = 0,
    seed: int = 7,
) -> tuple[DataLoader, DataLoader, Phase1WindowDataset, Phase1Normalization]:
    """
    Build chronological train/validation loaders with train-only normalization.
    """
    frame = build_phase1_frame(load_umich_dataset(csv_path))
    if max_rows is not None:
        max_rows = max(int(max_rows), sequence_length + prediction_horizon + 1)
        frame = frame.iloc[:max_rows].reset_index(drop=True)

    n_rows = len(frame)
    total_windows = n_rows - sequence_length - prediction_horizon + 1
    if total_windows < 2:
        raise ValueError("Dataset is too small for the requested sequence length and prediction horizon.")

    train_windows = int(total_windows * train_fraction)
    train_windows = max(1, min(train_windows, total_windows - 1))
    train_end = train_windows + sequence_length + prediction_horizon - 1

    normalization = fit_normalization(frame, train_end=train_end)

    train_dataset = Phase1WindowDataset(
        frame=frame,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        start_index=0,
        end_index=train_end,
        normalization=normalization,
    )
    val_dataset = Phase1WindowDataset(
        frame=frame,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        start_index=train_windows,
        end_index=n_rows,
        normalization=normalization,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training split produced no usable windows. Adjust train_fraction or sequence_length.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split produced no usable windows. Adjust train_fraction or sequence_length.")

    generator = torch.Generator().manual_seed(seed)
    _ = generator  # kept for interface symmetry; chronological split is deterministic.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the Phase 1 dataset pipeline")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_loader, val_loader, train_dataset, _ = build_dataloaders(
        csv_path=args.csv,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        max_rows=args.max_rows,
    )
    sample = train_dataset[0]
    print(f"Train windows: {len(train_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"x_raw shape: {tuple(sample['x_raw'].shape)}")
    print(f"x_phys shape: {tuple(sample['x_phys'].shape)}")
    print(f"x_phys_future shape: {tuple(sample['x_phys_future'].shape)}")
    print(f"aux_target shape: {tuple(sample['aux_target'].shape)}")


if __name__ == "__main__":
    main()
