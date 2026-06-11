"""
phase2_runtime.py

Runtime wrapper for mapping x_augmented -> K_dynamic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from phase2_gain_table import GainTable, LQRCell
from phase2_interpolation import interpolate_gain_cell


def default_schedule_extractor(x_augmented: np.ndarray) -> tuple[float, float]:
    """
    Extract the scheduling variables from the 14D augmented state.

    Assumption:
    - x_augmented[0] = Vx
    - x_augmented[6] = slip/context proxy from latent state

    If your latent semantics change later, only this function needs to change.
    """
    x_augmented = np.asarray(x_augmented, dtype=np.float32).reshape(-1)
    if x_augmented.shape[0] < 7:
        raise ValueError("x_augmented must have at least 7 elements.")
    vx = float(x_augmented[0])
    slip_proxy = float(np.clip(x_augmented[6], 0.0, 1.0))
    return vx, slip_proxy


@dataclass
class Phase2GainScheduler:
    gain_table: GainTable
    schedule_extractor: Callable[[np.ndarray], tuple[float, float]] = default_schedule_extractor

    def compute_cell(self, x_augmented: np.ndarray) -> LQRCell:
        vx, slip = self.schedule_extractor(x_augmented)
        return interpolate_gain_cell(self.gain_table, vx=vx, slip=slip)

    def compute_K_dynamic(self, x_augmented: np.ndarray) -> np.ndarray:
        cell = self.compute_cell(x_augmented)
        return cell.K

    def compute_ABK(self, x_augmented: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cell = self.compute_cell(x_augmented)
        return cell.A, cell.B, cell.K


def build_default_scheduler(
    vx_grid: list[float] | np.ndarray,
    slip_grid: list[float] | np.ndarray,
    state_dim: int,
    control_dim: int,
) -> Phase2GainScheduler:
    from phase2_gain_table import ScheduleGrid, GainTable

    grid = ScheduleGrid.from_sequences(vx_grid, slip_grid)
    table = GainTable.allocate(grid=grid, state_dim=state_dim, control_dim=control_dim)
    return Phase2GainScheduler(gain_table=table)
