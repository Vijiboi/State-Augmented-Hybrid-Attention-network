"""
phase2_gain_table.py

Gain-table primitives for Phase 2.

The table is scheduled over:
- forward speed Vx
- slip proxy

Each grid cell can store:
- A matrix
- B matrix
- LQR gain K

This file keeps the data model simple so you can fill it from offline
system identification or later controller design work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ScheduleGrid:
    vx_values: np.ndarray
    slip_values: np.ndarray

    @classmethod
    def from_sequences(cls, vx_values: Sequence[float], slip_values: Sequence[float]) -> "ScheduleGrid":
        vx_array = np.asarray(vx_values, dtype=np.float32)
        slip_array = np.asarray(slip_values, dtype=np.float32)
        if vx_array.ndim != 1 or slip_array.ndim != 1:
            raise ValueError("Grid axes must be one-dimensional.")
        if len(vx_array) < 2 or len(slip_array) < 2:
            raise ValueError("Each grid axis must have at least two points.")
        if not np.all(np.diff(vx_array) > 0):
            raise ValueError("vx_values must be strictly increasing.")
        if not np.all(np.diff(slip_array) > 0):
            raise ValueError("slip_values must be strictly increasing.")
        return cls(vx_values=vx_array, slip_values=slip_array)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.vx_values), len(self.slip_values)


@dataclass
class LQRCell:
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray

    @classmethod
    def zeros(cls, state_dim: int, control_dim: int) -> "LQRCell":
        return cls(
            A=np.zeros((state_dim, state_dim), dtype=np.float32),
            B=np.zeros((state_dim, control_dim), dtype=np.float32),
            K=np.zeros((control_dim, state_dim), dtype=np.float32),
        )

    def validate(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square.")
        if self.B.ndim != 2:
            raise ValueError("B must be a matrix.")
        if self.K.ndim != 2:
            raise ValueError("K must be a matrix.")
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("A and B row dimensions must match.")
        if self.K.shape[1] != self.A.shape[1]:
            raise ValueError("K column dimension must equal the state dimension.")
        if self.K.shape[0] != self.B.shape[1]:
            raise ValueError("K row dimension must equal the control dimension.")


@dataclass
class GainTable:
    grid: ScheduleGrid
    cells: np.ndarray

    @classmethod
    def allocate(
        cls,
        grid: ScheduleGrid,
        state_dim: int,
        control_dim: int,
    ) -> "GainTable":
        cells = np.empty(grid.shape, dtype=object)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cells[i, j] = LQRCell.zeros(state_dim=state_dim, control_dim=control_dim)
        return cls(grid=grid, cells=cells)

    def set_cell(self, i: int, j: int, cell: LQRCell) -> None:
        cell.validate()
        self.cells[i, j] = cell

    def get_cell(self, i: int, j: int) -> LQRCell:
        cell = self.cells[i, j]
        if not isinstance(cell, LQRCell):
            raise TypeError(f"Cell ({i}, {j}) is not an LQRCell.")
        return cell

    @property
    def state_dim(self) -> int:
        return self.get_cell(0, 0).A.shape[0]

    @property
    def control_dim(self) -> int:
        return self.get_cell(0, 0).K.shape[0]
