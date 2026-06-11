"""
phase2_interpolation.py

Interpolation utilities for the Phase 2 gain table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phase2_gain_table import GainTable, LQRCell


@dataclass(frozen=True)
class BlendWeights:
    ix0: int
    ix1: int
    is0: int
    is1: int
    wx: float
    ws: float


def _clamp_and_find_interval(values: np.ndarray, query: float) -> tuple[int, int, float]:
    if query <= float(values[0]):
        return 0, 1, 0.0
    if query >= float(values[-1]):
        last = len(values) - 1
        return last - 1, last, 1.0

    upper = int(np.searchsorted(values, query, side="right"))
    lower = upper - 1
    span = float(values[upper] - values[lower])
    if span <= 0.0:
        return lower, upper, 0.0
    weight = (float(query) - float(values[lower])) / span
    return lower, upper, float(weight)


def compute_blend_weights(gain_table: GainTable, vx: float, slip: float) -> BlendWeights:
    ix0, ix1, wx = _clamp_and_find_interval(gain_table.grid.vx_values, vx)
    is0, is1, ws = _clamp_and_find_interval(gain_table.grid.slip_values, slip)
    return BlendWeights(ix0=ix0, ix1=ix1, is0=is0, is1=is1, wx=wx, ws=ws)


def blend_cell_matrices(
    cell_00: LQRCell,
    cell_10: LQRCell,
    cell_01: LQRCell,
    cell_11: LQRCell,
    wx: float,
    ws: float,
) -> LQRCell:
    """
    Bilinear interpolation across one grid rectangle.
    """
    one_minus_wx = 1.0 - wx
    one_minus_ws = 1.0 - ws

    A = (
        one_minus_wx * one_minus_ws * cell_00.A
        + wx * one_minus_ws * cell_10.A
        + one_minus_wx * ws * cell_01.A
        + wx * ws * cell_11.A
    )
    B = (
        one_minus_wx * one_minus_ws * cell_00.B
        + wx * one_minus_ws * cell_10.B
        + one_minus_wx * ws * cell_01.B
        + wx * ws * cell_11.B
    )
    K = (
        one_minus_wx * one_minus_ws * cell_00.K
        + wx * one_minus_ws * cell_10.K
        + one_minus_wx * ws * cell_01.K
        + wx * ws * cell_11.K
    )
    return LQRCell(A=A.astype(np.float32), B=B.astype(np.float32), K=K.astype(np.float32))


def interpolate_gain_cell(gain_table: GainTable, vx: float, slip: float) -> LQRCell:
    """
    Return the interpolated cell for a scheduled operating point.
    """
    weights = compute_blend_weights(gain_table, vx=vx, slip=slip)
    cell_00 = gain_table.get_cell(weights.ix0, weights.is0)
    cell_10 = gain_table.get_cell(weights.ix1, weights.is0)
    cell_01 = gain_table.get_cell(weights.ix0, weights.is1)
    cell_11 = gain_table.get_cell(weights.ix1, weights.is1)
    return blend_cell_matrices(cell_00, cell_10, cell_01, cell_11, weights.wx, weights.ws)
