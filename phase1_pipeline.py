"""
phase1_pipeline.py
Lock-free triple-buffer channel for passing x_augmented from the
non-RT companion process to the RT control thread without blocking.
"""

import multiprocessing as mp
import ctypes
import time
import numpy as np


N_AUGMENTED = 14   # 6 physical states + 8 latent features


class AugmentedStateChannel:
    """
    Triple-buffer, lock-free shared-memory channel.
    Producer (20-50 Hz): call write()
    Consumer (1 kHz RT): call read()  — never acquires the write lock
    """

    def __init__(self, dim: int = N_AUGMENTED):
        self._dim        = dim
        self._data       = mp.Array(ctypes.c_float, 3 * dim)
        self._ready      = mp.Value(ctypes.c_int,    0)
        self._stamp      = mp.Value(ctypes.c_double, 0.0)
        self._write_slot = 1   # producer alternates between slots 1 and 2

    # ── Producer (non-RT) ──────────────────────────────────────────────────
    def write(self, x_augmented: np.ndarray):
        assert x_augmented.shape == (self._dim,), \
            f"Expected ({self._dim},), got {x_augmented.shape}"
        slot   = self._write_slot
        offset = slot * self._dim
        with self._data.get_lock():
            self._data[offset: offset + self._dim] = x_augmented.tolist()
        self._ready.value      = slot
        self._stamp.value      = time.monotonic()
        self._write_slot       = 3 - slot      # toggle 1↔2

    # ── Consumer (RT-safe read) ────────────────────────────────────────────
    def read(self, stale_threshold_s: float = 0.10):
        """
        Returns (x_augmented: np.ndarray | None, age_s: float).
        Returns (None, age) if data is stale — caller uses fallback.
        Does NOT acquire the write lock.
        """
        age    = time.monotonic() - self._stamp.value
        if age > stale_threshold_s:
            return None, age
        slot   = self._ready.value
        offset = slot * self._dim
        raw    = self._data[offset: offset + self._dim]
        return np.array(raw, dtype=np.float32), age


class AugmentedStateFallbackPolicy:
    """
    Graduated fallback when x_latent is stale.
    Tune thresholds to your estimator rate after integration testing.
    """
    HOLD_THRESHOLD_S  = 0.15   # hold last good value
    BLEND_THRESHOLD_S = 0.30   # blend toward nominal
    # Beyond 300ms → use nominal (conservative/zero-slip gains in Phase 2)

    def __init__(self, dim: int = N_AUGMENTED):
        self._last_good = None
        self._nominal   = np.zeros(dim, dtype=np.float32)

    def set_nominal(self, nominal: np.ndarray):
        self._nominal = nominal.astype(np.float32)

    def resolve(self, x: np.ndarray | None, age_s: float
               ) -> tuple[np.ndarray, str]:
        if x is not None:
            self._last_good = x.copy()
            return x, "live"

        if self._last_good is None:
            return self._nominal.copy(), "nominal-cold"

        if age_s < self.HOLD_THRESHOLD_S:
            return self._last_good.copy(), "hold"
        elif age_s < self.BLEND_THRESHOLD_S:
            alpha   = (age_s - self.HOLD_THRESHOLD_S) / \
                      (self.BLEND_THRESHOLD_S - self.HOLD_THRESHOLD_S)
            blended = (1 - alpha) * self._last_good + alpha * self._nominal
            return blended.astype(np.float32), "blend"
        else:
            return self._nominal.copy(), "nominal-stale"
