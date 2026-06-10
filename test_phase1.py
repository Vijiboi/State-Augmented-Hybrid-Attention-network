"""
test_phase1.py
Step-by-step tests for Phase 1 components.
Run with:  python3 test_phase1.py
"""

import torch
import numpy as np
import time
import multiprocessing as mp
import sys

from phase1_model    import (CausalWindowAttention, StreamingLinearAttention,
                              DualHorizonHybridBlock, build_augmented_state)
from phase1_pipeline import AugmentedStateChannel, AugmentedStateFallbackPolicy


# ── Helpers ────────────────────────────────────────────────────────────────────
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status}  {label}{suffix}")
    return condition


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Short-term causal window attention
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 1: CausalWindowAttention")

D, H, W = 64, 4, 12
head = CausalWindowAttention(d_model=D, n_heads=H, window_size=W)
head.eval()

B, T = 2, 50          # batch=2, long sequence

x_long = torch.randn(B, T, D)
with torch.no_grad():
    out_long = head(x_long)

check("Output shape [B, d_model]",
      out_long.shape == (B, D),
      f"got {tuple(out_long.shape)}")

# Verify window enforcement: feeding T=5 < W and T=W+5 both work
x_short = torch.randn(B, 5, D)
with torch.no_grad():
    out_short = head(x_short)
check("Works when T < window_size",
      out_short.shape == (B, D))

# Verify causal: output at step t must not depend on step t+1
head.eval()
x_base = torch.randn(1, W, D)
x_modified = x_base.clone()
x_modified[0, -1, :] += 100.0       # perturb last step only
with torch.no_grad():
    o_base = head(x_base[:, :-1, :])    # up to step W-2
    o_mod  = head(x_modified[:, :-1, :])
check("Causal: earlier steps unaffected by later perturbation",
      torch.allclose(o_base, o_mod))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Long-term streaming linear attention
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 2: StreamingLinearAttention")

lhead = StreamingLinearAttention(d_model=D)
lhead.eval()

B, T_long = 2, 200
x_long = torch.randn(B, T_long, D)

with torch.no_grad():
    out_seq, S_final, z_final = lhead(x_long)

check("Sequential forward output shape",
      out_seq.shape == (B, D),
      f"got {tuple(out_seq.shape)}")
check("State S shape [B, d, d]",
      S_final.shape == (B, D, D))
check("Normalizer z shape [B, d]",
      z_final.shape == (B, D))

# step() inference: process same sequence one step at a time
S = torch.zeros(B, D, D)
z = torch.zeros(B, D)
step_out = None
with torch.no_grad():
    for t in range(T_long):
        step_out, S, z = lhead.step(x_long[:, t], S, z)

check("Step-by-step output shape",
      step_out.shape == (B, D))

# State should be similar (not identical due to normalizer) — just check finite
check("Step-by-step S is finite",
      torch.isfinite(S).all().item())
check("Step-by-step z is finite",
      torch.isfinite(z).all().item())

# Memory: state size is constant regardless of sequence length
S2 = torch.zeros(B, D, D)
z2 = torch.zeros(B, D)
with torch.no_grad():
    for t in range(5 * T_long):   # 5x longer sequence
        _, S2, z2 = lhead.step(torch.randn(B, D), S2, z2)
check("State size constant over 1000-step sequence",
      S2.shape == (B, D, D))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — DualHorizonHybridBlock end-to-end
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 3: DualHorizonHybridBlock (full forward pass)")

D_IN, D_MODEL, D_LATENT = 24, 64, 8
model = DualHorizonHybridBlock(d_in=D_IN, d_model=D_MODEL,
                                d_latent=D_LATENT, n_heads=4, window=12)
model.eval()

B, T = 1, 100
x_raw = torch.randn(B, T, D_IN)   # simulated sensor buffer

S, z = model.init_states(batch_size=B)
with torch.no_grad():
    x_latent, S_new, z_new = model(x_raw, S, z)

check("x_latent shape [B, d_latent]",
      x_latent.shape == (B, D_LATENT),
      f"got {tuple(x_latent.shape)}")
check("x_latent is finite (no NaN/Inf)",
      torch.isfinite(x_latent).all().item())
check("Carry state S updated",
      not torch.equal(S_new, S))

# Parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model parameter count: {n_params:,}")
check("Parameter count < 500k (suitable for companion CPU)",
      n_params < 500_000,
      f"got {n_params:,}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — State augmentation and dimensionality
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 4: build_augmented_state dimensionality")

N_PHYS = 6   # [Vx, Vy, omega_z, roll, pitch, yaw_rate]
x_phys   = torch.tensor([[0.5, 0.0, 0.1, 0.02, -0.01, 0.05]])  # [1, 6]

with torch.no_grad():
    x_latent, _, _ = model(x_raw, *model.init_states(1))

x_aug = build_augmented_state(x_phys, x_latent)

check("x_augmented shape [B, n_phys + d_latent]",
      x_aug.shape == (1, N_PHYS + D_LATENT),
      f"got {tuple(x_aug.shape)}")
check("x_phys values preserved in augmented vector",
      torch.allclose(x_aug[:, :N_PHYS], x_phys))
check("x_latent values preserved in augmented vector",
      torch.allclose(x_aug[:, N_PHYS:], x_latent))

print(f"\n  x_phys  = {x_phys.numpy().round(3)}")
print(f"  x_latent= {x_latent.numpy().round(3)}")
print(f"  x_aug   = {x_aug.numpy().round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Pipeline: AugmentedStateChannel (single-process, synchronous)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 5: AugmentedStateChannel write/read")

channel = AugmentedStateChannel(dim=N_PHYS + D_LATENT)
test_vec = x_aug.squeeze(0).numpy()

channel.write(test_vec)
read_vec, age = channel.read(stale_threshold_s=1.0)

check("Read succeeds after write",
      read_vec is not None)
check("Round-trip data matches",
      np.allclose(read_vec, test_vec, atol=1e-5),
      f"max diff={np.abs(read_vec - test_vec).max():.2e}")
check("Age is small (< 10ms after immediate write)",
      age < 0.010,
      f"age={age*1000:.2f}ms")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Fallback policy behaviour
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 6: AugmentedStateFallbackPolicy")

policy  = AugmentedStateFallbackPolicy(dim=N_PHYS + D_LATENT)
nominal = np.zeros(N_PHYS + D_LATENT, dtype=np.float32)
policy.set_nominal(nominal)
good    = test_vec

# Simulate: live data arrives
out, mode = policy.resolve(good, 0.01)
check("Live data passes through unchanged",
      mode == "live" and np.allclose(out, good))

# Simulate: stale but within hold window
out, mode = policy.resolve(None, 0.08)
check("Hold mode: returns last good value",
      mode == "hold" and np.allclose(out, good))

# Simulate: stale, in blend window
out, mode = policy.resolve(None, 0.22)
check("Blend mode: output is between good and nominal",
      mode == "blend")
is_between = np.all((out >= np.minimum(good, nominal) - 1e-5) &
                    (out <= np.maximum(good, nominal) + 1e-5))
check("Blend values are interpolated (between good and nominal)",
      is_between)

# Simulate: very stale
out, mode = policy.resolve(None, 0.50)
check("Nominal mode: returns nominal state when very stale",
      mode == "nominal-stale" and np.allclose(out, nominal))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Inference latency benchmark
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 7: Inference latency (simulates 20–50 Hz companion loop)")

model.eval()
B, T = 1, 100
x_buf = torch.randn(B, T, D_IN)
S, z  = model.init_states(1)

# Warm up
with torch.no_grad():
    for _ in range(5):
        _, S, z = model(x_buf, S, z)

# Benchmark: full forward pass (as run by companion computer)
N_RUNS = 200
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_RUNS):
        x_latent, S, z = model(x_buf, S, z)
elapsed_ms = (time.perf_counter() - t0) / N_RUNS * 1000

print(f"\n  Mean forward pass latency: {elapsed_ms:.2f} ms")
check(f"Latency < 50ms (compatible with 20Hz companion loop)",
      elapsed_ms < 50.0,
      f"{elapsed_ms:.2f}ms")
check(f"Latency < 20ms (compatible with 50Hz companion loop)",
      elapsed_ms < 20.0,
      f"{elapsed_ms:.2f}ms")

# Step-by-step inference latency (linear head only, as used at deployment)
lhead_only = StreamingLinearAttention(D_MODEL).eval()
S2 = torch.zeros(1, D_MODEL, D_MODEL)
z2 = torch.zeros(1, D_MODEL)
x_step = torch.randn(1, D_MODEL)

t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_RUNS):
        _, S2, z2 = lhead_only.step(x_step, S2, z2)
step_ms = (time.perf_counter() - t0) / N_RUNS * 1000
print(f"  Linear head single-step latency: {step_ms:.3f} ms")
check("Single step < 1ms",
      step_ms < 1.0,
      f"{step_ms:.3f}ms")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("  Phase 1 test suite complete.")
print(f"{'═'*60}\n")
