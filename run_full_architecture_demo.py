"""
run_full_architecture_demo.py

End-to-end smoke test for the full current architecture:
- Phase 1 backbone inference
- Augmented-state packing
- Lock-free shared-memory channel
- Phase 2 gain scheduling / interpolation

This is a functional demonstration, not a controller benchmark.
It is intended to prove that the pieces connect correctly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from data_pipeline import build_dataloaders
from phase1_model import DualHorizonHybridBlock, build_augmented_state
from phase1_pipeline import AugmentedStateChannel, AugmentedStateFallbackPolicy
from phase2_gain_table import GainTable, LQRCell, ScheduleGrid
from phase2_runtime import Phase2GainScheduler


DEFAULT_VX_GRID = [0.0, 0.5, 1.0, 1.5]
DEFAULT_SLIP_GRID = [0.0, 0.33, 0.66, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end architecture smoke test")
    parser.add_argument("--csv", type=Path, required=True, help="Path to umich_quadruped_dataset.csv")
    parser.add_argument("--weights", type=Path, default=Path("phase1_weights.pt"))
    parser.add_argument("--values-file", type=Path, default=Path("some values.txt"))
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_backbone(weights_path: Path, device: torch.device) -> tuple[DualHorizonHybridBlock, dict]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    backbone = DualHorizonHybridBlock(
        d_in=24,
        d_model=int(config.get("d_model", 64)),
        d_latent=int(config.get("d_latent", 8)),
        n_heads=4,
        window=int(config.get("window", 12)),
    ).to(device)

    state_dict = checkpoint.get("backbone_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain a backbone_state_dict or model_state_dict.")
    backbone.load_state_dict(state_dict)
    backbone.eval()
    return backbone, checkpoint


def load_seed_values(values_file: Path) -> list[float]:
    if not values_file.exists():
        return []
    text = values_file.read_text(encoding="utf-8", errors="ignore")
    numbers: list[float] = []
    token = ""
    for character in text:
        if character.isdigit() or character in ".-+eE":
            token += character
        elif token:
            try:
                numbers.append(float(token))
            except ValueError:
                pass
            token = ""
    if token:
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    return numbers


def build_demo_gain_scheduler(seed_values: list[float], state_dim: int = 14, control_dim: int = 12) -> Phase2GainScheduler:
    grid = ScheduleGrid.from_sequences(DEFAULT_VX_GRID, DEFAULT_SLIP_GRID)
    table = GainTable.allocate(grid=grid, state_dim=state_dim, control_dim=control_dim)

    if not seed_values:
        seed_values = [45.0, 40.0, 35.0, 30.0, 25.0, 20.0]

    for i, vx_value in enumerate(grid.vx_values):
        for j, slip_value in enumerate(grid.slip_values):
            seed = float(seed_values[(i * len(grid.slip_values) + j) % len(seed_values)])
            scale = seed * (1.0 - 0.45 * float(slip_value)) + 2.0 * float(vx_value)
            A = np.eye(state_dim, dtype=np.float32) * (1.0 + 0.01 * scale)
            B = np.ones((state_dim, control_dim), dtype=np.float32) * (0.02 * scale)
            K = np.eye(control_dim, state_dim, dtype=np.float32) * scale
            table.set_cell(i, j, LQRCell(A=A, B=B, K=K))

    return Phase2GainScheduler(gain_table=table)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    _, _, train_dataset, _ = build_dataloaders(
        csv_path=args.csv,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    backbone, checkpoint = load_backbone(args.weights, device)
    latent_dim = int(checkpoint.get("config", {}).get("d_latent", 8))
    channel = AugmentedStateChannel(dim=6 + latent_dim)
    fallback = AugmentedStateFallbackPolicy(dim=6 + latent_dim)
    seed_values = load_seed_values(args.values_file)
    scheduler = build_demo_gain_scheduler(seed_values, state_dim=6 + latent_dim, control_dim=12)

    print("=== Full Architecture Smoke Test ===")
    print(f"Dataset windows available : {len(train_dataset)}")
    print(f"Phase 1 latent dim        : {latent_dim}")
    print(f"Phase 2 grid              : {scheduler.gain_table.grid.shape}")
    print(f"Seed values loaded        : {seed_values if seed_values else '[defaults used]'}")

    with torch.no_grad():
        for sample_index in range(min(args.samples, len(train_dataset))):
            sample = train_dataset[sample_index]
            x_raw = sample["x_raw"].unsqueeze(0).to(device)
            x_phys = sample["x_phys"].unsqueeze(0).to(device)

            S, z = backbone.init_states(batch_size=1, device=device)
            x_latent, _, _ = backbone(x_raw, S, z)
            x_augmented = build_augmented_state(x_phys, x_latent).squeeze(0).cpu().numpy().astype(np.float32)

            channel.write(x_augmented)
            read_vector, age_s = channel.read(stale_threshold_s=1.0)
            resolved_vector, mode = fallback.resolve(read_vector, age_s)
            K_dynamic = scheduler.compute_K_dynamic(resolved_vector)

            print(f"\nSample {sample_index}")
            print(f"  x_raw shape      : {tuple(x_raw.shape)}")
            print(f"  x_latent shape   : {tuple(x_latent.shape)}")
            print(f"  x_augmented shape: {x_augmented.shape}")
            print(f"  channel mode     : {mode}")
            print(f"  channel age (ms) : {age_s * 1000.0:.3f}")
            print(f"  K_dynamic shape  : {K_dynamic.shape}")
            print(f"  K_dynamic norm   : {float(np.linalg.norm(K_dynamic)):.3f}")
            print(f"  K[0, :4]         : {np.round(K_dynamic[0, :4], 3)}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
