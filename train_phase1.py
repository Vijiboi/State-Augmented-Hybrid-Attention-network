"""
train_phase1.py

Layer 1 training for the hybrid attention backbone.

Training signal:
- x_raw: sliding history of joint positions + joint velocities
- x_phys: current torso physical state
- target: future torso residual over a prediction horizon + slip proxy

This keeps MPC out of Phase 1. The backbone learns a compact latent that
is useful for state augmentation, while the auxiliary head teaches that
latent to carry predictive information.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.optim import Adam

from data_pipeline import build_dataloaders
from phase1_model import DualHorizonHybridBlock, build_augmented_state


class Phase1AuxHead(nn.Module):
    def __init__(self, phys_dim: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(phys_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, phys_dim)
        self.slip_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_phys: torch.Tensor, x_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_aug = build_augmented_state(x_phys, x_latent)
        hidden = self.trunk(x_aug)
        delta_pred = self.delta_head(hidden)
        slip_logit = self.slip_head(hidden)
        return delta_pred, slip_logit, x_aug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 hybrid attention backbone")
    parser.add_argument("--csv", type=Path, required=True, help="Path to umich_quadruped_dataset.csv")
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-latent", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--weights-out", type=Path, default=Path("phase1_weights.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(
    backbone: DualHorizonHybridBlock,
    aux_head: Phase1AuxHead,
    data_loader,
    optimizer: Adam | None,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    backbone.train(training)
    aux_head.train(training)

    mse = nn.MSELoss()
    total_loss = 0.0
    total_delta_loss = 0.0
    total_slip_loss = 0.0
    total_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x_raw = batch["x_raw"].to(device)
            x_phys = batch["x_phys"].to(device)
            aux_target = batch["aux_target"].to(device)

            S, z = backbone.init_states(batch_size=x_raw.size(0), device=device)
            x_latent, _, _ = backbone(x_raw, S, z)

            delta_pred, slip_logit, _ = aux_head(x_phys, x_latent)
            slip_pred = torch.sigmoid(slip_logit)

            delta_target = aux_target[:, : x_phys.shape[-1]]
            slip_target = aux_target[:, -1:].clamp(0.0, 1.0)

            delta_loss = mse(delta_pred, delta_target)
            slip_loss = mse(slip_pred, slip_target)
            latent_reg = 1e-4 * x_latent.pow(2).mean()

            loss = delta_loss + 0.5 * slip_loss + latent_reg

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(aux_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            total_loss += float(loss.item())
            total_delta_loss += float(delta_loss.item())
            total_slip_loss += float(slip_loss.item())
            total_batches += 1

    denom = max(1, total_batches)
    return {
        "loss": total_loss / denom,
        "delta_loss": total_delta_loss / denom,
        "slip_loss": total_slip_loss / denom,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, train_dataset, normalization = build_dataloaders(
        csv_path=args.csv,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        max_rows=args.max_rows,
    )

    backbone = DualHorizonHybridBlock(
        d_in=24,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_heads=4,
        window=args.window,
    ).to(device)
    aux_head = Phase1AuxHead(phys_dim=6, latent_dim=args.d_latent).to(device)

    optimizer = Adam(list(backbone.parameters()) + list(aux_head.parameters()), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(backbone, aux_head, train_loader, optimizer, device, max_batches=args.max_train_batches)
        val_metrics = run_epoch(backbone, aux_head, val_loader, None, device, max_batches=args.max_val_batches)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.6f} "
            f"(delta={train_metrics['delta_loss']:.6f}, slip={train_metrics['slip_loss']:.6f}) | "
            f"val={val_metrics['loss']:.6f} "
            f"(delta={val_metrics['delta_loss']:.6f}, slip={val_metrics['slip_loss']:.6f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "backbone_state_dict": backbone.state_dict(),
                "aux_head_state_dict": aux_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "csv": str(args.csv),
                    "sequence_length": args.sequence_length,
                    "prediction_horizon": args.prediction_horizon,
                    "batch_size": args.batch_size,
                    "train_fraction": args.train_fraction,
                    "d_model": args.d_model,
                    "d_latent": args.d_latent,
                    "window": args.window,
                },
                "normalization": {
                    "raw_mean": normalization.raw.mean,
                    "raw_std": normalization.raw.std,
                    "phys_mean": normalization.phys.mean,
                    "phys_std": normalization.phys.std,
                    "aux_mean": normalization.aux.mean,
                    "aux_std": normalization.aux.std,
                },
                "dataset_size": len(train_dataset),
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    torch.save(best_state, args.weights_out)
    print(f"Saved best checkpoint to: {args.weights_out}")

    backbone.eval()
    aux_head.eval()
    sample = train_dataset[0]
    with torch.no_grad():
        x_raw = sample["x_raw"].unsqueeze(0).to(device)
        x_phys = sample["x_phys"].unsqueeze(0).to(device)
        S, z = backbone.init_states(batch_size=1, device=device)
        x_latent, _, _ = backbone(x_raw, S, z)
        x_augmented = build_augmented_state(x_phys, x_latent)
    print(f"Example x_latent shape: {tuple(x_latent.shape)}")
    print(f"Example x_augmented shape: {tuple(x_augmented.shape)}")


if __name__ == "__main__":
    main()
