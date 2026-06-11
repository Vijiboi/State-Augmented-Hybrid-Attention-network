"""
run_live_inference.py

Mock deployment harness for Phase 1.

Producer thread:
- loads phase1_weights.pt
- steps through a CSV-backed window dataset
- runs the hybrid attention backbone
- writes 14D augmented states into AugmentedStateChannel

Consumer thread:
- polls the shared channel at 1 kHz
- applies the fallback policy
- records freshness / jitter metrics

This is intended as a verification script before moving to Phase 2.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from data_pipeline import build_dataloaders
from phase1_model import DualHorizonHybridBlock, build_augmented_state
from phase1_pipeline import AugmentedStateChannel, AugmentedStateFallbackPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock live inference and shared-memory test")
    parser.add_argument("--csv", type=Path, required=True, help="Path to umich_quadruped_dataset.csv")
    parser.add_argument("--weights", type=Path, default=Path("phase1_weights.pt"))
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--producer-hz", type=float, default=20.0)
    parser.add_argument("--consumer-hz", type=float, default=1000.0)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--prediction-horizon", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_backbone(weights_path: Path, device: torch.device) -> tuple[DualHorizonHybridBlock, dict]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint.get(
        "config",
        {
            "d_model": 64,
            "d_latent": 8,
            "window": 12,
        },
    )

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


def producer_loop(
    stop_event: threading.Event,
    stats: dict,
    stats_lock: threading.Lock,
    backbone: DualHorizonHybridBlock,
    dataset,
    channel: AugmentedStateChannel,
    device: torch.device,
    producer_hz: float,
) -> None:
    period_s = 1.0 / float(producer_hz)
    index = 0
    local_writes = 0
    local_deadline_miss = 0
    local_max_compute_ms = 0.0
    next_tick = time.perf_counter()

    with torch.no_grad():
        while not stop_event.is_set():
            sample = dataset[index % len(dataset)]
            index += 1

            x_raw = sample["x_raw"].unsqueeze(0).to(device)
            x_phys = sample["x_phys"].unsqueeze(0).to(device)

            S, z = backbone.init_states(batch_size=1, device=device)
            compute_start = time.perf_counter()
            x_latent, _, _ = backbone(x_raw, S, z)
            x_augmented = build_augmented_state(x_phys, x_latent).squeeze(0).detach().cpu().numpy().astype(np.float32)
            compute_ms = (time.perf_counter() - compute_start) * 1000.0
            local_max_compute_ms = max(local_max_compute_ms, compute_ms)

            channel.write(x_augmented)
            local_writes += 1

            next_tick += period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                local_deadline_miss += 1
                next_tick = time.perf_counter()

    with stats_lock:
        stats["producer_writes"] = local_writes
        stats["producer_deadline_miss"] = local_deadline_miss
        stats["producer_max_compute_ms"] = local_max_compute_ms


def consumer_loop(
    stop_event: threading.Event,
    stats: dict,
    stats_lock: threading.Lock,
    channel: AugmentedStateChannel,
    fallback: AugmentedStateFallbackPolicy,
    consumer_hz: float,
) -> None:
    period_s = 1.0 / float(consumer_hz)
    local_reads = 0
    local_live_reads = 0
    local_stale_reads = 0
    local_max_loop_ms = 0.0
    local_modes = {"live": 0, "hold": 0, "blend": 0, "nominal-cold": 0, "nominal-stale": 0}
    next_tick = time.perf_counter()

    while not stop_event.is_set():
        loop_start = time.perf_counter()
        x_live, age_s = channel.read()
        x_resolved, mode = fallback.resolve(x_live, age_s)
        _ = x_resolved

        local_reads += 1
        local_modes[mode] = local_modes.get(mode, 0) + 1
        if mode == "live":
            local_live_reads += 1
        else:
            local_stale_reads += 1

        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        local_max_loop_ms = max(local_max_loop_ms, loop_ms)

        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

    with stats_lock:
        stats["consumer_reads"] = local_reads
        stats["consumer_live_reads"] = local_live_reads
        stats["consumer_stale_reads"] = local_stale_reads
        stats["consumer_max_loop_ms"] = local_max_loop_ms
        stats["consumer_modes"] = local_modes


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    train_loader, _, train_dataset, _ = build_dataloaders(
        csv_path=args.csv,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    _ = train_loader

    backbone, checkpoint = load_backbone(args.weights, device)
    latent_dim = int(checkpoint.get("config", {}).get("d_latent", 8))
    channel = AugmentedStateChannel(dim=6 + latent_dim)
    fallback = AugmentedStateFallbackPolicy(dim=6 + latent_dim)

    stop_event = threading.Event()
    stats: dict = {}
    stats_lock = threading.Lock()

    producer = threading.Thread(
        target=producer_loop,
        args=(stop_event, stats, stats_lock, backbone, train_dataset, channel, device, args.producer_hz),
        daemon=True,
    )
    consumer = threading.Thread(
        target=consumer_loop,
        args=(stop_event, stats, stats_lock, channel, fallback, args.consumer_hz),
        daemon=True,
    )

    print(
        f"Starting live inference test for {args.duration_s:.1f}s "
        f"(producer={args.producer_hz:.1f}Hz, consumer={args.consumer_hz:.1f}Hz)"
    )
    producer.start()
    consumer.start()

    time.sleep(args.duration_s)
    stop_event.set()
    producer.join(timeout=5.0)
    consumer.join(timeout=5.0)

    with stats_lock:
        producer_writes = stats.get("producer_writes", 0)
        producer_deadline_miss = stats.get("producer_deadline_miss", 0)
        producer_max_compute_ms = stats.get("producer_max_compute_ms", 0.0)
        consumer_reads = stats.get("consumer_reads", 0)
        consumer_live_reads = stats.get("consumer_live_reads", 0)
        consumer_stale_reads = stats.get("consumer_stale_reads", 0)
        consumer_max_loop_ms = stats.get("consumer_max_loop_ms", 0.0)
        consumer_modes = stats.get("consumer_modes", {})

    print("\nLive inference summary")
    print(f"  producer writes         : {producer_writes}")
    print(f"  producer deadline misses : {producer_deadline_miss}")
    print(f"  producer max compute ms  : {producer_max_compute_ms:.3f}")
    print(f"  consumer reads          : {consumer_reads}")
    print(f"  consumer live reads     : {consumer_live_reads}")
    print(f"  consumer stale reads    : {consumer_stale_reads}")
    print(f"  consumer max loop ms    : {consumer_max_loop_ms:.3f}")
    print(f"  consumer modes          : {consumer_modes}")

    if consumer_reads == 0:
        raise RuntimeError("Consumer did not execute any reads.")
    if producer_writes == 0:
        raise RuntimeError("Producer did not write any augmented states.")
    if consumer_live_reads == 0:
        raise RuntimeError("Consumer never received a live vector.")
    if channel.read()[0] is None:
        print("Final channel state is stale; fallback is active, which is acceptable for this mock test.")


if __name__ == "__main__":
    main()
