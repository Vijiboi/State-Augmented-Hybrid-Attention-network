# Augmented State Space — Phase 1 + Phase 2 Smoke Test

This workspace now includes a complete testable path for the current architecture:

- **Phase 1**: hybrid attention backbone that turns proprioceptive history into `x_latent`
- **State augmentation**: `x_augmented = concat(x_phys, x_latent)`
- **Shared memory channel**: lock-free write/read via `AugmentedStateChannel`
- **Phase 2**: gain-scheduled LQR interpolation over `(Vx, slip)`

## Important Note

- Phase 1 is ready and good to go (uses MIT MiniCheetah based Dataset for training on different surfaces; extracted courtesy University of Michigan, Github Repo link: https://github.com/UMich-CURLY/deep-contact-estimator)
- Phase 2 is still a scheduler that is based on DUMMY values, that was added in the original repo just for testing purposes. The real `A_k`, `B_k`, and final LQR gains can be filled in later from system identification or offline linearization (Dataset to be selected can be based off expected inputs/outputs given below).
- `KMP_DUPLICATE_LIB_OK=TRUE` is set inside the training and demo scripts so the local Windows OpenMP runtime can start cleanly in this environment.

## Files

- `data_pipeline.py` — loads `umich_quadruped_dataset.csv` and builds sliding windows
- `train_phase1.py` — trains the Phase 1 backbone and saves `phase1_weights.pt`
- `run_live_inference.py` — mock producer/consumer deployment test with shared memory
- `phase2_gain_table.py` — grid and gain-cell storage
- `phase2_interpolation.py` — bilinear interpolation over the gain table
- `phase2_runtime.py` — maps `x_augmented` to `K_dynamic`
- `run_full_architecture_demo.py` — end-to-end architecture smoke test
- `p2runtimedemo.py` — compatibility wrapper for the demo

## Requirements

Install the Python packages you need:

```powershell
python -m pip install torch numpy pandas rosbags
```

If `torch` is already installed in your environment, you can skip reinstalling it.

## Phase 1 training

Train on the full CSV:

```powershell
python train_phase1.py --csv "D:\Downloads_D\Books\Research Papers\Augmented State Space\umich_quadruped_dataset.csv" --epochs 20
```

Quick test run:

```powershell
python train_phase1.py --csv "D:\Downloads_D\Books\Research Papers\Augmented State Space\umich_quadruped_dataset.csv" --max-rows 120 --epochs 1 --batch-size 8 --max-train-batches 2 --max-val-batches 1
```

The checkpoint is saved as:

```text
phase1_weights.pt
```

## Live inference test

This verifies the producer/consumer channel:

```powershell
python run_live_inference.py --csv "D:\Downloads_D\Books\Research Papers\Augmented State Space\umich_quadruped_dataset.csv" --weights "D:\Downloads_D\Books\Research Papers\Augmented State Space\phase1_weights.pt"
```

For a short run, add:

```powershell
--duration-s 3 --max-rows 120
```

## Full architecture smoke test

This runs the current end-to-end path:

- loads a few dataset windows
- computes `x_latent`
- builds `x_augmented`
- writes/reads the shared-memory channel
- computes a scheduled `K_dynamic`

Run:

```powershell
python run_full_architecture_demo.py --csv "D:\Downloads_D\Books\Research Papers\Augmented State Space\umich_quadruped_dataset.csv" --weights "D:\Downloads_D\Books\Research Papers\Augmented State Space\phase1_weights.pt" --values-file "D:\Downloads_D\Books\Research Papers\Augmented State Space\some values.txt"
```

If `some values.txt` is empty, the demo falls back to built-in test values.

## `some values.txt`

This file is optional.

If you want to override the demo’s hardcoded Phase 2 gain scales, put any floating-point numbers in the file. The demo will cycle through those numbers when filling the gain table.

Example:

```text
45 40 35 30
25 20
```

## What to expect

- `x_raw` shape: `[1, 50, 24]`
- `x_latent` shape: `[1, 8]`
- `x_augmented` shape: `[1, 14]`
- `K_dynamic` shape: `[12, 14]`