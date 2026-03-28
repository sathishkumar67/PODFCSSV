# Complete Pipeline Guide

This guide summarizes the current executable pipeline. If this guide and the code disagree, trust:

- `main.py`
- `src/*.py`

## 1. One Entry Point

The repository now uses a single executable file:

- `main.py`

Choose the behavior by setting `RUN_MODE`:

- `federated`
- `baseline`

## 2. Shared Model Recipe

Both modes:

1. load `facebook/vit-mae-base`
2. freeze the MAE backbone
3. inject residual adapters into the upper half of the encoder
4. train only the adapter weights during continual training

Image preprocessing:

1. convert to RGB
2. resize to `224 x 224`
3. convert to tensor

No ImageNet normalization is used in the current pipeline.

## 3. Benchmark Datasets

The benchmark datasets are:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Client benchmark schedule:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

## 4. Retention-Stress Stages

Both modes use the same extra stress stages between benchmark stages and after the last benchmark stage:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

These datasets are used only to stress retention. They are not part of the benchmark evaluation set.

## 5. Budget Rule

- `EuroSAT` uses a fixed deterministic split of `10000` train and `5000` eval samples
- every benchmark training split targets `10000` train samples
- splits below `1000` samples are rejected
- larger splits are deterministically subsampled
- smaller valid splits are deterministically repeated up to the target budget

The baseline uses the same budget policy as the federated benchmark.

## 6. Federated Mode

In `RUN_MODE = "federated"`, `main.py`:

1. builds one shared adapter-injected MAE model
2. assigns one client per usable GPU
3. trains through benchmark and stress stages
4. applies MAE reconstruction and GPAD locally
5. merges local prototypes and aggregates adapter weights
6. preserves global memory and enriches the client-local prototype banks across stages instead of resetting them
7. evaluates on seen benchmark datasets after every stage
8. saves checkpoints, histories, metrics, and plots

Tracked stage-wise benchmark metrics:

- per-dataset accuracy
- average accuracy
- forgetting
- retention ratio
- backward transfer

Saved stage-wise plots:

- benchmark accuracy heatmap
- stage metric curves
- final forgetting bars

## 7. Baseline Mode

In `RUN_MODE = "baseline"`, `main.py`:

1. builds the same adapter-injected MAE model
2. trains sequentially on both benchmark and stress datasets
3. uses reconstruction loss without federation or GPAD
4. keeps the same model and optimizer state across stages
5. evaluates on seen benchmark datasets after every stage
6. saves checkpoints, histories, metrics, and plots

## 8. Download Safety

The default publishable schedule intentionally avoids datasets with manual-download caveats, including:

- `FER2013`
- `PCAM`
- `Stanford Cars`

`main.py` validates the schedule and rejects those datasets in the default benchmark or stress lists.

## 9. Runtime Safety

Before using CUDA, the runtime checks that a small CUDA kernel can actually execute. If CUDA is reported but unusable, the code falls back early instead of failing later inside training.

## 10. Dataset Smoke Test

Use `test.py` when you want to verify that every benchmark and stress dataset downloads correctly. The script downloads one dataset at a time, touches the required splits, deletes that dataset folder, and then continues to the next dataset.

## 11. Source of Truth

The current source of truth lives in:

- `main.py`
- `src/client.py`
- `src/server.py`
- `src/loss.py`
- `src/mae_with_adapter.py`

The Python docstrings in those files are intended to be read as a step-by-step walkthrough of the current pipeline. When the code and any external note disagree, trust those files first.


