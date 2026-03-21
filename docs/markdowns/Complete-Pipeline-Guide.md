# Complete Pipeline Guide

This guide summarizes the current executable pipeline in the repository. The source of truth is still the code in `main.py`, `base.py`, `evaluate.py`, and `src/`.

## 1. Shared Model Setup

All three entrypoints use the same backbone and adapter recipe:

1. Load `facebook/vit-mae-base`.
2. Freeze the pretrained MAE backbone.
3. Inject residual adapters into the upper half of the encoder.
4. Train only the adapter weights in downstream stages.

The shared implementation lives in `src/mae_with_adapter.py`.

## 2. Federated Continual Pipeline (`main.py`)

`main.py` is the current main experiment.

### Dataset schedule

- Client 0: `EuroSAT` -> `Oxford-IIIT Pet` -> `Flowers102`
- Client 1: `GTSRB` -> `FGVC Aircraft` -> `DTD`

### Stage flow

Each stage follows the same order:

1. Load one dataset per client.
2. Fit each training split to an effective budget of `10000` samples.
3. Create one dataloader per client.
4. Synchronize the latest global adapter weights into both clients.
5. Run local MAE + GPAD training on both clients.
6. Generate or reuse local prototypes.
7. Merge client prototypes into the global prototype bank.
8. Average adapter weights on the server.
9. Save checkpoints, metrics, and plots.

### Data-budget rule

- if a train split is below `1000`, the run fails for that dataset
- if it is above the target budget, it is deterministically subsampled
- if it is below the target budget but above the minimum, it is deterministically repeated

### Persistent state

Across dataset transitions, the following state is intentionally preserved:

- global adapter weights
- global prototype bank
- each client's local prototypes
- each client's novelty buffer
- each client's optimizer state

### Saved artifacts

`main.py` writes:

- adapter-only checkpoints
- JSON training history
- training summary plots
- communication statistics

## 3. Single-Model Continual Baseline (`base.py`)

`base.py` keeps the same adapter-injected MAE architecture but removes federation and GPAD.

### Baseline flow

1. Build one adapter-injected MAE model.
2. Train it on one dataset at a time.
3. Use only reconstruction loss.
4. Carry the same model and optimizer state across datasets.
5. Save checkpoints, JSON history, and training-only plots.

### Dataset order

- `EuroSAT`
- `GTSRB`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`
- `Flowers102`
- `DTD`

### Data usage

Unlike `main.py`, `base.py` uses the full train split of each dataset.

## 4. Checkpoint Comparison (`evaluate.py`)

`evaluate.py` compares saved checkpoints from `main.py` and `base.py`.

### Evaluation flow

1. Load the federated checkpoint and the baseline checkpoint.
2. Freeze both models.
3. Disable MAE masking by setting `mask_ratio = 0.0`.
4. Use only the encoder path to extract features.
5. Train one dataset-specific linear probe per model.
6. Evaluate on the official held-out split.
7. Save JSON metrics and plots.

### Probe fitting policy

- if train split `< 1000`: skip the dataset
- if train split is `1000..10000`: use the full train split
- if train split `> 10000`: use a deterministic `4000`-sample subset

### Held-out evaluation policy

- use `val + test` when both are explicitly supported by the loader
- use the official non-train split when only one held-out split exists
- skip datasets without an official held-out split in the current loader logic

### Saved metrics

- accuracy
- macro precision
- macro recall
- macro F1
- eval loss

### Saved plots

- side-by-side comparison bars
- federated-minus-base delta bars
- accuracy heatmap

## 5. GPAD Summary

GPAD is used only in `main.py`.

The client-side GPAD path is:

1. extract embeddings from the MAE encoder
2. compare embeddings with the global prototype bank
3. compute an adaptive anchor threshold from assignment entropy
4. apply GPAD only to confidently anchored samples
5. route non-anchored samples through local memory updates

The implementation lives in `src/loss.py` and `src/client.py`.

## 6. Server Summary

Server-side aggregation lives in `src/server.py`.

Each round:

1. local prototypes are normalized and merged into the global bank
2. trainable adapter weights are averaged across clients
3. optional server-side EMA smooths the global adapter update

## 7. Runtime Safety

The runtime no longer trusts `torch.cuda.is_available()` alone.

Before using CUDA, the code now runs a tiny kernel-execution smoke test. If CUDA is reported but cannot actually execute kernels, the runtime does not continue as if the GPU were valid.

This behavior affects:

- `main.py`
- `base.py` through shared runtime resolution
- `evaluate.py`

## 8. Source of Truth

If this guide and the code ever disagree, trust the code:

- `main.py`
- `base.py`
- `evaluate.py`
- `src/*.py`
