# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

This repository contains three aligned entrypoints built on the same pretrained `facebook/vit-mae-base` backbone with injected residual adapters:

- `main.py`: federated continual learning with 2 clients and GPAD
- `base.py`: single-model continual learning without federation or GPAD
- `evaluate.py`: frozen-feature linear-probe comparison between saved `main.py` and `base.py` checkpoints

## Current Pipeline

### `main.py`

`main.py` is the main experiment entrypoint.

It runs a 2-client sequential schedule over 6 datasets:

- Client 0: `EuroSAT` -> `Oxford-IIIT Pet` -> `Flowers102`
- Client 1: `GTSRB` -> `FGVC Aircraft` -> `DTD`

Each stage trains one dataset per client in parallel, then moves to the next pair.

The federated stage loop is:

1. Load one dataset per client for the current stage.
2. Fit each training split to an effective budget of `10000` samples.
3. Run local MAE reconstruction and GPAD on both clients.
4. Generate or reuse local prototypes.
5. Merge prototypes and average adapter weights on the server.
6. Broadcast the updated adapter weights to both clients.
7. Preserve global prototypes, local prototypes, novelty buffers, and optimizer state across dataset transitions.
8. Save checkpoints, JSON history, and plots.

Important current behavior:

- no ImageNet normalization in the multi-dataset path
- RGB conversion + resize + `ToTensor()` only
- `EuroSAT` uses a fixed deterministic split of `10000` train and `5000` eval samples
- train splits smaller than `1000` samples are rejected
- larger splits are deterministically subsampled to the configured budget
- smaller valid splits are deterministically repeated up to the configured budget
- CUDA is used only if a real kernel-execution smoke test passes

Outputs:

```text
multidataset_outputs_2client/
  checkpoints/
  metrics/
  plots/
```

### `base.py`

`base.py` is the single-model continual baseline.

It uses the same adapter-injected MAE model as `main.py`, but:

- there is only one model
- there is no federation
- there is no GPAD
- training uses only reconstruction loss
- the full train split is used for each dataset
- model weights and optimizer state persist across dataset transitions

The dataset order follows the same stage order flattened into one sequence:

- `EuroSAT`
- `GTSRB`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`
- `Flowers102`
- `DTD`

Outputs:

```text
base_outputs/
  checkpoints/
  metrics/
  plots/
```

### `evaluate.py`

`evaluate.py` compares a saved federated checkpoint and a saved baseline checkpoint.

The evaluation flow is:

1. Load both checkpoints into the same adapter-injected MAE architecture.
2. Freeze both models.
3. Use encoder-only inference with `mask_ratio = 0.0`.
4. Extract frozen features from full images.
5. Train one dataset-specific linear probe per checkpoint.
6. Evaluate on the official held-out split for that dataset.
7. Save JSON metrics and comparison plots.

Probe-fit policy:

- if train split `< 1000`: skip the dataset
- if train split is `1000..10000`: use the full train split
- if train split `> 10000`: use a deterministic `4000`-sample subset

Held-out evaluation policy:

- `EuroSAT` uses the repo-defined deterministic `10000`-train / `5000`-eval split
- if both `val` and `test` exist in the repo loader, both are used
- if only one official held-out split exists, that split is used
- if the repo only has a generated split and no official held-out split, the dataset is skipped

Saved evaluation metrics:

- accuracy
- macro precision
- macro recall
- macro F1
- eval loss

Saved evaluation artifacts:

- side-by-side bar charts
- federated-minus-base delta charts
- accuracy heatmap
- JSON summary with evaluated and skipped datasets

## Core Modules

- `src/mae_with_adapter.py`
  Freezes ViT-MAE and injects residual adapters into the upper half of the encoder.
- `src/loss.py`
  Implements GPAD, including adaptive thresholds, anchor masks, and the gated prototype loss.
- `src/client.py`
  Handles local MAE training, GPAD routing, local prototype updates, novelty buffering, and local spherical K-means.
- `src/server.py`
  Handles prototype-bank updates, FedAvg-style adapter aggregation, and server-side EMA smoothing.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Federated continual run:

```bash
python main.py
```

Single-model continual baseline:

```bash
python base.py
```

Checkpoint comparison:

```bash
python evaluate.py path\to\federated_final_model.pt path\to\base_final_model.pt
```

Optional dataset override:

```bash
python evaluate.py path\to\federated_final_model.pt path\to\base_final_model.pt --datasets gtsrb fgvcaircraft flowers102 dtd
```

## Notes

- Only trainable adapter parameters are exchanged during federation.
- The pretrained MAE backbone remains frozen in all three entrypoints.
- `main.py` and `base.py` both save the dataset sequence inside checkpoints so later evaluation can recover the correct default order.
- If a CUDA runtime reports availability but cannot execute kernels, the repo now falls back earlier instead of failing deep inside the first convolution.

## License

This project is released under the MIT License.
