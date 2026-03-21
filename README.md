# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

This repository implements a federated continual self-supervised learning pipeline built around a frozen ViT-MAE backbone, trainable residual adapters, client-side novelty handling, and a server-side global prototype bank.

## What Is In The Repo

- `main.py`
  Runs the 2-client sequential continual-learning experiment with 6 datasets, balanced sample fitting, stage-wise dataset progression, training metrics, saved plots, JSON history, and checkpoints.
- `base.py`
  Runs the single-model continual baseline on the same 6-dataset sequence with reconstruction-only training, post-dataset evaluation, forgetting metrics, and comparison plots.
- `evaluate.py`
  Loads a saved checkpoint later and compares it against the base Hugging Face model with a separate linear-probe pass.
- `src/mae_with_adapter.py`
  Freezes the backbone and injects adapters into the upper half of the transformer.
- `src/loss.py`
  Implements GPAD with entropy-adaptive thresholding and the paper's `>=` anchor rule.
- `src/client.py`
  Handles local MAE training, GPAD routing, local EMA prototype updates, novelty buffering, and spherical K-means.
- `src/server.py`
  Handles global prototype merge-or-add, FedAvg aggregation, and server-side EMA smoothing.

## Key Corrections In This Version

- GPAD gradients now reach the adapters instead of being cut off by detached embeddings.
- Server-aggregated adapter weights are broadcast back to every client before the next round.
- Weight aggregation is fixed for true multi-client runs.
- Round-1 prototype extraction and later GPAD routing now use the same embedding definition.
- The Dirichlet partition no longer drops leftover samples after integer rounding.
- Configured local epochs are now honored during client-side training.
- Tiny ImageNet preprocessing matches the expected normalization for `facebook/vit-mae-base`.
- The 2-client sequential run intentionally avoids ImageNet-style normalization and uses only RGB conversion, resize, and `ToTensor()`.
- Training history, communication statistics, checkpoints, JSON metrics, and plots are written automatically.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Federated Sequential Experiment

```bash
python main.py
```

This script:

1. Loads `facebook/vit-mae-base`.
2. Injects adapters into the upper half of the encoder.
3. Assigns one client per GPU across the 2-client sequential schedule.
4. Trains clients with MAE reconstruction and GPAD.
5. Aggregates local prototypes and adapter weights at the server.
6. Broadcasts the updated adapter state back to every client.
7. Saves checkpoints, training history JSON, and plots.

Outputs are written under:

```text
multidataset_outputs/
  checkpoints/
  metrics/
  plots/
```

## Run The Single-Model Continual Baseline

```bash
python base.py
```

This script keeps the same adapter-injected ViT-MAE model but removes federation and GPAD:

- Dataset order: `EuroSAT` -> `GTSRB` -> `Oxford-IIIT Pet` -> `FGVC Aircraft` -> `Flowers102` -> `DTD`
- One model is trained on the full train split of each dataset in sequence.
- Evaluation runs after each dataset using the non-train split(s) of every seen dataset.
- The baseline saves forgetting metrics and plots so catastrophic forgetting can be measured directly.

Outputs are written under:

```text
base_outputs/
  checkpoints/
  metrics/
  plots/
```

## Publication-Oriented Artifacts

Both entrypoints save:

- JSON training history
- Adapter-only checkpoints
- Global prototype snapshots
- Loss curves
- Routing-fraction plots
- Communication-cost plots

`base.py` also saves:

- Stage-by-stage evaluation history
- Accuracy heatmaps and final accuracy bars
- Forgetting plots for the continual baseline

## Compare A Saved Checkpoint Later

```bash
python evaluate.py path\to\final_model.pt --datasets eurosat gtsrb svhn
```

This script rebuilds the checkpoint config, restores the saved dataset order by default, and compares the fine-tuned adapter model against the untouched Hugging Face base model with a separate linear-probe pass.

## Configuration

The central training config lives in `main.py` as `CONFIG`.

Important fields:

- `num_clients`
- `rounds_per_dataset`
- `local_epochs`
- `client_lr`
- `merge_threshold`
- `gpad_base_tau`
- `client_local_update_threshold`
- `k_init_prototypes`
- `novelty_buffer_size`

The sequential entrypoints extend that config with:

- `train_samples_per_dataset`
- `min_train_samples_per_dataset`
- `max_global_prototypes`
- `linear_eval_batch_size`
- `linear_eval_epochs`
- `linear_eval_lr`
- `linear_eval_weight_decay`

## Notes

- `train.ipynb` is a lightweight notebook wrapper for manual experimentation.
- The multi-dataset script intentionally spans satellite, medical, sentiment-text, scene, traffic, face, pet, food, and character domains instead of reusing ImageNet-1K itself.
- `PCAM` uses an HDF5-backed dataset reader, so `h5py` is included in the project dependencies.
- Only trainable adapter parameters are exchanged during federation; the frozen MAE backbone is never averaged.
- The checkpoints written by `main.py` and `base.py` both store the dataset sequence so later evaluation can recover the correct default order.

## License

This project is released under the MIT License. See `LICENSE`.
