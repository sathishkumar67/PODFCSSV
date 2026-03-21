# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

This repository implements a federated continual self-supervised learning pipeline built around a frozen ViT-MAE backbone, trainable residual adapters, client-side novelty handling, and a server-side global prototype bank.

## What Is In The Repo

- `main.py`
  Runs the paper-aligned Tiny ImageNet baseline with a Dirichlet non-IID client split.
- `new_main.py`
  Runs the 2-client sequential continual-learning experiment with 6 datasets, balanced sample fitting, stage-wise dataset progression, training metrics, saved plots, JSON history, and checkpoints.
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

## Run The Tiny ImageNet Baseline

```bash
python main.py
```

This script:

1. Loads `facebook/vit-mae-base`.
2. Injects adapters into the upper half of the encoder.
3. Builds a non-IID round schedule over Tiny ImageNet classes.
4. Trains clients with MAE reconstruction and GPAD.
5. Aggregates local prototypes and adapter weights at the server.
6. Broadcasts the updated adapter state back to every client.
7. Saves checkpoints, training history JSON, and plots.

Outputs are written under:

```text
checkpoints/
  checkpoints/
  metrics/
  plots/
```

## Run The 2-Client Sequential Experiment

```bash
python new_main.py
```

This script keeps the same federated-learning math but changes the data schedule:

- Client 0: `EuroSAT` -> `Oxford-IIIT Pet` -> `Flowers102`
- Client 1: `GTSRB` -> `FGVC Aircraft` -> `DTD`
- Each stage runs on 2 GPUs with one client per GPU.
- Every training split is deterministically fitted to `10000` samples so all clients run for the same number of local steps.
- Larger datasets are subsampled and smaller datasets are repeated deterministically to hit the same effective stage budget.
- Each client keeps its local prototypes, novelty buffer, and optimizer state when it switches to the next dataset, while the global model and global prototype bank also continue across stages.
- The preprocessing path does not use ImageNet mean/std normalization.

Unlike earlier sequential variants, `new_main.py` does not run linear-probe evaluation during training. That comparison is now handled separately by `evaluate.py` after the checkpoint is saved.

Outputs are written under:

```text
multidataset_outputs_2client/
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

`new_main.py` also saves:

- Stage-by-stage training history
- Communication tracking for the 2-client run
- Training summary plots for loss, routing, prototype growth, and communication

## Compare A Saved Checkpoint Later

```bash
python evaluate.py path\to\final_model.pt --datasets eurosat gtsrb svhn
```

This script rebuilds the checkpoint config, restores the saved dataset order by default, and compares the fine-tuned adapter model against the untouched Hugging Face base model with a separate linear-probe pass.

## Configuration

The central training config lives in `main.py` as `CONFIG`.

Important fields:

- `num_clients`
- `num_rounds`
- `local_epochs`
- `client_lr`
- `merge_threshold`
- `gpad_base_tau`
- `client_local_update_threshold`
- `k_init_prototypes`
- `novelty_buffer_size`

`new_main.py` extends that config with:

- `rounds_per_dataset`
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
- The checkpoint written by `new_main.py` stores the client dataset sequence so `evaluate.py` can recover the correct default evaluation order later.

## License

This project is released under the MIT License. See `LICENSE`.
