# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

This repository implements a federated continual self-supervised learning pipeline built around a frozen ViT-MAE backbone, trainable residual adapters, client-side novelty handling, and a server-side global prototype bank.

## What Is In The Repo

- `main.py`
  Runs the paper-aligned Tiny ImageNet baseline with a Dirichlet non-IID client split.
- `new_main.py`
  Runs a sequential continual-learning experiment with 10 diverse non-ImageNet datasets, 5 datasets per client, stage-wise linear evaluation, forgetting analysis, saved plots, JSON metrics, and checkpoints.
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
- Dataset preprocessing now matches the expected normalization for `facebook/vit-mae-base`.
- Training history, communication statistics, evaluation summaries, checkpoints, JSON metrics, and plots are written automatically.

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

## Run The 10-Dataset Sequential Experiment

```bash
python new_main.py
```

This script keeps the same federated-learning math but changes the data schedule:

- Client 0 trains on `EuroSAT`, `PCAM`, `Country211`, `FGVC Aircraft`, and `DTD`.
- Client 1 trains on `Oxford-IIIT Pet`, `Flowers102`, `Food101`, `GTSRB`, and `SVHN`.
- Each client completes the configured number of rounds on its current dataset before moving to the next one.
- The global model, global prototypes, local prototypes, and novelty state persist across dataset transitions.
- The full sequence spans satellite imagery, medical histopathology, geography, textures, traffic signs, and fine-grained natural-image recognition.

Additional outputs from `new_main.py`:

- Per-stage linear-probe evaluation on every seen dataset
- Final per-dataset accuracy
- Forgetting per dataset
- Accuracy heatmap
- Final-accuracy bar chart
- Forgetting bar chart

Outputs are written under:

```text
multidataset_outputs/
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

- Stage-by-stage evaluation history
- Linear-probe accuracy heatmap
- Final accuracy chart
- Forgetting chart

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
- `linear_eval_batch_size`
- `linear_eval_epochs`
- `linear_eval_lr`
- `linear_eval_weight_decay`

## Notes

- `train.ipynb` is a lightweight notebook wrapper for manual experimentation.
- The multi-dataset script intentionally uses datasets outside ImageNet-1K so it does not reuse the MAE pre-training dataset itself.
- `PCAM` uses an HDF5-backed dataset reader, so `h5py` is included in the project dependencies.
- Only trainable adapter parameters are exchanged during federation; the frozen MAE backbone is never averaged.

## License

This project is released under the MIT License. See `LICENSE`.
