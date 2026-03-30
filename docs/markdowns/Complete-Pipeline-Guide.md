# Complete Pipeline Guide

This guide summarizes the current executable workflow. When this guide and the
code disagree, trust:

- `main.py`
- `src/client.py`
- `src/server.py`
- `src/loss.py`
- `src/mae_with_adapter.py`

## 1. One Entry Point

The repository now runs from a single file:

- `main.py`

Choose the behavior by setting `RUN_MODE` inside the file:

- `federated`
- `baseline`

## 2. Shared Model Recipe

Both modes follow the same backbone recipe:

1. load `facebook/vit-mae-base`
2. freeze the original MAE backbone
3. inject residual adapters into the upper half of the encoder
4. train through the current continual dataset stream

Image preprocessing is:

1. convert to RGB
2. resize to `224 x 224`
3. convert to tensor

No ImageNet normalization is used.

## 3. Benchmark Datasets

The reported benchmark datasets are:

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

Both modes use the same extra stress stages between benchmark stages and after
the last benchmark stage:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

The full stage order is:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

These stress datasets are used only to create extra forgetting pressure. They
are not part of the reported benchmark evaluation set.

## 5. Split Policy

### Benchmark datasets

Benchmark training uses full train-side splits, except for `EuroSAT`, which is
fixed to a deterministic `22000`-sample train split and a `5000`-sample held-out
evaluation split.

### Stress datasets

Stress datasets are merged into one self-supervised training pool:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

## 6. Training Configuration

Current shared training defaults:

- `local_epochs = 1`
- `rounds_per_dataset = 3`
- `batch_size = 96`
- `client_lr = 1e-4`
- `client_weight_decay = 0.05`

## 7. Federated Mode

In `RUN_MODE = "federated"`, `main.py`:

1. builds one shared adapter-injected MAE model
2. creates two client copies
3. loads one dataset per client for the current stage
4. optimizes MAE reconstruction plus GPAD
5. keeps local prototypes and novelty buffers across dataset transitions
6. uploads only trainable adapter weights and local prototypes
7. merges global prototypes and aggregates adapter weights on the server
8. broadcasts the updated global state back to the clients

## 8. Baseline Mode

In `RUN_MODE = "baseline"`, `main.py`:

1. builds the same adapter-injected MAE model
2. walks through the exact same benchmark-plus-stress stage stream sequentially
3. optimizes MAE reconstruction only
4. preserves model and optimizer state across dataset transitions
5. does not use federation, GPAD, or prototype communication

## 9. Stage-Wise Evaluation

After every stage, both modes evaluate the benchmark datasets seen so far in two
separate passes.

### Linear probe

1. freeze the encoder
2. disable MAE masking so full images go through the encoder
3. extract frozen features
4. train one linear classifier per dataset
5. evaluate on the held-out split

Current linear-probe settings:

- epochs: `5`
- batch size: `256`
- learning rate: `1e-2`
- weight decay: `1e-4`

### Partial fine-tuning

1. start from the current checkpoint state
2. create a fresh dataset-specific model
3. freeze the lower half of the encoder
4. freeze adapters in the upper half
5. unfreeze the original transformer weights in the upper half
6. add a linear classification head
7. train on the dataset train split
8. evaluate on the held-out split

Current partial-fine-tuning settings:

- epochs: `3`
- batch size: `64`
- learning rate: `1e-4`
- weight decay: `1e-4`

## 10. Tracked Metrics

Both evaluation streams track:

- per-dataset accuracy
- average benchmark accuracy
- per-dataset forgetting
- average forgetting
- per-dataset retention ratio
- average retention ratio
- per-dataset backward transfer
- average backward transfer

The federated training history also tracks:

- total loss
- MAE loss
- GPAD loss
- anchor, local-match, and novel fractions
- global prototype count
- client prototype counts
- upload bytes
- download bytes
- total communication bytes

## 11. Saved Outputs

The run writes:

- per-round checkpoints
- final checkpoint
- JSON training history
- training summary plots
- linear-probe retention plots
- partial-fine-tuning retention plots

## 12. Runtime Safety

Before using CUDA, the runtime validates that a small CUDA kernel can actually
execute. If CUDA is reported but unusable, the code falls back early instead of
failing later inside training.
