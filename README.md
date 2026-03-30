# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

The repository now uses a single executable experiment file: `main.py`.

## Run Modes

Set `RUN_MODE` near the top of `main.py` before launching the script:

- `federated`
- `baseline`

Run the pipeline with:

```bash
python main.py
```

The script does not expect a command-line mode argument. The selected mode is
read directly from the `RUN_MODE` constant.

## Shared Model

Both modes use the same representation backbone:

- pretrained model: `facebook/vit-mae-base`
- embedding dimension: `768`
- adapter bottleneck dimension: `256`
- input size: `224 x 224`
- preprocessing: `RGB -> Resize(224, 224) -> ToTensor()`
- normalization: no ImageNet normalization

Adapters are injected only into the upper half of the ViT-MAE encoder. The MAE
backbone stays frozen during continual training, and only the adapter weights
are exchanged in the federated mode.

## Dataset Flow

### Benchmark Datasets

These datasets define the reported continual-learning benchmark:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

### Retention-Stress Datasets

These datasets are inserted between benchmark stages and again after the last
benchmark stage to create additional distribution shift:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

The full stage order is:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

The stress datasets are trained in both modes but are excluded from the
reported benchmark evaluation curves.

## Split Policy

### Benchmark Training Splits

The benchmark datasets use the full train-side splits in the current pipeline,
except for `EuroSAT`, which uses a fixed deterministic split:

- `EuroSAT`: `22000` train, `5000` eval
- `Food101`: full `train`
- `Oxford-IIIT Pet`: full `trainval`
- `GTSRB`: full `train`
- `Country211`: full `train`
- `FGVC Aircraft`: full `trainval`

### Stress Training Splits

The stress datasets are treated as self-supervised training pools, so all
available official splits are merged into one training dataset:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

### Evaluation Splits

Benchmark evaluation always uses held-out data:

- `EuroSAT`: fixed `5000`-sample held-out split
- `Food101`: `test`
- `Oxford-IIIT Pet`: `test`
- `GTSRB`: `test`
- `Country211`: `test`
- `FGVC Aircraft`: `test`

Stress datasets are not part of the reported benchmark evaluation set.

## Training Configuration

Current shared training defaults:

- `local_epochs = 1`
- `rounds_per_dataset = 3`
- `batch_size = 96`
- `client_lr = 1e-4`
- `client_weight_decay = 0.05`

The baseline and federated modes now use the same batch size and the same full
split policy so the comparison stays aligned.

## Federated Mode

`RUN_MODE = "federated"` performs the proposed method:

1. Build one adapter-injected ViT-MAE backbone.
2. Create two clients, one per selected device when GPUs are available.
3. Load one dataset per client for the current stage.
4. Train with MAE reconstruction plus GPAD on globally anchored samples.
5. Maintain client-local prototypes and novelty buffers across dataset changes.
6. Upload only the trainable adapter weights and local prototype banks.
7. Merge prototypes and aggregate adapter weights at the server.
8. Broadcast the updated global adapter state and global prototypes back to the clients.

Tracked federated communication includes:

- uploaded bytes
- downloaded bytes
- total communication bytes
- global prototype count
- local prototype counts per client

## Baseline Mode

`RUN_MODE = "baseline"` performs the non-federated continual baseline:

1. Build the same adapter-injected ViT-MAE backbone.
2. Walk through the exact same benchmark-plus-stress stage stream sequentially.
3. Optimize MAE reconstruction loss only.
4. Preserve the same model state and optimizer state across dataset changes.

No GPAD, no server aggregation, and no prototype communication are used in the
baseline mode.

## Stage-Wise Evaluation

After every stage in both modes, `main.py` evaluates the benchmark datasets
seen so far in two separate ways.

### 1. Linear Probe

This stage measures frozen representation quality:

1. Freeze the encoder.
2. Disable MAE masking so full images pass through the encoder.
3. Extract benchmark features.
4. Train one linear classifier per dataset.
5. Evaluate on the held-out split.

Current linear-probe settings:

- epochs: `5`
- batch size: `256`
- learning rate: `1e-2`
- weight decay: `1e-4`

### 2. Partial Fine-Tuning

This stage measures transfer quality from the current checkpoint:

1. Start from the current checkpoint state.
2. Create a fresh dataset-specific model.
3. Freeze the lower half of the encoder.
4. In the upper half, keep adapters frozen and unfreeze the original transformer weights.
5. Add a dataset-specific linear classification head.
6. Train that model on the dataset train split.
7. Evaluate on the held-out split.

Current partial-fine-tuning settings:

- epochs: `3`
- batch size: `64`
- learning rate: `1e-4`
- weight decay: `1e-4`

## Retention Metrics

Both evaluation streams track stage-wise continual-learning metrics on the
benchmark datasets seen so far:

- per-dataset accuracy
- average benchmark accuracy
- per-dataset forgetting
- average forgetting
- per-dataset retention ratio
- average retention ratio
- per-dataset backward transfer
- average backward transfer

## Saved Outputs

The pipeline writes:

- per-round checkpoints
- final checkpoint
- JSON training history
- training summary plots
- linear-probe retention plots
- partial-fine-tuning retention plots

The active output directories are:

- federated: `multidataset_outputs_2client`
- baseline: `baseline_outputs`

## Safety and Reproducibility

The default publishable schedule avoids manual-setup datasets such as:

- `FER2013`
- `PCAM`
- `Stanford Cars`

Before using CUDA, the runtime runs a small kernel smoke test to confirm that a
GPU can actually execute the operations required by the training loop.

## Active Source Files

The current source of truth lives in:

- `main.py`
- `src/client.py`
- `src/server.py`
- `src/loss.py`
- `src/mae_with_adapter.py`

## License

This project is released under the MIT License.
