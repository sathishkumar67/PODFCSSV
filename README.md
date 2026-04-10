# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

This repository currently runs from a single executable file, `main.py`. The file contains the active training pipeline, the final benchmark linear-probe evaluation, the plotting utilities, and the final checkpoint export logic used by the current research workflow.

## Current Workflow

The active experiment pipeline follows these steps:

1. Select the execution mode by setting `RUN_MODE` inside `main.py`.
2. Resolve the real runtime device by validating CUDA with a small smoke test before any heavy training starts.
3. Build the continual stage plan that interleaves benchmark datasets with additional stress datasets.
4. Prepare every benchmark, stress, and final-probe dataset before the first training round starts.
5. Build a pretrained `facebook/vit-mae-base` backbone and inject lightweight adapters into the upper half of the encoder.
6. Train through that stage stream in either federated mode or baseline mode.
7. After the full training stream finishes, freeze the final encoder and evaluate the benchmark datasets through one linear-probe pass.
8. Save the final checkpoint and training artifacts as soon as training ends, then export the final probe summary into a separate probe folder.

## Run Modes

The script does not take a command-line argument for the mode. Instead, set the constant near the top of `main.py`:

- `RUN_MODE = "federated"` runs the proposed two-client GPAD workflow.
- `RUN_MODE = "baseline"` runs the sequential reconstruction-only continual baseline.

The current checked-in default is `RUN_MODE = "federated"`.

Launch the pipeline with:

```bash
python main.py
```

## Shared Model Recipe

Both modes use the same parameter-efficient model recipe:

1. Load `facebook/vit-mae-base`.
2. Keep the original MAE backbone frozen.
3. Insert residual adapters into the upper half of the transformer encoder.
4. Train only the adapter parameters during continual learning.

Current shared model settings:

- backbone: `facebook/vit-mae-base`
- embedding dimension: `768`
- adapter bottleneck dimension: `256`
- image size: `224 x 224`
- dtype: `torch.float32`

Image preprocessing is:

1. convert each image to RGB,
2. resize it to `224 x 224`,
3. convert it to a tensor.

No ImageNet normalization is used in the current pipeline.

## Dataset Design

The continual stream is divided into benchmark datasets and stress datasets.

### Benchmark Datasets

These datasets define the core benchmark portion of the continual stream:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Client schedules:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

### Stress Datasets

These datasets are trained between benchmark stages to create stronger distribution shift:

- `CIFAR10`
- `SVHN`
- `STL10`
- `CIFAR100`
- `Flowers102`
- `DTD`

Stress schedules:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

### Full Stage Order

The current interleaved stage plan is:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

The stress datasets influence the final checkpoint through training, but they are not part of the final reported linear-probe comparison.

## Split Policy

### Benchmark Splits

Benchmark training uses the full train-side split for each dataset, except for `EuroSAT`, which is handled through a fixed head/tail split:

- `EuroSAT`: first `22000` samples for train, last `5000` samples for held-out evaluation
- `Food101`: full `train`, evaluated on `test`
- `Oxford-IIIT Pet`: full `trainval`, evaluated on `test`
- `GTSRB`: full `train`, evaluated on `test`
- `Country211`: full `train`, evaluated on `valid`
- `FGVC Aircraft`: full `trainval`, evaluated on `test`

### Stress Splits

Stress datasets are treated as self-supervised training pools, so all available official splits are merged into one training dataset:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

## Training Configuration

Current shared training defaults in `main.py`:

- `local_epochs = 1`
- `rounds_per_dataset = 3`
- `batch_size = 512`
- `client_lr = 1e-4`
- `client_weight_decay = 0.05`
- `dataloader_shuffle = True`
- `dataloader_persistent_workers = True`
- `dataloader_prefetch_factor = 8`
- `server_model_ema_alpha = 0.3`

Training and evaluation dataloaders currently use a worker cap of `16`.
Baseline stage-training dataloaders keep persistent workers enabled when
multiprocessing is active. Federated multi-GPU stage training now uses
single-process dataloaders because the two clients already run in parallel
threads, and the final benchmark linear probe uses non-persistent workers with
explicit teardown so long runs do not accumulate file descriptors across probe
datasets.

## Federated Mode

In federated mode, the training loop in `main.py` performs the following steps at every stage:

1. Prepare every benchmark, stress, and final-probe dataset before training begins.
2. Build one shared adapter-injected MAE backbone.
3. Create two client copies and place them on the selected devices.
4. Reuse the prepared dataset objects stage by stage during client training.
5. Train each client locally with MAE reconstruction loss.
6. Apply GPAD only to the samples that are confidently anchored to the global prototype bank.
7. Preserve each client's optimizer state, local prototype bank, and novelty buffer across dataset transitions.
8. Upload only trainable adapter weights and local prototypes to the server.
9. Merge client prototypes into the global bank and aggregate adapter weights on the server.
10. Broadcast the updated global adapter state and global prototype bank back to the clients for the next round.

During the first round of each stage, client-side prototype extraction now
stages temporary embeddings on CPU before K-means so large stages such as
merged `SVHN` do not exhaust GPU memory.

When federated mode runs with GPUs, each round now builds stage dataloaders
with `num_workers = 0` so the threaded two-client execution path does not nest
DataLoader multiprocessing underneath `ThreadPoolExecutor`.

Current GPAD and prototype settings:

- `lambda_proto = 0.1`
- `gpad_base_tau = 0.85`
- `gpad_temp_gate = 0.1`
- `gpad_lambda_entropy = 0.2`
- `gpad_soft_assign_temp = 0.1`
- `merge_threshold = 0.85`
- `server_ema_alpha = 0.1`
- `server_model_ema_alpha = 0.3`
- `k_init_prototypes = 20`
- `max_global_prototypes = 2000`

## Baseline Mode

In baseline mode, `main.py` uses the same benchmark-plus-stress stage order but removes all federated components:

1. Prepare every benchmark, stress, and final-probe dataset before training begins.
2. Build the same adapter-injected MAE backbone.
3. Train one single model sequentially across the full stage stream.
4. Use reconstruction loss only.
5. Preserve the model weights and optimizer state across dataset transitions.
6. Skip GPAD, prototype exchange, and server aggregation entirely.

This makes the baseline a direct continual-learning comparison against the federated method under the same stage order.

## Final Linear-Probe Evaluation

After the full training stream finishes, the final model is evaluated once on the benchmark datasets only. The held-out benchmark splits used by the probe are already prepared during the startup dataset phase, so the final probe does not need to trigger late first-use downloads.

The current pipeline uses one evaluation view:

### Frozen-Feature Linear Probe

For each benchmark dataset:

1. Load the training split used to fit the probe.
2. Load the held-out split used for reporting.
3. Temporarily disable MAE masking so full images go through the encoder.
4. Extract frozen encoder embeddings.
5. Train a linear classifier on those frozen features.
6. Evaluate on the held-out split and store the resulting accuracy.

Current linear-probe settings:

- epochs: `5`
- batch size: `512`
- learning rate: `1e-2`
- weight decay: `1e-4`

## Tracked Metrics

The final probe summary currently tracks:

- per-dataset accuracy
- per-dataset train sample count
- per-dataset evaluation sample count
- average accuracy across the benchmark datasets

Federated runs also track:

- MAE loss
- GPAD loss
- total loss
- anchor fraction
- local-match fraction
- novel fraction
- global prototype count
- client prototype counts
- upload bytes
- download bytes
- total communication bytes

## Outputs

Each run writes:

- one final checkpoint
- JSON metric and history files
- training summary plots
- one separate final benchmark linear-probe JSON summary
- one separate final linear-probe accuracy bar plot

The final checkpoint is state-only. Training metrics, histories, and final probe
results are written as separate JSON and plot artifacts.

The active output directories are:

- federated: `multidataset_outputs_2client`
- baseline: `baseline_outputs`

Inside each run directory, final probe exports are written to:

- `final_linear_probe/metrics`
- `final_linear_probe/plots`

## Runtime and Numeric Safety

The current code keeps the active math path in `float32` and validates device placement carefully:

- CUDA is only used after a real smoke test confirms kernels can execute.
- Training and evaluation inputs are moved to the configured device with the configured dtype.
- Server aggregation aligns tensors onto a common device and dtype before averaging.
- Pinned memory is enabled only when the run is actually on CUDA.

The remaining CPU transfers in the code are intentional and are used only for communication payloads, saved histories, and checkpoint export.
