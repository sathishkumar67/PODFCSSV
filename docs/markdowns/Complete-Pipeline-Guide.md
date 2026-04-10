# Complete Pipeline Guide

This guide explains the current executable workflow used by the repository. If this guide and the code ever disagree, treat these files as the final source of truth:

- `main.py`
- `src/client.py`
- `src/server.py`
- `src/loss.py`
- `src/mae_with_adapter.py`

## 1. Single Entry Point

The repository now runs from a single file:

- `main.py`

Set `RUN_MODE` inside that file before launching the script:

- `federated`
- `baseline`

The current checked-in default is `federated`.

Run the pipeline with:

```bash
python main.py
```

The script does not expect a mode argument on the command line.

Before either mode starts training, the pipeline now prepares every benchmark,
stress, and final-probe dataset that the run will need. Training begins only
after that dataset-preparation phase finishes.

## 2. Shared Model Construction

Both modes build the same representation backbone and only differ in how training is orchestrated afterward.

The model recipe is:

1. load `facebook/vit-mae-base`,
2. freeze the original MAE backbone,
3. inject residual adapters into the upper half of the encoder,
4. keep the continual-learning updates focused on those adapters.

Current shared model values:

- backbone: `facebook/vit-mae-base`
- embedding dimension: `768`
- adapter bottleneck dimension: `256`
- image size: `224 x 224`
- dtype: `torch.float32`

## 3. Input Pipeline

All datasets use the same preprocessing path:

1. convert the image to RGB,
2. resize it to `224 x 224`,
3. convert it to a tensor.

No ImageNet normalization is applied in the active workflow.

The dataloaders use:

- a worker cap of `16`,
- `shuffle = True` for training,
- persistent workers for stage-training loaders when multiprocessing is enabled,
- prefetching with factor `8`,
- pinned memory only when the run is actually on CUDA.

The final benchmark linear probe intentionally turns persistent workers off and
explicitly tears each probe dataloader down after use so the run does not build
up file descriptors across datasets.

Federated mode is more conservative everywhere: because the two clients already
train in parallel threads, the federated training loop and the federated final
probe both use `num_workers = 0` and `persistent_workers = False` so the run
never nests multiprocessing DataLoader workers underneath `ThreadPoolExecutor`.

## 4. Dataset Layout

The continual stream is split into benchmark datasets and stress datasets.

### Benchmark Datasets

These datasets define the core benchmark portion of the continual stream:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Benchmark client schedules:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

### Stress Datasets

These datasets are inserted to create additional distribution shift:

- `CIFAR10`
- `SVHN`
- `STL10`
- `CIFAR100`
- `Flowers102`
- `DTD`

Stress client schedules:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

### Full Stage Order

The current interleaved order is:

1. `EuroSAT` vs `GTSRB`
2. `CIFAR10` vs `SVHN`
3. `Food101` vs `Country211`
4. `STL10` vs `CIFAR100`
5. `Oxford-IIIT Pet` vs `FGVC Aircraft`
6. `Flowers102` vs `DTD`

## 5. Split Policy

### Benchmark Datasets

Benchmark training uses the full train-side split for each dataset except `EuroSAT`, which is handled through a fixed head/tail split:

- `EuroSAT`: first `22000` for train, last `5000` for held-out evaluation
- `Food101`: full `train`, evaluated on `test`
- `Oxford-IIIT Pet`: full `trainval`, evaluated on `test`
- `GTSRB`: full `train`, evaluated on `test`
- `Country211`: full `train`, evaluated on `valid`
- `FGVC Aircraft`: full `trainval`, evaluated on `test`

### Stress Datasets

Stress datasets are treated as self-supervised training pools, so all available official splits are merged into one training dataset:

- `CIFAR10`: `train + test`
- `STL10`: `train + test + unlabeled`
- `Flowers102`: `train + val + test`
- `SVHN`: `train + test + extra`
- `CIFAR100`: `train + test`
- `DTD`: `train + val + test`

## 6. Shared Training Defaults

The current common training defaults are:

- `local_epochs = 1`
- `rounds_per_dataset = 3`
- `batch_size = 512`
- `client_lr = 1e-4`
- `client_weight_decay = 0.05`
- `dataloader_persistent_workers = True` in the shared config, overridden to `False` in federated mode
- `dataloader_prefetch_factor = 8`
- `merge_threshold = 0.85`
- `server_ema_alpha = 0.1`
- `server_model_ema_alpha = 0.3`
- `k_init_prototypes = 20`

Baseline training and evaluation dataloaders currently use a worker cap of
`16`. Federated mode uses `0` workers for both round training and final probe
evaluation.

Current GPAD values:

- `gpad_base_tau = 0.85`
- `gpad_temp_gate = 0.1`
- `gpad_lambda_entropy = 0.2`
- `gpad_soft_assign_temp = 0.1`
- `lambda_proto = 0.1`

## 7. Federated Mode

When `RUN_MODE = "federated"`, the pipeline behaves as follows:

1. Prepare every benchmark, stress, and final-probe dataset before training begins.
2. Build one shared adapter-injected MAE backbone.
3. Create two client copies.
4. Reuse the prepared dataset objects for the current stage.
5. Train each client locally for the configured round and epoch budget.
6. Use GPAD only for samples whose embeddings are confidently anchored to the global prototype bank.
7. Route the remaining embeddings through each client's local prototype bank and novelty buffer.
8. Upload only trainable adapter weights and local prototypes.
9. Merge prototypes and aggregate adapter weights on the server.
10. Smooth the aggregated adapter weights with server-side EMA.
11. Broadcast the updated adapter weights and global prototype bank back to the clients.

During the first round of each stage, client-side prototype extraction now
stages temporary embeddings on CPU before K-means so very large training pools
do not accumulate full-dataset embeddings on the GPU.

During the round-training phase itself, the per-client dataloaders are rebuilt
from scratch every round and stay single-process in federated mode to avoid
worker crashes that can happen when multiprocessing loaders are driven from
multiple training threads at once.

Important continual-learning state that persists across dataset changes on each client:

- optimizer state,
- local prototype bank,
- novelty buffer.

## 8. Baseline Mode

When `RUN_MODE = "baseline"`, the pipeline keeps the same stage stream but removes all federated machinery:

1. Prepare every benchmark, stress, and final-probe dataset before training begins.
2. Build the same adapter-injected MAE backbone.
3. Walk through the exact same benchmark-plus-stress stage stream sequentially.
4. Optimize reconstruction loss only.
5. Preserve the model weights and optimizer state across dataset transitions.
6. Skip GPAD, prototype exchange, and server aggregation.

## 9. Final Linear-Probe Evaluation

After the full training stream finishes, both modes evaluate the benchmark datasets through one final linear-probe pass. Those held-out benchmark splits are prepared during startup, so the final probe reuses already prepared datasets instead of triggering late first-use downloads.

The evaluation path is:

1. load the official train-side split for each benchmark dataset,
2. load the corresponding held-out reporting split,
3. temporarily disable MAE masking so the encoder processes the full image,
4. extract frozen encoder embeddings,
5. train a linear classifier on those embeddings,
6. evaluate on the held-out split, and
7. write one final comparison summary.

Current linear-probe settings:

- epochs: `5`
- batch size: `512`
- learning rate: `1e-2`
- weight decay: `1e-4`

## 10. Final Probe Metrics

The final linear-probe evaluation records:

- per-dataset accuracy
- per-dataset train sample count
- per-dataset evaluation sample count
- average accuracy across the benchmark datasets

Federated training history also records:

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

## 11. Saved Outputs

The run writes:

- one final checkpoint
- JSON histories
- training summary plots
- one separate final benchmark linear-probe JSON summary
- one separate final linear-probe accuracy bar plot

The final checkpoint contains only run state. Training metrics, histories, and
final probe results are exported separately.

There are no mid-run checkpoints in the current workflow.

The final probe exports are written into a dedicated folder:

- `final_linear_probe/metrics`
- `final_linear_probe/plots`

## 12. Device and Numeric Safety

The current code keeps the active math path in `float32`:

- model inputs are moved to the configured device with the configured dtype,
- server aggregation aligns tensors before averaging,
- prototype-bank operations use explicit float32 tensors,
- linear-probe evaluation uses the same dtype-aware device placement.

CUDA is only used after a real runtime smoke test confirms that kernels can execute on the visible GPU.

## 13. Default Output Directories

The current output folders are:

- federated: `multidataset_outputs_2client`
- baseline: `baseline_outputs`
