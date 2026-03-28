# PODFCSSV

Prototype-Oriented Distillation for Federated Continual Self-Supervised Vision.

The current repository uses one executable pipeline file: `main.py`.

Set `RUN_MODE` in `main.py` to one of:

- `federated`
- `baseline`

## Benchmark

Reported benchmark datasets:

- `EuroSAT`
- `GTSRB`
- `Food101`
- `Country211`
- `Oxford-IIIT Pet`
- `FGVC Aircraft`

Benchmark client schedule:

- Client 0: `EuroSAT -> Food101 -> Oxford-IIIT Pet`
- Client 1: `GTSRB -> Country211 -> FGVC Aircraft`

Retention-stress datasets inserted into both modes:

- Client 0: `CIFAR10 -> STL10 -> Flowers102`
- Client 1: `SVHN -> CIFAR100 -> DTD`

The stress datasets create harder distribution shifts during training, but they are never included in the reported benchmark evaluation set.

## Shared Model and Budget

Both modes use `facebook/vit-mae-base`, a frozen MAE backbone, residual adapters in the upper half of the encoder, and preprocessing of `RGB -> resize to 224 x 224 -> ToTensor()`. No ImageNet normalization is used.

Shared training defaults:

- `local_epochs = 1`
- `rounds_per_dataset = 3`
- `client_lr = 1e-4`
- `client_weight_decay = 0.05`
- federated batch size `64`
- baseline batch size `64`

Training budget policy:

- `EuroSAT` uses a fixed split of `10000` train and `5000` held-out evaluation samples
- every benchmark training split targets `10000` effective samples
- splits below `1000` samples are rejected
- larger splits are deterministically subsampled
- smaller valid splits are deterministically repeated up to the target budget

The baseline uses the same budget policy as the federated benchmark.

## Modes

`RUN_MODE = "federated"`:

1. loads one dataset per client for the current stage
2. alternates benchmark stages with retention-stress stages
3. trains locally with MAE reconstruction and GPAD
4. merges local prototypes and aggregates only the trainable adapter weights
5. preserves global memory and carries client-local memory forward by enriching the existing local prototype banks instead of resetting them
6. evaluates on all benchmark datasets seen so far after every stage
7. saves checkpoints, histories, metrics, and plots

`RUN_MODE = "baseline"`:

1. builds one adapter-injected MAE model
2. trains sequentially over both the benchmark datasets and the same stress datasets used by the federated run
3. uses reconstruction loss only
4. preserves the same model and optimizer state across stages
5. runs the same stage-wise benchmark evaluation after every stage
6. saves checkpoints, histories, metrics, and plots

## Built-In Evaluation

After each stage, `main.py` performs frozen-feature linear-probe evaluation on every benchmark dataset seen so far. The stage summaries track:

- per-dataset accuracy
- average benchmark accuracy
- forgetting
- retention ratio
- backward transfer

Saved plots include:

- training summary
- stage accuracy heatmap
- stage metric curves
- final forgetting bar chart

## Download Safety

The default publishable schedule avoids manual-setup datasets such as `FER2013`, `PCAM`, and `Stanford Cars`. The dependency metadata now includes `scipy` so the default `SVHN` and `Flowers102` paths install cleanly in a fresh environment.

To smoke-test dataset downloads without running training, use:

```bash
python test.py
```

`test.py` downloads every benchmark and stress dataset one by one, touches the required splits, deletes that dataset folder, and then moves on to the next dataset.

## Run

```bash
python main.py
```

No separate `base.py` or `evaluate.py` entrypoint is required for the current retention-analysis workflow.

## License

This project is released under the MIT License.
