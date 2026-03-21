"""Compare the federated checkpoint and the baseline checkpoint with linear probes.

The evaluation path is intentionally separate from training:
1. Load the saved federated model and the saved baseline model.
2. Freeze both models and keep only the encoder path active.
3. Disable MAE masking so full images pass through the encoder.
4. Extract frozen features for one dataset at a time.
5. Fit one dataset-specific linear probe per model.
6. Evaluate on the official non-train split(s) and save comparison outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from main import (
    DATASET_DISPLAY_NAMES,
    MULTI_DATASET_CONFIG,
    build_base_model,
    build_dataset_order_by_stage,
    create_standard_dataloader,
    load_named_dataset,
    load_named_evaluation_dataset,
    resolve_runtime_config,
    stable_int_from_parts,
    set_random_seed,
)

logger = logging.getLogger("PODFCSSV_Evaluate")

TORCH_DTYPE_LOOKUP: Dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}

GENERATED_EVAL_SPLIT_DATASETS = {
    "eurosat",
    "caltech101",
    "caltech256",
    "sun397",
}

PROBE_TRAIN_CAP_THRESHOLD = 10000
PROBE_TRAIN_SAMPLE_BUDGET = 4000
PROBE_MIN_TRAIN_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    """Parse the two checkpoint paths and the optional evaluation settings."""
    parser = argparse.ArgumentParser(
        description="Compare the federated checkpoint against the single-model baseline with linear probes.",
    )
    parser.add_argument(
        "federated_checkpoint_path",
        type=Path,
        help="Path to the federated main.py checkpoint.",
    )
    parser.add_argument(
        "base_checkpoint_path",
        type=Path,
        help="Path to the single-model base.py checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DISPLAY_NAMES.keys()),
        default=None,
        help="Datasets to evaluate. Defaults to the checkpoint dataset order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluate_outputs"),
        help="Directory where JSON metrics and plots will be saved.",
    )
    return parser.parse_args()


def deserialize_config(serialized_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config values loaded from a checkpoint back into runtime objects."""
    config = dict(serialized_config)
    dtype_value = config.get("dtype")
    if isinstance(dtype_value, str) and dtype_value in TORCH_DTYPE_LOOKUP:
        config["dtype"] = TORCH_DTYPE_LOOKUP[dtype_value]
    return config


def build_runtime_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Build a runtime config by combining defaults with checkpoint metadata."""
    config = dict(MULTI_DATASET_CONFIG)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config.update(deserialize_config(checkpoint.get("config", {})))
    resolve_runtime_config(config)
    return config


def default_dataset_order(*configs: Dict[str, Any]) -> List[str]:
    """Recover the saved stage order so evaluation follows the training sequence."""
    for config in configs:
        dataset_sequence = config.get("client_dataset_sequence")
        if isinstance(dataset_sequence, dict) and dataset_sequence:
            return build_dataset_order_by_stage(dataset_sequence)
    return build_dataset_order_by_stage()


def freeze_encoder_model(model: nn.Module) -> None:
    """Freeze every parameter and disable MAE masking for encoder-only evaluation."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "config") and hasattr(model.config, "mask_ratio"):
        model.config.mask_ratio = 0.0
    if hasattr(model, "vit") and hasattr(model.vit, "config") and hasattr(model.vit.config, "mask_ratio"):
        model.vit.config.mask_ratio = 0.0
    model.eval()


def choose_devices(usable_gpu_count: int) -> Tuple[str, str]:
    """Choose one device per model based on the usable-GPU count."""
    if usable_gpu_count >= 2:
        return "cuda:0", "cuda:1"
    if usable_gpu_count == 1:
        return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def supports_official_eval_split(dataset_name: str) -> bool:
    """Return whether the dataset has an official held-out split in this repo."""
    return dataset_name not in GENERATED_EVAL_SPLIT_DATASETS


def load_probe_train_dataset(
    dataset_name: str,
    config: Dict[str, Any],
) -> torch.utils.data.Dataset | None:
    """Load the train split used to fit the linear probe.

    The probe-fit rule is:
    1. Skip datasets with fewer than 1,000 train samples.
    2. Use the full train split when it has 1,000 to 10,000 samples.
    3. Use a deterministic 4,000-sample subset when the train split is larger.
    """
    full_train_dataset = load_named_dataset(
        dataset_name=dataset_name,
        data_root=config["data_root"],
        image_size=config["image_size"],
        train=True,
        seed=config["seed"],
        max_samples=None,
        min_samples=None,
    )
    train_size = len(full_train_dataset)
    if train_size < PROBE_MIN_TRAIN_SAMPLES:
        logger.warning(
            "Skipping %s because its train split has only %s samples (< %s).",
            DATASET_DISPLAY_NAMES[dataset_name],
            train_size,
            PROBE_MIN_TRAIN_SAMPLES,
        )
        return None
    if train_size > PROBE_TRAIN_CAP_THRESHOLD:
        return load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
            max_samples=PROBE_TRAIN_SAMPLE_BUDGET,
            min_samples=PROBE_MIN_TRAIN_SAMPLES,
        )
    return full_train_dataset


def load_probe_eval_dataset(
    dataset_name: str,
    config: Dict[str, Any],
) -> torch.utils.data.Dataset | None:
    """Load the official held-out split or split-combination for one dataset."""
    if not supports_official_eval_split(dataset_name):
        logger.warning(
            "Skipping %s because it does not have an official eval/test split in this pipeline.",
            DATASET_DISPLAY_NAMES[dataset_name],
        )
        return None
    eval_dataset = load_named_evaluation_dataset(
        dataset_name=dataset_name,
        data_root=config["data_root"],
        image_size=config["image_size"],
        seed=config["seed"],
        max_samples=None,
        min_samples=None,
    )
    if len(eval_dataset) == 0:
        logger.warning(
            "Skipping %s because its eval/test split is empty.",
            DATASET_DISPLAY_NAMES[dataset_name],
        )
        return None
    return eval_dataset


def load_checkpoint_model(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: str,
) -> nn.Module:
    """Load one checkpoint into the shared adapter-injected MAE architecture."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = dict(config)
    model_config["device"] = device
    model = build_base_model(model_config)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise KeyError("The checkpoint does not contain a 'model_state_dict' entry.")
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device=device, dtype=config["dtype"])
    freeze_encoder_model(model)
    return model


def pool_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pool patch tokens into one feature vector per image."""
    if hidden_states.size(1) > 1:
        hidden_states = hidden_states[:, 1:, :]
    return hidden_states.mean(dim=1)


@torch.inference_mode()
def collect_encoder_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect frozen encoder features and labels for one dataset split.

    This helper is the feature-extraction stage:
    1. Move images to the selected device.
    2. Run the encoder with masking disabled.
    3. Pool the last hidden state into one vector per image.
    4. Return the features and labels on CPU for probe training.
    """
    feature_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device=device, dtype=dtype, non_blocking=device.startswith("cuda"))
        outputs = model.vit(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        features = pool_hidden_states(hidden_states)
        feature_batches.append(features.float().cpu())
        label_batches.append(torch.as_tensor(labels, dtype=torch.long).cpu())

    if not feature_batches:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


def remap_labels(
    train_labels: torch.Tensor,
    eval_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Map arbitrary dataset labels into ``0..num_classes-1`` for the probe."""
    unique_labels = torch.unique(torch.cat([train_labels, eval_labels], dim=0)).tolist()
    mapping = {original_label: new_label for new_label, original_label in enumerate(unique_labels)}
    remapped_train = train_labels.clone()
    remapped_eval = eval_labels.clone()
    for original_label, new_label in mapping.items():
        remapped_train[train_labels == original_label] = new_label
        remapped_eval[eval_labels == original_label] = new_label
    return remapped_train, remapped_eval, len(unique_labels)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Compute the classification metrics saved for one dataset and one model."""
    loss = nn.CrossEntropyLoss()(logits, labels).item()
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())

    flat_indices = labels * num_classes + predictions
    confusion = torch.bincount(flat_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
    true_positives = confusion.diag()
    precision = true_positives / confusion.sum(dim=0).clamp_min(1.0)
    recall = true_positives / confusion.sum(dim=1).clamp_min(1.0)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp_min(1e-8)

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "eval_loss": float(loss),
    }


def train_and_eval_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    device: str,
    config: Dict[str, Any],
    shuffle_seed: int,
) -> Dict[str, Any]:
    """Fit one linear probe and evaluate it on the held-out split.

    Both checkpoints follow the same probe recipe:
    1. Build a dataset-specific linear head.
    2. Train it on frozen encoder features from the train split.
    3. Evaluate it on the official held-out split.
    4. Return a compact metric dictionary for comparison plots and JSON logs.
    """
    if train_features.numel() == 0 or eval_features.numel() == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "eval_loss": 0.0,
        }

    train_labels, eval_labels, num_classes = remap_labels(train_labels, eval_labels)
    probe = nn.Linear(train_features.size(1), num_classes).to(device)
    optimizer = optim.AdamW(
        probe.parameters(),
        lr=config["linear_eval_lr"],
        weight_decay=config["linear_eval_weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(train_features, train_labels)
    train_generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["linear_eval_batch_size"],
        shuffle=True,
        generator=train_generator,
        num_workers=config["linear_eval_num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=False,
    )

    for _ in range(config["linear_eval_epochs"]):
        probe.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=device.startswith("cuda"))
            batch_labels = batch_labels.to(device, non_blocking=device.startswith("cuda"))
            logits = probe(batch_features)
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    eval_logits: List[torch.Tensor] = []
    with torch.inference_mode():
        for start_index in range(0, eval_features.size(0), config["linear_eval_batch_size"]):
            end_index = start_index + config["linear_eval_batch_size"]
            batch_features = eval_features[start_index:end_index].to(device, non_blocking=device.startswith("cuda"))
            eval_logits.append(probe(batch_features).cpu())
    return compute_classification_metrics(torch.cat(eval_logits, dim=0), eval_labels, num_classes)


def evaluate_single_model_on_dataset(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    device: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate one frozen encoder on one dataset with a fresh linear probe."""
    logger.info("%s | starting probe on %s (%s)", model_name, DATASET_DISPLAY_NAMES[dataset_name], device)
    train_dataset = load_probe_train_dataset(dataset_name=dataset_name, config=config)
    eval_dataset = load_probe_eval_dataset(dataset_name=dataset_name, config=config)
    if train_dataset is None or eval_dataset is None:
        skip_reason = "train_split_below_minimum" if train_dataset is None else "missing_or_empty_eval_split"
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "eval_loss": 0.0,
            "num_train_samples": 0,
            "num_eval_samples": 0,
            "train_split_used": 0,
            "skipped": 1.0,
            "skip_reason": skip_reason,
        }

    train_loader = create_standard_dataloader(
        dataset=train_dataset,
        batch_size=config["linear_eval_batch_size"],
        num_workers=config["linear_eval_num_workers"],
        pin_memory=config["pin_memory"],
        shuffle=False,
    )
    eval_loader = create_standard_dataloader(
        dataset=eval_dataset,
        batch_size=config["linear_eval_batch_size"],
        num_workers=config["linear_eval_num_workers"],
        pin_memory=config["pin_memory"],
        shuffle=False,
    )

    model_dtype = next(model.parameters()).dtype
    train_features, train_labels = collect_encoder_features(model, train_loader, device, model_dtype)
    eval_features, eval_labels = collect_encoder_features(model, eval_loader, device, model_dtype)
    metrics = train_and_eval_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        eval_features=eval_features,
        eval_labels=eval_labels,
        device=device,
        config=config,
        shuffle_seed=stable_int_from_parts(config["seed"], dataset_name, "probe_shuffle"),
    )
    metrics["num_train_samples"] = int(train_labels.numel())
    metrics["num_eval_samples"] = int(eval_labels.numel())
    metrics["train_split_used"] = int(len(train_dataset))
    metrics["skipped"] = 0.0
    metrics["skip_reason"] = ""
    logger.info(
        "%s | %s | train=%s | eval=%s | acc=%.4f | macro_f1=%.4f",
        model_name,
        DATASET_DISPLAY_NAMES[dataset_name],
        metrics["num_train_samples"],
        metrics["num_eval_samples"],
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def print_comparison_table(results: Dict[str, Dict[str, Dict[str, Any]]], dataset_names: Sequence[str]) -> None:
    """Print a compact terminal table for the federated-vs-base comparison."""
    header = f"{'Dataset':<20} {'Fed Acc':>10} {'Base Acc':>10} {'Delta':>10} {'Fed F1':>10} {'Base F1':>10}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for dataset_name in dataset_names:
        federated = results["federated"][dataset_name]
        base = results["base"][dataset_name]
        if federated.get("skipped") or base.get("skipped"):
            print(f"{DATASET_DISPLAY_NAMES[dataset_name]:<20} {'SKIPPED':>10} {'SKIPPED':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        print(
            f"{DATASET_DISPLAY_NAMES[dataset_name]:<20} "
            f"{federated['accuracy']:>10.4f} "
            f"{base['accuracy']:>10.4f} "
            f"{(federated['accuracy'] - base['accuracy']):>10.4f} "
            f"{federated['macro_f1']:>10.4f} "
            f"{base['macro_f1']:>10.4f}"
        )
    print(separator)


def get_evaluated_dataset_names(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    dataset_names: Sequence[str],
) -> List[str]:
    """Return only the datasets that produced valid held-out evaluation results."""
    return [
        dataset_name
        for dataset_name in dataset_names
        if not results["federated"][dataset_name].get("skipped")
        and not results["base"][dataset_name].get("skipped")
    ]


def save_results(output_dir: Path, dataset_names: Sequence[str], results: Dict[str, Dict[str, Dict[str, Any]]]) -> Path:
    """Write the full evaluation payload and summary metrics to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_dataset_names = get_evaluated_dataset_names(results, dataset_names)
    skipped_dataset_names = [
        dataset_name for dataset_name in dataset_names if dataset_name not in evaluated_dataset_names
    ]

    def average_metric(model_name: str, metric_key: str) -> float:
        if not evaluated_dataset_names:
            return 0.0
        return float(
            np.mean([results[model_name][dataset_name][metric_key] for dataset_name in evaluated_dataset_names])
        )

    payload = {
        "datasets": list(dataset_names),
        "evaluated_datasets": evaluated_dataset_names,
        "skipped_datasets": skipped_dataset_names,
        "results": results,
        "summary": {
            "num_requested_datasets": len(dataset_names),
            "num_evaluated_datasets": len(evaluated_dataset_names),
            "num_skipped_datasets": len(skipped_dataset_names),
            "federated_average_accuracy": average_metric("federated", "accuracy"),
            "base_average_accuracy": average_metric("base", "accuracy"),
            "federated_average_macro_precision": average_metric("federated", "macro_precision"),
            "base_average_macro_precision": average_metric("base", "macro_precision"),
            "federated_average_macro_recall": average_metric("federated", "macro_recall"),
            "base_average_macro_recall": average_metric("base", "macro_recall"),
            "federated_average_macro_f1": average_metric("federated", "macro_f1"),
            "base_average_macro_f1": average_metric("base", "macro_f1"),
            "federated_average_eval_loss": average_metric("federated", "eval_loss"),
            "base_average_eval_loss": average_metric("base", "eval_loss"),
            "federated_accuracy_wins": int(
                sum(
                    results["federated"][dataset_name]["accuracy"] > results["base"][dataset_name]["accuracy"]
                    for dataset_name in evaluated_dataset_names
                )
            ),
            "base_accuracy_wins": int(
                sum(
                    results["base"][dataset_name]["accuracy"] > results["federated"][dataset_name]["accuracy"]
                    for dataset_name in evaluated_dataset_names
                )
            ),
        },
    }
    output_path = output_dir / "linear_probe_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def plot_comparison_bars(
    output_dir: Path,
    dataset_names: Sequence[str],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    metric_key: str,
    title: str,
    filename: str,
) -> Path:
    """Plot one side-by-side comparison bar chart for a chosen metric."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_dataset_names = get_evaluated_dataset_names(results, dataset_names)
    output_path = output_dir / filename
    if not evaluated_dataset_names:
        return output_path

    figure, axis = plt.subplots(figsize=(max(8, len(evaluated_dataset_names) * 1.4), 5))
    x_positions = np.arange(len(evaluated_dataset_names))
    width = 0.36
    federated_values = [results["federated"][name][metric_key] for name in evaluated_dataset_names]
    base_values = [results["base"][name][metric_key] for name in evaluated_dataset_names]
    axis.bar(x_positions - width / 2, federated_values, width=width, label="Federated")
    axis.bar(x_positions + width / 2, base_values, width=width, label="Base")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([DATASET_DISPLAY_NAMES[name] for name in evaluated_dataset_names], rotation=45, ha="right")
    if metric_key != "eval_loss":
        axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_delta_bars(
    output_dir: Path,
    dataset_names: Sequence[str],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    metric_key: str,
    title: str,
    filename: str,
) -> Path:
    """Plot federated-minus-base deltas so gains and regressions are obvious."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_dataset_names = get_evaluated_dataset_names(results, dataset_names)
    output_path = output_dir / filename
    if not evaluated_dataset_names:
        return output_path

    deltas = [
        results["federated"][dataset_name][metric_key] - results["base"][dataset_name][metric_key]
        for dataset_name in evaluated_dataset_names
    ]
    colors = ["tab:green" if delta >= 0.0 else "tab:red" for delta in deltas]

    figure, axis = plt.subplots(figsize=(max(8, len(evaluated_dataset_names) * 1.3), 5))
    x_positions = np.arange(len(evaluated_dataset_names))
    axis.bar(x_positions, deltas, color=colors)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([DATASET_DISPLAY_NAMES[name] for name in evaluated_dataset_names], rotation=45, ha="right")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_accuracy_heatmap(
    output_dir: Path,
    dataset_names: Sequence[str],
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Path:
    """Plot an accuracy heatmap with one row per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated_dataset_names = get_evaluated_dataset_names(results, dataset_names)
    output_path = output_dir / "linear_probe_accuracy_heatmap.png"
    if not evaluated_dataset_names:
        return output_path

    matrix = np.array(
        [
            [results["federated"][name]["accuracy"] for name in evaluated_dataset_names],
            [results["base"][name]["accuracy"] for name in evaluated_dataset_names],
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(max(8, len(evaluated_dataset_names) * 1.3), 4))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(evaluated_dataset_names)))
    axis.set_xticklabels([DATASET_DISPLAY_NAMES[name] for name in evaluated_dataset_names], rotation=45, ha="right")
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Federated", "Base"])
    axis.set_title("Linear-Probe Accuracy Heatmap")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    """Run the full dual-checkpoint linear-probe comparison."""
    args = parse_args()
    federated_config = build_runtime_config(args.federated_checkpoint_path)
    base_config = build_runtime_config(args.base_checkpoint_path)
    set_random_seed(federated_config.get("seed"))
    dataset_names = args.datasets or default_dataset_order(federated_config, base_config)

    eval_config = dict(MULTI_DATASET_CONFIG)
    eval_config.update(
        {
            "seed": federated_config.get("seed", base_config.get("seed", 42)),
            "data_root": federated_config.get("data_root", base_config.get("data_root", "./data")),
            "image_size": federated_config.get("image_size", base_config.get("image_size", 224)),
            "dtype": federated_config.get("dtype", torch.float32),
        }
    )
    resolve_runtime_config(eval_config)
    eval_config["pin_memory"] = eval_config["device"] == "cuda"
    federated_device, base_device = choose_devices(eval_config["gpu_count"])

    logger.info(
        "Starting linear-probe evaluation | fed_ckpt=%s | base_ckpt=%s | datasets=%s | devices=(%s, %s)",
        args.federated_checkpoint_path,
        args.base_checkpoint_path,
        ", ".join(DATASET_DISPLAY_NAMES[name] for name in dataset_names),
        federated_device,
        base_device,
    )

    # Both checkpoints are loaded into the same adapter-injected MAE shape.
    federated_model = load_checkpoint_model(args.federated_checkpoint_path, federated_config, federated_device)
    base_model = load_checkpoint_model(args.base_checkpoint_path, base_config, base_device)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {"federated": {}, "base": {}}
    for dataset_name in dataset_names:
        # The two probes see the same dataset order but train independently.
        if federated_device != base_device:
            with ThreadPoolExecutor(max_workers=2) as executor:
                federated_future = executor.submit(
                    evaluate_single_model_on_dataset,
                    federated_model,
                    "Federated",
                    dataset_name,
                    federated_device,
                    eval_config,
                )
                base_future = executor.submit(
                    evaluate_single_model_on_dataset,
                    base_model,
                    "Base",
                    dataset_name,
                    base_device,
                    eval_config,
                )
                results["federated"][dataset_name] = federated_future.result()
                results["base"][dataset_name] = base_future.result()
        else:
            results["federated"][dataset_name] = evaluate_single_model_on_dataset(
                federated_model,
                "Federated",
                dataset_name,
                federated_device,
                eval_config,
            )
            results["base"][dataset_name] = evaluate_single_model_on_dataset(
                base_model,
                "Base",
                dataset_name,
                base_device,
                eval_config,
            )

    print_comparison_table(results, dataset_names)

    output_path = save_results(args.output_dir, dataset_names, results)
    evaluated_dataset_names = get_evaluated_dataset_names(results, dataset_names)
    accuracy_plot = plot_comparison_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="accuracy",
        title="Linear-Probe Accuracy",
        filename="linear_probe_accuracy.png",
    )
    f1_plot = plot_comparison_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="macro_f1",
        title="Linear-Probe Macro F1",
        filename="linear_probe_macro_f1.png",
    )
    eval_loss_plot = plot_comparison_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="eval_loss",
        title="Linear-Probe Eval Loss",
        filename="linear_probe_eval_loss.png",
    )
    accuracy_delta_plot = plot_delta_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="accuracy",
        title="Federated Minus Base Accuracy",
        filename="linear_probe_accuracy_delta.png",
    )
    f1_delta_plot = plot_delta_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="macro_f1",
        title="Federated Minus Base Macro F1",
        filename="linear_probe_macro_f1_delta.png",
    )
    eval_loss_delta_plot = plot_delta_bars(
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        results=results,
        metric_key="eval_loss",
        title="Federated Minus Base Eval Loss",
        filename="linear_probe_eval_loss_delta.png",
    )
    heatmap_plot = plot_accuracy_heatmap(args.output_dir, dataset_names, results)

    logger.info(
        "Evaluation complete | evaluated=%s/%s | json=%s | accuracy_plot=%s | f1_plot=%s | eval_loss_plot=%s | accuracy_delta=%s | f1_delta=%s | eval_loss_delta=%s | heatmap=%s",
        len(evaluated_dataset_names),
        len(dataset_names),
        output_path,
        accuracy_plot,
        f1_plot,
        eval_loss_plot,
        accuracy_delta_plot,
        f1_delta_plot,
        eval_loss_delta_plot,
        heatmap_plot,
    )


if __name__ == "__main__":
    main()
