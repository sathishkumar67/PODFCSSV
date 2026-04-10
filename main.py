"""Run the full current PODFCSSV workflow from one file.

This module is the active experiment entrypoint for the repository. The file is
designed to keep the whole pipeline visible in one place so the current
training, evaluation, and export behavior can be audited step by step.

The workflow is:
1. read the selected ``RUN_MODE``,
2. resolve the device and keep the active math path in ``torch.float32``,
3. prepare every training and held-out dataset needed by the run before
   training begins,
4. build the shared adapter-injected ViT-MAE backbone,
5. build the benchmark-plus-stress stage plan,
6. train through that continual stream in either federated or baseline mode,
7. run one final frozen-feature linear-probe pass on the benchmark datasets,
   and
8. save the final checkpoint and training artifacts, then export probe
   artifacts separately.

This file therefore owns configuration, dataset loading, training loops, server
communication, evaluation, plotting, and final export for the active pipeline.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from transformers import ViTMAEForPreTraining

from src.client import ClientManager
from src.loss import GPADLoss
from src.mae_with_adapter import inject_adapters
from src.server import FederatedModelServer, GlobalPrototypeBank, run_server_round

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)-16s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PODFCSSV_Main")
baseline_logger = logging.getLogger("PODFCSSV_Baseline")

RUN_MODE = "federated"

CONFIG: Dict[str, Any] = {
    "seed": 42,
    "num_clients": 2,
    "local_epochs": 1,
    "batch_size": 512,
    "client_lr": 1e-4,
    "client_weight_decay": 0.05,
    "gpu_count": 0,
    "device": "cpu",
    "dtype": torch.float32,
    "pin_memory": True,
    "dataloader_shuffle": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 8,
    "cudnn_benchmark": True,
    "pretrained_model_name": "facebook/vit-mae-base",
    "embedding_dim": 768,
    "image_size": 224,
    "adapter_bottleneck_dim": 256,
    "merge_threshold": 0.85,
    "server_ema_alpha": 0.1,
    "server_model_ema_alpha": 0.3,
    "max_global_prototypes": 2000,
    "gpad_base_tau": 0.85,
    "gpad_temp_gate": 0.1,
    "gpad_lambda_entropy": 0.2,
    "gpad_soft_assign_temp": 0.1,
    "gpad_epsilon": 1e-8,
    "lambda_proto": 0.1,
    "k_init_prototypes": 20,
    "client_local_update_threshold": 0.85,
    "client_local_ema_alpha": 0.05,
    "novelty_buffer_size": 256,
    "novelty_k": 10,
    "kmeans_max_iters": 100,
    "kmeans_tol": 1e-4,
    "data_root": "./data",
    "save_dir": "multidataset_outputs",
}

MULTI_DATASET_CONFIG: Dict[str, Any] = {
    **CONFIG,
    "num_clients": 2,
    "rounds_per_dataset": 3,
    "linear_eval_batch_size": 512,
    "linear_eval_epochs": 5,
    "linear_eval_lr": 1e-2,
    "linear_eval_weight_decay": 1e-4,
    "save_dir": "multidataset_outputs_2client",
}

BASELINE_CONFIG: Dict[str, Any] = {
    **MULTI_DATASET_CONFIG,
    "num_clients": 1,
    "save_dir": "baseline_outputs",
}

EUROSAT_TRAIN_SPLIT_SAMPLES = 22000
EUROSAT_EVAL_SPLIT_SAMPLES = 5000

BENCHMARK_CLIENT_DATASET_SEQUENCE: Dict[int, List[str]] = {
    0: ["eurosat", "food101", "oxfordiiitpet"],
    1: ["gtsrb", "country211", "fgvcaircraft"],
}

FEDERATED_RETENTION_NOISE_SEQUENCE: Dict[int, List[str]] = {
    0: ["cifar10", "stl10", "flowers102"],
    1: ["svhn", "cifar100", "dtd"],
}

MANUAL_SETUP_DATASETS = {
    "fer2013",
    "pcam",
    "stanfordcars",
}

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "eurosat": "EuroSAT",
    "pcam": "PCAM",
    "fer2013": "FER2013",
    "fgvcaircraft": "FGVC Aircraft",
    "dtd": "DTD",
    "oxfordiiitpet": "Oxford-IIIT Pet",
    "flowers102": "Flowers102",
    "food101": "Food101",
    "gtsrb": "GTSRB",
    "svhn": "SVHN",
    "stanfordcars": "Stanford Cars",
    "country211": "Country211",
    "caltech101": "Caltech101",
    "caltech256": "Caltech256",
    "sun397": "SUN397",
    "cifar100": "CIFAR100",
    "stl10": "STL10",
    "lfwpeople": "LFW People",
    "cifar10": "CIFAR10",
    "fashionmnist": "FashionMNIST",
    "renderedsst2": "Rendered SST2",
    "usps": "USPS",
    "emnistletters": "EMNIST Letters",
    "omniglot": "Omniglot",
}


def set_random_seed(seed: Optional[int]) -> None:
    """Seed every random source used by training and evaluation."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_device_passes_smoke_test(device_index: int) -> tuple[bool, str]:
    """Run a tiny convolution on one CUDA device to confirm kernels really execute."""
    device = torch.device(f"cuda:{device_index}")
    try:
        with torch.inference_mode():
            inputs = torch.zeros((1, 3, 32, 32), device=device, dtype=torch.float32)
            weights = torch.zeros((4, 3, 3, 3), device=device, dtype=torch.float32)
            outputs = F.conv2d(inputs, weights)
            _ = float(outputs.sum().item())
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def get_usable_cuda_device_count() -> int:
    """Count how many CUDA devices pass the minimal execution smoke test."""
    if not torch.cuda.is_available():
        return 0

    usable_device_count = 0
    detected_device_count = torch.cuda.device_count()
    for device_index in range(detected_device_count):
        passes_smoke_test, error_message = cuda_device_passes_smoke_test(device_index)
        if passes_smoke_test:
            usable_device_count += 1
        else:
            logger.warning(
                "Ignoring cuda:%s because a minimal CUDA kernel could not run on it: %s",
                device_index,
                error_message,
            )
    return usable_device_count


def resolve_runtime_config(config: Dict[str, Any]) -> None:
    """Resolve the real execution device for the current run.

    The code does not rely on ``torch.cuda.is_available()`` alone because some
    environments report CUDA but still fail once the first real kernel is
    launched. This helper performs the repository's safety check:
    1. count the visible CUDA devices,
    2. run a tiny convolution on each one,
    3. keep only devices that actually execute that kernel, and
    4. fall back to CPU when none of them pass.
    """
    usable_gpu_count = get_usable_cuda_device_count()
    if usable_gpu_count > 0:
        config["gpu_count"] = usable_gpu_count
        config["device"] = "cuda"
    else:
        config["gpu_count"] = 0
        config["device"] = "cpu"


def convert_to_rgb(image: Any) -> Any:
    """Convert images to RGB before resizing and tensor conversion."""
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert runtime config values into a checkpoint-safe dictionary.

    Torch dtypes are converted to strings so the same config block can be
    stored inside both JSON histories and PyTorch checkpoints.
    """
    serialized: Dict[str, Any] = {}
    for key, value in config.items():
        serialized[key] = str(value) if isinstance(value, torch.dtype) else value
    return serialized


def tensor_num_bytes(tensor: Optional[torch.Tensor]) -> int:
    """Estimate the communication size of one tensor in bytes."""
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def state_dict_num_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Estimate the total communication size of a state-dict payload."""
    return sum(tensor_num_bytes(tensor) for tensor in state_dict.values())


def extract_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU copy of only the trainable model parameters.

    In this project those parameters correspond to the injected adapters, so
    the helper defines the exact weight payload exchanged or checkpointed.
    """
    trainable_state: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_state[name] = parameter.detach().cpu().clone()
    return trainable_state


def prepare_output_dirs(save_dir: str) -> Dict[str, Path]:
    """Create the standard output directory structure for one run.

    Every run writes into stable training and evaluation locations:
    1. ``checkpoints`` for model snapshots.
    2. ``metrics`` for JSON histories.
    3. ``plots`` for publication-oriented visualizations.
    4. ``final_linear_probe/metrics`` for the final probe JSON exports.
    5. ``final_linear_probe/plots`` for the final probe figures.
    """
    root = Path(save_dir)
    checkpoints_dir = root / "checkpoints"
    metrics_dir = root / "metrics"
    plots_dir = root / "plots"
    probe_root = root / "final_linear_probe"
    probe_metrics_dir = probe_root / "metrics"
    probe_plots_dir = probe_root / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    probe_metrics_dir.mkdir(parents=True, exist_ok=True)
    probe_plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "checkpoints": checkpoints_dir,
        "metrics": metrics_dir,
        "plots": plots_dir,
        "probe_root": probe_root,
        "probe_metrics": probe_metrics_dir,
        "probe_plots": probe_plots_dir,
    }


def resolve_worker_count() -> int:
    """Choose the worker count used by the current run.

    The policy is intentionally conservative so the training and evaluation
    dataloaders do not oversubscribe large machines:
    1. read the visible CPU count,
    2. reserve one core for the main process when possible, and
    3. cap the resulting worker count at ``16``.
    """
    cpu_count = os.cpu_count() or 1
    return min(16, max(1, cpu_count - 1))


def build_base_model(config: Dict[str, Any]) -> nn.Module:
    """Build the shared adapter-injected MAE backbone used by all modes.

    The helper always performs the same construction path:
    1. load the selected pretrained ViT-MAE checkpoint,
    2. inject adapters into the upper encoder layers,
    3. move the model to the configured device and dtype, and
    4. log the total and trainable parameter counts.
    """
    model = ViTMAEForPreTraining.from_pretrained(config["pretrained_model_name"])
    model = inject_adapters(model, bottleneck_dim=config["adapter_bottleneck_dim"])
    model = model.to(device=config["device"], dtype=config["dtype"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if config.get("log_model_ready", True):
        logger.info(
            "Model ready | total=%s | trainable=%s (%.2f%%)",
            f"{total_parameters:,}",
            f"{trainable_parameters:,}",
            100.0 * trainable_parameters / total_parameters,
        )
    return model


def build_gpad_loss(config: Dict[str, Any]) -> GPADLoss:
    """Build the GPAD loss object from the active configuration values."""
    return GPADLoss(
        base_tau=config["gpad_base_tau"],
        temp_gate=config["gpad_temp_gate"],
        lambda_entropy=config["gpad_lambda_entropy"],
        soft_assign_temp=config["gpad_soft_assign_temp"],
        epsilon=config["gpad_epsilon"],
    )


def average_client_metric(client_results: List[Dict[str, float]], key: str) -> float:
    """Average one logged client metric across the current round."""
    if not client_results:
        return 0.0
    return float(sum(result.get(key, 0.0) for result in client_results) / len(client_results))


def save_history(
    history: Dict[str, Any],
    metrics_dir: Path,
    filename: str = "training_history.json",
) -> Path:
    """Write one history dictionary to disk as JSON."""
    history_path = metrics_dir / filename
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return history_path


def save_json_payload(
    payload: Dict[str, Any],
    metrics_dir: Path,
    filename: str,
) -> Path:
    """Write one generic JSON payload to the metrics directory."""
    output_path = metrics_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def save_checkpoint(
    checkpoint_dir: Path,
    round_idx: int,
    base_model: nn.Module,
    config: Dict[str, Any],
    proto_bank: Optional[GlobalPrototypeBank] = None,
    is_final: bool = False,
) -> Path:
    """Save the final experiment snapshot in the repository checkpoint format.

    The active workflow writes the final checkpoint as soon as training ends.
    Each saved file contains:
    1. the final round index,
    2. the trainable adapter state,
    3. the serialized runtime config, and
    4. the optional global prototype bank for federated runs.

    Final probe artifacts are intentionally stored outside the checkpoint so the
    model snapshot remains available even if later evaluation fails.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "round": round_idx,
        "model_state_dict": extract_trainable_state_dict(base_model),
        "config": serialize_config(config),
    }
    if proto_bank is not None:
        checkpoint["global_prototypes"] = proto_bank.get_prototypes().detach().cpu()
    filename = "final_model.pt" if is_final else f"round_{round_idx}.pt"
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def plot_training_history(
    history: Dict[str, Any],
    plots_dir: Path,
    prefix: str = "main",
) -> Path:
    """Create the federated training summary figure.

    The figure summarizes four aspects of one run:
    1. Losses.
    2. Global prototype-bank growth.
    3. Routing fractions.
    4. Communication cost per round.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    rounds = history.get("round_ids", [])
    figure_path = plots_dir / f"{prefix}_training_summary.png"
    if not rounds:
        return figure_path

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(rounds, history["avg_total_loss"], label="Total Loss", marker="o")
    axes[0, 0].plot(rounds, history["avg_mae_loss"], label="MAE Loss", marker="o")
    axes[0, 0].plot(rounds, history["avg_gpad_loss"], label="GPAD Loss", marker="o")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(rounds, history["global_prototype_count"], marker="o")
    axes[0, 1].set_title("Global Prototype Bank Size")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Prototype Count")

    axes[1, 0].plot(rounds, history["avg_anchor_fraction"], label="Anchored", marker="o")
    axes[1, 0].plot(rounds, history["avg_local_match_fraction"], label="Local Match", marker="o")
    axes[1, 0].plot(rounds, history["avg_novel_fraction"], label="Novel", marker="o")
    axes[1, 0].set_title("Routing Fractions")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Fraction of Samples")
    axes[1, 0].legend()

    axes[1, 1].plot(rounds, history["upload_bytes"], label="Upload", marker="o")
    axes[1, 1].plot(rounds, history["download_bytes"], label="Download", marker="o")
    axes[1, 1].plot(rounds, history["total_communication_bytes"], label="Total", marker="o")
    axes[1, 1].set_title("Communication per Round")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Bytes")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def print_round_summary(
    round_idx: int,
    num_rounds: int,
    client_results: List[Dict[str, float]],
    proto_bank: GlobalPrototypeBank,
    round_time: float,
    upload_bytes: int,
    download_bytes: int,
) -> None:
    """Log the compact round summary shown after each communication round."""
    average_loss = average_client_metric(client_results, "loss")
    average_anchor_fraction = average_client_metric(client_results, "anchored_fraction")
    average_novel_fraction = average_client_metric(client_results, "novel_fraction")
    prototype_count = int(proto_bank.get_prototypes().size(0))
    logger.info(
        "Round %s/%s complete | avg_loss=%.6f | anchored=%.4f | novel=%.4f | global_prototypes=%s | upload=%s bytes | download=%s bytes | time=%.2fs",
        round_idx,
        num_rounds,
        average_loss,
        average_anchor_fraction,
        average_novel_fraction,
        prototype_count,
        upload_bytes,
        download_bytes,
        round_time,
    )


def initialize_history() -> Dict[str, Any]:
    """Create the federated history structure used across the whole run.

    The history dictionary stores only training-time information:
    1. round-wise optimization metrics,
    2. communication statistics, and
    3. per-round client summaries.

    Final linear-probe artifacts are exported separately so finished training
    state is preserved even if evaluation later fails.
    """
    return {
        "round_ids": [],
        "round_times": [],
        "avg_total_loss": [],
        "avg_mae_loss": [],
        "avg_gpad_loss": [],
        "avg_anchor_fraction": [],
        "avg_local_match_fraction": [],
        "avg_novel_fraction": [],
        "global_prototype_count": [],
        "client_prototype_counts": [],
        "upload_bytes": [],
        "download_bytes": [],
        "total_communication_bytes": [],
        "task_classes": [],
        "stage_kinds": [],
        "client_results": [],
    }


def normalize_client_dataset_sequence(
    dataset_sequence: Dict[Any, Sequence[str]],
) -> Dict[int, List[str]]:
    """Convert client keys to integers and dataset sequences to plain lists."""
    return {
        int(client_index): list(dataset_names)
        for client_index, dataset_names in dataset_sequence.items()
    }


def validate_client_dataset_sequence(
    dataset_sequence: Dict[Any, Sequence[str]],
    expected_num_clients: int,
    sequence_name: str,
) -> Dict[int, List[str]]:
    """Validate one client-to-datasets mapping before it is used.

    The validation checks that:
    1. The mapping matches the configured client count.
    2. Every client has the same number of stages.
    3. No dataset name is duplicated across clients.
    4. Every dataset has a display-name entry.
    5. The default publishable schedule avoids manual-setup datasets.
    """
    normalized_sequence = normalize_client_dataset_sequence(dataset_sequence)
    if len(normalized_sequence) != expected_num_clients:
        raise ValueError(
            f"{sequence_name} does not match the configured client count. "
            f"Expected {expected_num_clients} clients but found {len(normalized_sequence)}."
        )

    stage_counts = {len(dataset_names) for dataset_names in normalized_sequence.values()}
    if len(stage_counts) != 1:
        raise ValueError(f"Every client must receive the same number of datasets in {sequence_name}.")

    all_dataset_names = [
        dataset_name
        for client_index in sorted(normalized_sequence)
        for dataset_name in normalized_sequence[client_index]
    ]
    duplicate_names = sorted(
        dataset_name
        for dataset_name in set(all_dataset_names)
        if all_dataset_names.count(dataset_name) > 1
    )
    if duplicate_names:
        raise ValueError(f"Duplicate dataset entries found in {sequence_name}: {duplicate_names}")

    missing_display_names = sorted(
        dataset_name
        for dataset_name in all_dataset_names
        if dataset_name not in DATASET_DISPLAY_NAMES
    )
    if missing_display_names:
        raise ValueError(
            f"Missing display-name entries for datasets in {sequence_name}: {missing_display_names}"
        )

    manual_setup_datasets = sorted(
        dataset_name
        for dataset_name in all_dataset_names
        if dataset_name in MANUAL_SETUP_DATASETS
    )
    if manual_setup_datasets:
        raise ValueError(
            f"{sequence_name} includes datasets with manual-download caveats: {manual_setup_datasets}"
        )
    return normalized_sequence


def build_dataset_order_by_stage(
    dataset_sequence: Optional[Dict[Any, Sequence[str]]] = None,
) -> List[str]:
    """Flatten one client schedule into the stage order used by evaluation."""
    sequence = dataset_sequence or BENCHMARK_CLIENT_DATASET_SEQUENCE
    normalized_sequence = normalize_client_dataset_sequence(sequence)
    stage_count = len(next(iter(normalized_sequence.values())))

    dataset_order: List[str] = []
    for stage_index in range(stage_count):
        for client_index in sorted(normalized_sequence):
            dataset_order.append(normalized_sequence[client_index][stage_index])
    return dataset_order


def build_federated_stage_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the federated stage plan with benchmark and stress stages interleaved.

    The resulting plan alternates between:
    1. benchmark stages,
    2. stress stages that exist only to create extra forgetting pressure, and
    3. an optional final stress stage after the last benchmark stage when the
       noise schedule is as long as the benchmark schedule.
    """
    benchmark_sequence = validate_client_dataset_sequence(
        BENCHMARK_CLIENT_DATASET_SEQUENCE,
        expected_num_clients=config["num_clients"],
        sequence_name="BENCHMARK_CLIENT_DATASET_SEQUENCE",
    )
    noise_sequence = validate_client_dataset_sequence(
        FEDERATED_RETENTION_NOISE_SEQUENCE,
        expected_num_clients=config["num_clients"],
        sequence_name="FEDERATED_RETENTION_NOISE_SEQUENCE",
    )

    benchmark_stage_count = len(next(iter(benchmark_sequence.values())))
    noise_stage_count = len(next(iter(noise_sequence.values())))
    allowed_noise_stage_counts = {max(benchmark_stage_count - 1, 0), benchmark_stage_count}
    if noise_stage_count not in allowed_noise_stage_counts:
        raise ValueError(
            "The retention-noise schedule must have either exactly one fewer "
            "stage than the benchmark schedule or the same number of stages."
        )

    stage_plan: List[Dict[str, Any]] = []
    for stage_index in range(benchmark_stage_count):
        benchmark_datasets = {
            client_index: benchmark_sequence[client_index][stage_index]
            for client_index in sorted(benchmark_sequence)
        }
        stage_plan.append(
            {
                "stage_kind": "benchmark",
                "datasets": benchmark_datasets,
            }
        )
        if stage_index < noise_stage_count:
            noise_datasets = {
                client_index: noise_sequence[client_index][stage_index]
                for client_index in sorted(noise_sequence)
            }
            stage_plan.append(
                {
                    "stage_kind": "retention_noise",
                    "datasets": noise_datasets,
                }
            )
    return stage_plan


def build_baseline_stage_plan() -> List[Dict[str, Any]]:
    """Flatten the benchmark and stress schedule into one sequential baseline order.

    The baseline sees the same dataset stream as the federated run, but because
    it owns only one model, each client-stage dataset becomes its own sequential
    training stage:
    1. Reuse the validated federated stage plan.
    2. Keep benchmark and stress ordering identical to the federated stream.
    3. Flatten each multi-client stage into a one-model sequential list.
    """
    federated_stage_plan = build_federated_stage_plan(dict(MULTI_DATASET_CONFIG))
    baseline_stage_plan: List[Dict[str, Any]] = []

    for stage_spec in federated_stage_plan:
        for client_index in sorted(stage_spec["datasets"]):
            baseline_stage_plan.append(
                {
                    "stage_kind": stage_spec["stage_kind"],
                    "dataset_name": stage_spec["datasets"][client_index],
                    "source_client": client_index,
                }
            )

    return baseline_stage_plan


def stable_int_from_parts(seed: int, *parts: str) -> int:
    """Create one reproducible integer seed from a base seed plus string tags."""
    joined = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_deterministic_split_indices(
    dataset_size: int,
    seed: int,
    train_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Create a deterministic train/test split for datasets without official splits."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(dataset_size * train_fraction)
    return indices[:train_size], indices[train_size:]


def fit_dataset_to_sample_budget(
    dataset: torch.utils.data.Dataset,
    dataset_name: str,
    train: bool,
    seed: int,
    target_samples: Optional[int],
    min_samples: Optional[int],
) -> torch.utils.data.Dataset:
    """Optionally reshape one split to a deterministic sample count.

    The current training path usually keeps full splits, but the helper remains
    available because the evaluation code still supports optional caps. Its
    logic is:
    1. Reject a split that falls below the requested minimum.
    2. Return the split unchanged when no target size is requested.
    3. Deterministically subsample when the split is larger than the target.
    4. Deterministically repeat samples when the split is smaller than the target.
    """
    dataset_size = len(dataset)
    split_name = "train" if train else "test"

    if min_samples is not None and dataset_size < min_samples:
        raise ValueError(
            f"{DATASET_DISPLAY_NAMES[dataset_name]} {split_name} split has only {dataset_size} "
            f"samples, which is below the required minimum of {min_samples}."
        )

    if target_samples is None or dataset_size == target_samples:
        logger.info(
            "Prepared %s %s split | original=%s | used=%s",
            DATASET_DISPLAY_NAMES[dataset_name],
            split_name,
            dataset_size,
            dataset_size,
        )
        return dataset

    sample_seed = stable_int_from_parts(seed, dataset_name, split_name, "fit")
    generator = torch.Generator().manual_seed(sample_seed)

    if dataset_size > target_samples:
        selected_indices = torch.randperm(dataset_size, generator=generator)[:target_samples].tolist()
        logger.info(
            "Prepared %s %s split | original=%s | used=%s | mode=subsampled",
            DATASET_DISPLAY_NAMES[dataset_name],
            split_name,
            dataset_size,
            target_samples,
        )
        return Subset(dataset, selected_indices)

    if dataset_size < target_samples:
        selected_indices = torch.randint(
            low=0,
            high=dataset_size,
            size=(target_samples,),
            generator=generator,
        ).tolist()
        logger.info(
            "Prepared %s %s split | original=%s | used=%s | mode=upsampled",
            DATASET_DISPLAY_NAMES[dataset_name],
            split_name,
            dataset_size,
            target_samples,
        )
        return Subset(dataset, selected_indices)

    return dataset


def build_dataset_transform(image_size: int) -> transforms.Compose:
    """Build the shared image transform used by every dataset loader.

    The transform deliberately stays simple:
    1. Convert to RGB.
    2. Resize to the ViT-MAE input size.
    3. Convert to a tensor.

    No ImageNet normalization is used in the current pipeline.
    """
    return transforms.Compose(
        [
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_split_subset(
    dataset: torch.utils.data.Dataset,
    dataset_name: str,
    train: bool,
    seed: int,
    train_fraction: float = 0.8,
) -> torch.utils.data.Dataset:
    """Create a deterministic split for datasets that ship as one combined folder."""
    split_seed = stable_int_from_parts(seed, dataset_name, "split")
    train_indices, test_indices = build_deterministic_split_indices(
        dataset_size=len(dataset),
        seed=split_seed,
        train_fraction=train_fraction,
    )
    return Subset(dataset, train_indices if train else test_indices)


def build_fixed_count_split_subset(
    dataset: torch.utils.data.Dataset,
    dataset_name: str,
    train: bool,
    seed: int,
    train_samples: int,
    eval_samples: int,
) -> torch.utils.data.Dataset:
    """Create a deterministic fixed-size train/eval split for one combined dataset.

    The current `EuroSAT` protocol uses the leading portion of the dataset for
    training and the trailing portion for evaluation:
    1. take the first ``train_samples`` examples for training,
    2. take the last ``eval_samples`` examples for evaluation, and
    3. leave any middle portion unused when the full dataset is larger than the
       requested split sizes.
    """
    required_samples = train_samples + eval_samples
    dataset_size = len(dataset)
    if dataset_size < required_samples:
        raise ValueError(
            f"{DATASET_DISPLAY_NAMES[dataset_name]} has only {dataset_size} samples, "
            f"but the fixed split needs {required_samples} samples."
        )

    train_indices = list(range(train_samples))
    eval_start_index = dataset_size - eval_samples
    eval_indices = list(range(eval_start_index, dataset_size))
    return Subset(dataset, train_indices if train else eval_indices)


def load_named_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    train: bool,
    seed: int,
    max_samples: Optional[int] = None,
    min_samples: Optional[int] = None,
    merge_all_training_splits: bool = False,
) -> torch.utils.data.Dataset:
    """Load one dataset using the exact split policy of the active pipeline.

    This helper keeps the training and evaluation paths aligned by centralizing
    all dataset rules in one place. The current rules are:
    1. benchmark datasets use their normal train-side split in full,
    2. ``EuroSAT`` uses the repository's fixed head/tail ``22000 / 5000``
       split,
    3. stress datasets can merge all official splits into one self-supervised
       pool when ``merge_all_training_splits`` is enabled, and
    4. deterministic sample shaping is applied only when a caller explicitly
       asks for a cap or minimum.
    """
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if train and merge_all_training_splits:
        # Stress datasets are treated as broad self-supervised pools, so the
        # training path intentionally consumes every official split they expose.
        if dataset_name == "cifar10":
            dataset = ConcatDataset(
                [
                    datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform),
                    datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform),
                ]
            )
        elif dataset_name == "stl10":
            dataset = ConcatDataset(
                [
                    datasets.STL10(root=str(root), split="train", download=True, transform=transform),
                    datasets.STL10(root=str(root), split="test", download=True, transform=transform),
                    datasets.STL10(root=str(root), split="unlabeled", download=True, transform=transform),
                ]
            )
        elif dataset_name == "flowers102":
            dataset = ConcatDataset(
                [
                    datasets.Flowers102(root=str(root), split="train", download=True, transform=transform),
                    datasets.Flowers102(root=str(root), split="val", download=True, transform=transform),
                    datasets.Flowers102(root=str(root), split="test", download=True, transform=transform),
                ]
            )
        elif dataset_name == "svhn":
            dataset = ConcatDataset(
                [
                    datasets.SVHN(root=str(root), split="train", download=True, transform=transform),
                    datasets.SVHN(root=str(root), split="test", download=True, transform=transform),
                    datasets.SVHN(root=str(root), split="extra", download=True, transform=transform),
                ]
            )
        elif dataset_name == "cifar100":
            dataset = ConcatDataset(
                [
                    datasets.CIFAR100(root=str(root), train=True, download=True, transform=transform),
                    datasets.CIFAR100(root=str(root), train=False, download=True, transform=transform),
                ]
            )
        elif dataset_name == "dtd":
            dataset = ConcatDataset(
                [
                    datasets.DTD(root=str(root), split="train", download=True, transform=transform),
                    datasets.DTD(root=str(root), split="val", download=True, transform=transform),
                    datasets.DTD(root=str(root), split="test", download=True, transform=transform),
                ]
            )
        else:
            raise ValueError(
                f"merge_all_training_splits=True is not supported for dataset '{dataset_name}'."
            )
    elif dataset_name == "eurosat":
        dataset = build_fixed_count_split_subset(
            datasets.EuroSAT(root=str(root), download=True, transform=transform),
            dataset_name=dataset_name,
            train=train,
            seed=seed,
            train_samples=EUROSAT_TRAIN_SPLIT_SAMPLES,
            eval_samples=EUROSAT_EVAL_SPLIT_SAMPLES,
        )
    elif dataset_name == "pcam":
        dataset = datasets.PCAM(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "fer2013":
        dataset = datasets.FER2013(
            root=str(root),
            split="train" if train else "test",
            transform=transform,
        )
    elif dataset_name == "fgvcaircraft":
        dataset = datasets.FGVCAircraft(
            root=str(root),
            split="trainval" if train else "test",
            annotation_level="variant",
            download=True,
            transform=transform,
        )
    elif dataset_name == "dtd":
        dataset = datasets.DTD(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "oxfordiiitpet":
        dataset = datasets.OxfordIIITPet(
            root=str(root),
            split="trainval" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "flowers102":
        dataset = datasets.Flowers102(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "food101":
        dataset = datasets.Food101(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "gtsrb":
        dataset = datasets.GTSRB(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "svhn":
        dataset = datasets.SVHN(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "stanfordcars":
        dataset = datasets.StanfordCars(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "country211":
        dataset = datasets.Country211(
            root=str(root),
            split="train" if train else "valid",
            download=True,
            transform=transform,
        )
    elif dataset_name == "caltech101":
        dataset = build_split_subset(
            datasets.Caltech101(
                root=str(root),
                target_type="category",
                download=True,
                transform=transform,
            ),
            dataset_name=dataset_name,
            train=train,
            seed=seed,
        )
    elif dataset_name == "caltech256":
        dataset = build_split_subset(
            datasets.Caltech256(
                root=str(root),
                download=True,
                transform=transform,
            ),
            dataset_name=dataset_name,
            train=train,
            seed=seed,
        )
    elif dataset_name == "sun397":
        dataset = build_split_subset(
            datasets.SUN397(
                root=str(root),
                download=True,
                transform=transform,
            ),
            dataset_name=dataset_name,
            train=train,
            seed=seed,
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "stl10":
        dataset = datasets.STL10(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "lfwpeople":
        dataset = datasets.LFWPeople(
            root=str(root),
            split="train" if train else "test",
            image_set="funneled",
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "renderedsst2":
        dataset = datasets.RenderedSST2(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "usps":
        dataset = datasets.USPS(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "emnistletters":
        dataset = datasets.EMNIST(
            root=str(root),
            split="letters",
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset_name == "omniglot":
        dataset = datasets.Omniglot(
            root=str(root),
            background=train,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return fit_dataset_to_sample_budget(
        dataset=dataset,
        dataset_name=dataset_name,
        train=train,
        seed=seed,
        target_samples=max_samples,
        min_samples=min_samples,
    )


def load_named_evaluation_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    seed: int,
    max_samples: Optional[int] = None,
    min_samples: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """Load the held-out split used by the final benchmark linear probe.

    The reporting side of the pipeline is centralized here so every final probe
    number is based on one consistent split policy:
    1. reuse the repository's fixed `EuroSAT` held-out split,
    2. use the official held-out split for datasets that already define one,
    3. use `Country211` validation rather than test in the current benchmark,
       and
    4. keep any exceptional merge logic for held-out reporting in one place.
    """
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if dataset_name == "flowers102":
        dataset = ConcatDataset(
            [
                datasets.Flowers102(
                    root=str(root),
                    split="val",
                    download=True,
                    transform=transform,
                ),
                datasets.Flowers102(
                    root=str(root),
                    split="test",
                    download=True,
                    transform=transform,
                ),
            ]
        )
        return fit_dataset_to_sample_budget(
            dataset=dataset,
            dataset_name=dataset_name,
            train=False,
            seed=seed,
            target_samples=max_samples,
            min_samples=min_samples,
        )

    if dataset_name == "dtd":
        dataset = ConcatDataset(
            [
                datasets.DTD(
                    root=str(root),
                    split="val",
                    download=True,
                    transform=transform,
                ),
                datasets.DTD(
                    root=str(root),
                    split="test",
                    download=True,
                    transform=transform,
                ),
            ]
        )
        return fit_dataset_to_sample_budget(
            dataset=dataset,
            dataset_name=dataset_name,
            train=False,
            seed=seed,
            target_samples=max_samples,
            min_samples=min_samples,
        )

    return load_named_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        train=False,
        seed=seed,
        max_samples=max_samples,
        min_samples=min_samples,
    )


def prepare_dataset_cache(
    config: Dict[str, Any],
    training_stage_plan: Sequence[Dict[str, Any]],
    probe_dataset_order: Sequence[str],
    active_logger: logging.Logger,
) -> Dict[str, torch.utils.data.Dataset]:
    """Prepare every dataset variant needed by the run before training begins."""
    dataset_cache: Dict[str, torch.utils.data.Dataset] = {}
    seen_specs: set[tuple[str, bool]] = set()
    ordered_specs: List[tuple[str, bool]] = []

    for stage_spec in training_stage_plan:
        merge_all_training_splits = stage_spec["stage_kind"] == "retention_noise"
        if "dataset_name" in stage_spec:
            stage_dataset_names = [stage_spec["dataset_name"]]
        else:
            stage_dataset_names = [
                stage_spec["datasets"][client_index]
                for client_index in sorted(stage_spec["datasets"])
            ]

        for dataset_name in stage_dataset_names:
            spec = (dataset_name, merge_all_training_splits)
            if spec not in seen_specs:
                seen_specs.add(spec)
                ordered_specs.append(spec)

    active_logger.info(
        "Preparing all datasets before training starts | train_variants=%s | probe_eval_variants=%s",
        len(ordered_specs),
        len(probe_dataset_order),
    )

    for dataset_name, merge_all_training_splits in ordered_specs:
        split_label = "merged" if merge_all_training_splits else "standard"
        cache_key = f"train::{dataset_name}::{split_label}"
        dataset_cache[cache_key] = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
            max_samples=None,
            min_samples=None,
            merge_all_training_splits=merge_all_training_splits,
        )
        active_logger.info(
            "Prepared training dataset | dataset=%s | split=%s | samples=%s",
            DATASET_DISPLAY_NAMES[dataset_name],
            split_label,
            len(dataset_cache[cache_key]),
        )

    for dataset_name in probe_dataset_order:
        train_cache_key = f"train::{dataset_name}::standard"
        if train_cache_key not in dataset_cache:
            dataset_cache[train_cache_key] = load_named_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                train=True,
                seed=config["seed"],
                max_samples=None,
                min_samples=None,
            )

        eval_cache_key = f"eval::{dataset_name}"
        dataset_cache[eval_cache_key] = load_named_evaluation_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            seed=config["seed"],
            max_samples=None,
            min_samples=None,
        )
        active_logger.info(
            "Prepared evaluation dataset | dataset=%s | samples=%s",
            DATASET_DISPLAY_NAMES[dataset_name],
            len(dataset_cache[eval_cache_key]),
        )

    active_logger.info(
        "All required datasets are ready | cached_entries=%s",
        len(dataset_cache),
    )
    return dataset_cache


def create_standard_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    """Create one dataloader using the active repository loading policy.

    The helper exists so training and evaluation share the same low-level
    dataloader behavior:
    1. respect the caller's dataset, batch size, and shuffle choice,
    2. use the resolved worker cap,
    3. enable pinned memory only when the current runtime makes it useful,
    4. keep worker processes alive between batches when multiprocessing is
       enabled, and
    5. apply the configured prefetch depth.
    """
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def shutdown_dataloader_workers(dataloader: Optional[DataLoader]) -> None:
    """Shut down one dataloader's worker pool if it is still alive.

    The current pipeline uses persistent workers for long training loaders, so
    cleanup must be explicit when a loader is no longer needed. This helper:
    1. inspects the private iterator cache used by persistent workers,
    2. requests worker shutdown when the iterator exposes that hook, and
    3. clears the cached iterator reference so file descriptors can be released.
    """
    if dataloader is None:
        return

    iterator = getattr(dataloader, "_iterator", None)
    if iterator is None:
        return

    shutdown_fn = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_fn):
        try:
            shutdown_fn()
        except Exception:
            pass

    try:
        dataloader._iterator = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _iter_mask_ratio_holders(model: nn.Module) -> List[Any]:
    """Collect every config object that can carry the MAE mask ratio."""
    holders: List[Any] = []
    candidate_holders = [
        getattr(model, "config", None),
        getattr(getattr(model, "vit", None), "config", None),
        getattr(getattr(getattr(model, "vit", None), "embeddings", None), "config", None),
    ]
    for holder in candidate_holders:
        if holder is not None and hasattr(holder, "mask_ratio") and holder not in holders:
            holders.append(holder)
    return holders


def forward_encoder_without_mask(
    model: nn.Module,
    images: torch.Tensor,
    output_hidden_states: bool = False,
) -> Any:
    """Run the MAE encoder on full images by temporarily disabling masking.

    The final linear-probe path uses the encoder alone and sets the MAE mask
    ratio to zero for the duration of the forward pass so the representation is
    measured on the complete image rather than on a masked view.
    """
    if not hasattr(model, "vit"):
        raise AttributeError("Expected a ViT-MAE model with a 'vit' encoder module.")

    mask_ratio_holders = _iter_mask_ratio_holders(model)
    original_mask_ratios = [float(holder.mask_ratio) for holder in mask_ratio_holders]
    for holder in mask_ratio_holders:
        holder.mask_ratio = 0.0

    try:
        return model.vit(
            pixel_values=images,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    finally:
        for holder, original_mask_ratio in zip(mask_ratio_holders, original_mask_ratios):
            holder.mask_ratio = original_mask_ratio


def pool_model_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pool the last encoder hidden state the same way training code does.

    The evaluation path intentionally mirrors the training path by dropping the
    CLS token when present and averaging only the patch tokens.
    """
    if hidden_states.size(1) > 1:
        hidden_states = hidden_states[:, 1:, :]
    return hidden_states.mean(dim=1)


@torch.inference_mode()
def collect_labeled_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized embeddings and labels from one labeled dataset split.

    This is the frozen-feature stage used during the final probe pass:
    1. Run the current MAE encoder on full images.
    2. Pool the last hidden state into one embedding per image.
    3. L2-normalize the embedding.
    4. Return features and labels on CPU for probing.
    """
    model.eval()
    feature_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device=device, dtype=dtype)
        encoder_outputs = forward_encoder_without_mask(
            model=model,
            images=images,
            output_hidden_states=True,
        )
        embeddings = pool_model_hidden_states(encoder_outputs.hidden_states[-1])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        feature_batches.append(embeddings.cpu())
        label_batches.append(torch.as_tensor(labels, dtype=torch.long).cpu())

    if not feature_batches:
        return (
            torch.empty(0, 0, dtype=dtype),
            torch.empty(0, dtype=torch.long),
        )

    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


def remap_labels_to_contiguous(
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Map arbitrary dataset labels onto ``0..num_classes-1`` for probing."""
    unique_labels = torch.unique(torch.cat([train_labels, test_labels], dim=0)).tolist()
    label_mapping = {
        original_label: new_index for new_index, original_label in enumerate(unique_labels)
    }

    remapped_train = train_labels.clone()
    remapped_test = test_labels.clone()
    for original_label, new_index in label_mapping.items():
        remapped_train[train_labels == original_label] = new_index
        remapped_test[test_labels == original_label] = new_index

    return remapped_train, remapped_test, len(unique_labels)


def run_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    config: Dict[str, Any],
) -> float:
    """Train one dataset-specific linear classifier on frozen encoder features.

    This is the core evaluation step used by the final comparison pass:
    1. remap labels onto a contiguous class range,
    2. build one linear classification head,
    3. train that head on top of frozen encoder features,
    4. evaluate the head on the held-out split, and
    5. return the held-out accuracy used by the final summary.

    The encoder itself stays frozen throughout this function.
    """
    if train_features.numel() == 0 or test_features.numel() == 0:
        return 0.0

    train_labels, test_labels, num_classes = remap_labels_to_contiguous(
        train_labels=train_labels,
        test_labels=test_labels,
    )

    probe_device = config["device"]
    probe = nn.Linear(train_features.size(1), num_classes).to(probe_device)
    optimizer = optim.AdamW(
        probe.parameters(),
        lr=config["linear_eval_lr"],
        weight_decay=config["linear_eval_weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader_kwargs: Dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": config["linear_eval_batch_size"],
        "shuffle": True,
        "num_workers": config["linear_eval_num_workers"],
        "pin_memory": config["pin_memory"],
        "drop_last": False,
    }
    if config["linear_eval_num_workers"] > 0:
        train_loader_kwargs["persistent_workers"] = False
        train_loader_kwargs["prefetch_factor"] = config["dataloader_prefetch_factor"]
    train_loader = DataLoader(**train_loader_kwargs)

    try:
        for _ in range(config["linear_eval_epochs"]):
            probe.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(probe_device)
                batch_labels = batch_labels.to(probe_device)

                logits = probe(batch_features)
                loss = loss_fn(logits, batch_labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probe.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for start_index in range(0, test_features.size(0), config["linear_eval_batch_size"]):
                end_index = start_index + config["linear_eval_batch_size"]
                batch_features = test_features[start_index:end_index].to(probe_device)
                batch_labels = test_labels[start_index:end_index].to(probe_device)
                predictions = probe(batch_features).argmax(dim=1)
                correct += int((predictions == batch_labels).sum().item())
                total += int(batch_labels.size(0))

        return correct / total if total > 0 else 0.0
    finally:
        shutdown_dataloader_workers(train_loader)
        del train_loader


def evaluate_datasets(
    model: nn.Module,
    dataset_names: Sequence[str],
    config: Dict[str, Any],
    dataset_cache: Optional[Dict[str, torch.utils.data.Dataset]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the final frozen-feature linear probe on the requested benchmark datasets.

    The final comparison path is shared by baseline and federated runs and
    reuses the prepared dataset cache whenever it is available:
    1. reuse the benchmark train-side split for probe fitting,
    2. reuse the held-out split used for reporting,
    3. extract full-image encoder features with masking disabled,
    4. train a linear classifier on those frozen features, and
    5. record the held-out accuracy and sample counts for the final summary.
    """
    results: Dict[str, Dict[str, float]] = {}
    for dataset_name in dataset_names:
        train_cache_key = f"train::{dataset_name}::standard"
        eval_cache_key = f"eval::{dataset_name}"
        train_dataset = (
            dataset_cache[train_cache_key]
            if dataset_cache and train_cache_key in dataset_cache
            else load_named_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                train=True,
                seed=config["seed"],
            )
        )
        test_dataset = (
            dataset_cache[eval_cache_key]
            if dataset_cache and eval_cache_key in dataset_cache
            else load_named_evaluation_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                seed=config["seed"],
            )
        )

        train_loader = create_standard_dataloader(
            dataset=train_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            persistent_workers=False,
            prefetch_factor=config["dataloader_prefetch_factor"],
        )
        test_loader = create_standard_dataloader(
            dataset=test_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            persistent_workers=False,
            prefetch_factor=config["dataloader_prefetch_factor"],
        )
        try:
            train_features, train_labels = collect_labeled_embeddings(
                model=model,
                dataloader=train_loader,
                device=config["device"],
                dtype=config["dtype"],
            )
            test_features, test_labels = collect_labeled_embeddings(
                model=model,
                dataloader=test_loader,
                device=config["device"],
                dtype=config["dtype"],
            )

            accuracy = run_linear_probe(
                train_features=train_features,
                train_labels=train_labels,
                test_features=test_features,
                test_labels=test_labels,
                config=config,
            )
            results[dataset_name] = {
                "accuracy": float(accuracy),
                "num_train_samples": float(len(train_dataset)),
                "num_eval_samples": float(len(test_dataset)),
            }
        finally:
            shutdown_dataloader_workers(train_loader)
            shutdown_dataloader_workers(test_loader)
            del train_loader
            del test_loader

    return results


def summarize_final_linear_probe(
    evaluation_results: Dict[str, Dict[str, float]],
    dataset_order: Sequence[str],
) -> Dict[str, Any]:
    """Turn the final probe outputs into the saved comparison summary block.

    The final summary keeps one consistent structure for both baseline and
    federated runs:
    1. preserve the requested dataset order,
    2. store per-dataset accuracy and sample counts, and
    3. compute the overall mean accuracy across the reported benchmark datasets.
    """
    ordered_dataset_names = [
        dataset_name for dataset_name in dataset_order if dataset_name in evaluation_results
    ]
    dataset_results = {
        dataset_name: {
            "accuracy": float(evaluation_results[dataset_name]["accuracy"]),
            "num_train_samples": int(evaluation_results[dataset_name]["num_train_samples"]),
            "num_eval_samples": int(evaluation_results[dataset_name]["num_eval_samples"]),
        }
        for dataset_name in ordered_dataset_names
    }
    average_accuracy = float(
        sum(result["accuracy"] for result in dataset_results.values())
        / max(len(dataset_results), 1)
    )
    return {
        "dataset_order": ordered_dataset_names,
        "dataset_results": dataset_results,
        "average_accuracy": average_accuracy,
    }


def plot_final_linear_probe_results(
    final_linear_probe_summary: Dict[str, Any],
    plots_dir: Path,
    prefix: str,
) -> Path:
    """Plot the final per-dataset linear-probe accuracies of one finished run."""
    output_path = plots_dir / f"{prefix}_final_linear_probe_accuracy.png"
    dataset_order = final_linear_probe_summary.get("dataset_order", [])
    dataset_results = final_linear_probe_summary.get("dataset_results", {})
    if not dataset_order or not dataset_results:
        return output_path

    accuracies = [float(dataset_results[dataset_name]["accuracy"]) for dataset_name in dataset_order]
    figure, axis = plt.subplots(figsize=(max(10, len(dataset_order) * 1.2), 5.5))
    x_positions = np.arange(len(dataset_order))
    axis.bar(x_positions, accuracies, color="tab:blue")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [DATASET_DISPLAY_NAMES[name] for name in dataset_order],
        rotation=45,
        ha="right",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Final Linear-Probe Accuracy by Dataset")
    axis.axhline(
        final_linear_probe_summary["average_accuracy"],
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Average = {final_linear_probe_summary['average_accuracy']:.4f}",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_final_linear_probe_outputs(
    final_linear_probe_summary: Dict[str, Any],
    metrics_dir: Path,
    plots_dir: Path,
    prefix: str,
) -> None:
    """Write the final probe artifacts into the dedicated probe output folder."""
    save_json_payload(
        final_linear_probe_summary,
        metrics_dir,
        filename=f"{prefix}_final_linear_probe.json",
    )
    plot_final_linear_probe_results(
        final_linear_probe_summary=final_linear_probe_summary,
        plots_dir=plots_dir,
        prefix=prefix,
    )


def log_final_linear_probe_summary(
    active_logger: logging.Logger,
    run_label: str,
    final_linear_probe_summary: Dict[str, Any],
) -> None:
    """Write one readable per-dataset final probe summary to the active logger."""
    for dataset_name in final_linear_probe_summary.get("dataset_order", []):
        dataset_result = final_linear_probe_summary["dataset_results"][dataset_name]
        active_logger.info(
            "%s final probe | dataset=%s | train=%s | eval=%s | acc=%.4f",
            run_label,
            DATASET_DISPLAY_NAMES[dataset_name],
            dataset_result["num_train_samples"],
            dataset_result["num_eval_samples"],
            dataset_result["accuracy"],
        )


def initialize_baseline_history() -> Dict[str, Any]:
    """Create the history structure written by the baseline mode.

    The baseline records two groups of information:
    1. Round-wise reconstruction metrics.
    2. Per-stage dataset summaries.

    The final linear probe is exported separately from the training history.
    """
    return {
        "round_ids": [],
        "dataset_names": [],
        "dataset_rounds": [],
        "round_times": [],
        "avg_total_loss": [],
        "avg_mae_loss": [],
        "stage_summaries": [],
    }


def train_reconstruction_round(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    local_epochs: int,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Train one baseline model for one round using reconstruction loss only.

    The baseline path is intentionally simple:
    1. Iterate over the dataloader for the configured local-epoch count.
    2. Run MAE reconstruction.
    3. Optimize only the trainable adapter parameters.
    4. Return the aggregated loss and throughput counters for logging.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    for _ in range(local_epochs):
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device=device, dtype=dtype)
            outputs = model(inputs)
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("The MAE model did not return a reconstruction loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_batches += 1
            total_samples += int(inputs.size(0))

    average_loss = total_loss / max(total_batches, 1)
    return {
        "loss": average_loss,
        "mae_loss": average_loss,
        "num_batches": total_batches,
        "num_samples": total_samples,
    }


def create_baseline_dataloader(
    dataset: torch.utils.data.Dataset,
    config: Dict[str, Any],
) -> DataLoader:
    """Create the baseline training dataloader from the active runtime config.

    The baseline uses the same worker, prefetch, shuffle, and pinned-memory
    policy as the rest of the current pipeline so throughput differences do not
    come from a separate dataloader implementation.
    """
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config["batch_size"],
        "shuffle": config["dataloader_shuffle"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"],
        "drop_last": False,
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = config["dataloader_persistent_workers"]
        loader_kwargs["prefetch_factor"] = config["dataloader_prefetch_factor"]
    return DataLoader(**loader_kwargs)


def plot_baseline_training_history(history: Dict[str, Any], plots_dir: Path) -> Path:
    """Plot the baseline reconstruction-loss and round-time curves."""
    figure_path = plots_dir / "baseline_training_summary.png"
    rounds = history.get("round_ids", [])
    if not rounds:
        return figure_path

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(rounds, history["avg_total_loss"], marker="o")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")

    axes[1].plot(rounds, history["round_times"], marker="o")
    axes[1].set_title("Round Time")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Seconds")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def run_baseline_experiment() -> None:
    """Run the sequential continual-learning baseline from start to finish.

    The baseline keeps the same backbone, stage order, and evaluation path as
    the federated run, but removes the federated-specific machinery. The
    execution path is:
    1. resolve the full stage plan and the benchmark probe list,
    2. prepare every required training and held-out dataset before training starts,
    3. build one adapter-injected ViT-MAE model,
    4. walk through the same benchmark-plus-stress stage order,
    5. optimize reconstruction loss only,
    6. preserve the model and optimizer state across dataset transitions,
    7. save the finished training checkpoint and histories,
    8. run one final linear probe on the benchmark datasets only, and
    9. export the probe summary into the dedicated probe folder.
    """
    config = dict(BASELINE_CONFIG)
    config["run_mode"] = "baseline"
    resolve_runtime_config(config)
    config["num_workers"] = resolve_worker_count()
    config["linear_eval_num_workers"] = resolve_worker_count()
    config["pin_memory"] = bool(config["pin_memory"] and config["device"] == "cuda")
    if config["device"] == "cuda" and config["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    benchmark_sequence = validate_client_dataset_sequence(
        BENCHMARK_CLIENT_DATASET_SEQUENCE,
        expected_num_clients=MULTI_DATASET_CONFIG["num_clients"],
        sequence_name="BENCHMARK_CLIENT_DATASET_SEQUENCE",
    )
    noise_sequence = validate_client_dataset_sequence(
        FEDERATED_RETENTION_NOISE_SEQUENCE,
        expected_num_clients=MULTI_DATASET_CONFIG["num_clients"],
        sequence_name="FEDERATED_RETENTION_NOISE_SEQUENCE",
    )
    baseline_stage_plan = build_baseline_stage_plan()
    final_probe_dataset_order = build_dataset_order_by_stage(benchmark_sequence)
    config["benchmark_client_dataset_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in benchmark_sequence.items()
    }
    config["retention_noise_client_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in noise_sequence.items()
    }
    config["client_dataset_sequence"] = {
        "0": [stage_spec["dataset_name"] for stage_spec in baseline_stage_plan]
    }
    config["dataset_order_by_stage"] = [
        stage_spec["dataset_name"] for stage_spec in baseline_stage_plan
    ]
    config["training_stage_plan"] = baseline_stage_plan
    config["linear_probe_dataset_order"] = list(final_probe_dataset_order)
    config["num_stages"] = len(baseline_stage_plan)
    config["total_sequential_rounds"] = config["num_stages"] * config["rounds_per_dataset"]

    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])
    dataset_cache = prepare_dataset_cache(
        config=config,
        training_stage_plan=baseline_stage_plan,
        probe_dataset_order=config["linear_probe_dataset_order"],
        active_logger=baseline_logger,
    )

    model = build_base_model(config)
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["client_lr"],
        weight_decay=config["client_weight_decay"],
    )
    history = initialize_baseline_history()
    global_round_idx = 0

    baseline_logger.info(
        "Starting unified baseline | stages=%s | probe_datasets=%s | rounds_per_dataset=%s | batch_size=%s | linear_probe_epochs=%s | device=%s | output=%s",
        len(config["dataset_order_by_stage"]),
        len(config["linear_probe_dataset_order"]),
        config["rounds_per_dataset"],
        config["batch_size"],
        config["linear_eval_epochs"],
        config["device"],
        output_dirs["root"],
    )

    for stage_index, stage_spec in enumerate(baseline_stage_plan, start=1):
        dataset_name = stage_spec["dataset_name"]
        stage_kind = stage_spec["stage_kind"]
        baseline_logger.info(
            "Starting baseline stage %s/%s | kind=%s | dataset=%s",
            stage_index,
            config["num_stages"],
            stage_kind,
            DATASET_DISPLAY_NAMES[dataset_name],
        )
        stage_losses: List[float] = []

        dataset = dataset_cache[
            f"train::{dataset_name}::{'merged' if stage_kind == 'retention_noise' else 'standard'}"
        ]
        dataloader = create_baseline_dataloader(dataset=dataset, config=config)

        for dataset_round in range(1, config["rounds_per_dataset"] + 1):
            global_round_idx += 1
            round_start = time.time()
            round_result = train_reconstruction_round(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                local_epochs=config["local_epochs"],
                device=config["device"],
                dtype=config["dtype"],
            )
            round_time = time.time() - round_start
            stage_losses.append(round_result["loss"])

            history["round_ids"].append(global_round_idx)
            history["dataset_names"].append(dataset_name)
            history["dataset_rounds"].append(dataset_round)
            history["round_times"].append(round_time)
            history["avg_total_loss"].append(round_result["loss"])
            history["avg_mae_loss"].append(round_result["mae_loss"])

            baseline_logger.info(
                "Baseline round %s/%s complete | kind=%s | dataset=%s | round=%s/%s | loss=%.4f | time=%.2fs",
                global_round_idx,
                config["total_sequential_rounds"],
                stage_kind,
                DATASET_DISPLAY_NAMES[dataset_name],
                dataset_round,
                config["rounds_per_dataset"],
                round_result["loss"],
                round_time,
            )

            save_history(history, output_dirs["metrics"], filename="baseline_training_history.json")
            plot_baseline_training_history(history, output_dirs["plots"])

        history["stage_summaries"].append(
            {
                "dataset_name": dataset_name,
                "stage_kind": stage_kind,
                "train_sample_count": len(dataset),
                "average_round_loss": float(sum(stage_losses) / max(len(stage_losses), 1)),
                "last_round_loss": float(stage_losses[-1]),
            }
        )

        shutdown_dataloader_workers(dataloader)
        del dataloader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_history(history, output_dirs["metrics"], filename="baseline_training_history.json")
        plot_baseline_training_history(history, output_dirs["plots"])

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=model,
        config=config,
        is_final=True,
    )
    save_history(history, output_dirs["metrics"], filename="baseline_training_history.json")
    plot_baseline_training_history(history, output_dirs["plots"])
    baseline_logger.info(
        "Baseline training complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
    )
    try:
        final_probe_results = evaluate_datasets(
            model=model,
            dataset_names=config["linear_probe_dataset_order"],
            config=config,
            dataset_cache=dataset_cache,
        )
        final_linear_probe_summary = summarize_final_linear_probe(
            evaluation_results=final_probe_results,
            dataset_order=config["linear_probe_dataset_order"],
        )
        baseline_logger.info(
            "Baseline final linear probe | datasets=%s | avg_acc=%.4f",
            ", ".join(
                DATASET_DISPLAY_NAMES[name]
                for name in final_linear_probe_summary["dataset_order"]
            ),
            final_linear_probe_summary["average_accuracy"],
        )
        log_final_linear_probe_summary(
            active_logger=baseline_logger,
            run_label="Baseline",
            final_linear_probe_summary=final_linear_probe_summary,
        )
        render_final_linear_probe_outputs(
            final_linear_probe_summary=final_linear_probe_summary,
            metrics_dir=output_dirs["probe_metrics"],
            plots_dir=output_dirs["probe_plots"],
            prefix="baseline",
        )
    except Exception as exc:
        save_json_payload(
            {
                "run_mode": "baseline",
                "error": str(exc),
                "checkpoint_path": str(output_dirs["checkpoints"] / "final_model.pt"),
            },
            output_dirs["probe_metrics"],
            filename="baseline_final_linear_probe_error.json",
        )
        baseline_logger.exception(
            "Baseline final linear probe failed after the final checkpoint was saved."
        )
        raise
    baseline_logger.info(
        "Baseline complete | rounds=%s | final_checkpoint=%s | probe_dir=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
        output_dirs["probe_root"],
    )


def run_federated_experiment() -> None:
    """Run the full two-client federated continual-learning benchmark.

    This is the proposed method implemented by the repository. The execution
    path is:
    1. resolve the full stage plan and the benchmark probe list,
    2. prepare every required training and held-out dataset before training starts,
    3. build the shared adapter-injected ViT-MAE backbone and the global
       prototype bank,
    4. alternate benchmark stages with retention-stress stages,
    5. train one dataset per client for the configured rounds,
    6. upload only trainable adapter weights and local prototype banks,
    7. merge prototypes and aggregate adapter weights on the server,
    8. smooth the aggregated adapter state with server-side EMA,
    9. broadcast the updated global state back to the clients,
    10. preserve both global memory and client-local memory across dataset
        changes,
    11. save the finished training checkpoint and training artifacts,
    12. run one final linear probe on the benchmark datasets only, and
    13. export the probe summary into the dedicated probe folder.
    """
    config = dict(MULTI_DATASET_CONFIG)
    config["run_mode"] = "federated"
    resolve_runtime_config(config)
    config["num_workers"] = resolve_worker_count()
    config["linear_eval_num_workers"] = resolve_worker_count()
    config["pin_memory"] = bool(config["pin_memory"] and config["device"] == "cuda")
    if config["device"] == "cuda" and config["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    benchmark_sequence = validate_client_dataset_sequence(
        BENCHMARK_CLIENT_DATASET_SEQUENCE,
        expected_num_clients=config["num_clients"],
        sequence_name="BENCHMARK_CLIENT_DATASET_SEQUENCE",
    )
    stage_plan = build_federated_stage_plan(config)

    available_gpu_count = config["gpu_count"]
    if available_gpu_count > config["num_clients"]:
        logger.info(
            "Detected %s GPUs and using the first %s for this run.",
            available_gpu_count,
            config["num_clients"],
        )
        config["gpu_count"] = config["num_clients"]

    if config["gpu_count"] not in (0, config["num_clients"]):
        raise ValueError(
            "This entrypoint expects either CPU execution or one selected GPU per client. "
            f"Detected {config['gpu_count']} usable GPUs for {config['num_clients']} clients."
        )

    config["benchmark_client_dataset_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in benchmark_sequence.items()
    }
    config["retention_noise_client_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in normalize_client_dataset_sequence(
            FEDERATED_RETENTION_NOISE_SEQUENCE
        ).items()
    }
    config["client_dataset_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in benchmark_sequence.items()
    }
    config["dataset_order_by_stage"] = build_dataset_order_by_stage(benchmark_sequence)
    config["linear_probe_dataset_order"] = build_dataset_order_by_stage(
        benchmark_sequence
    )
    config["training_stage_plan"] = [
        {
            "stage_kind": stage_spec["stage_kind"],
            "datasets": {
                str(client_index): dataset_name
                for client_index, dataset_name in stage_spec["datasets"].items()
            },
        }
        for stage_spec in stage_plan
    ]
    config["num_stages"] = len(stage_plan)
    config["num_benchmark_stages"] = len(next(iter(benchmark_sequence.values())))
    config["total_sequential_rounds"] = config["num_stages"] * config["rounds_per_dataset"]

    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])
    dataset_cache = prepare_dataset_cache(
        config=config,
        training_stage_plan=stage_plan,
        probe_dataset_order=config["linear_probe_dataset_order"],
        active_logger=logger,
    )

    logger.info(
        "Starting unified federated run | clients=%s | total_stages=%s | benchmark_stages=%s | rounds_per_stage=%s | total_rounds=%s | batch_size=%s | linear_probe_epochs=%s | device=%s | output=%s",
        config["num_clients"],
        config["num_stages"],
        config["num_benchmark_stages"],
        config["rounds_per_dataset"],
        config["total_sequential_rounds"],
        config["batch_size"],
        config["linear_eval_epochs"],
        config["device"],
        output_dirs["root"],
    )

    proto_bank = GlobalPrototypeBank(
        embedding_dim=config["embedding_dim"],
        merge_threshold=config["merge_threshold"],
        ema_alpha=config["server_ema_alpha"],
        device=config["device"],
        max_prototypes=config["max_global_prototypes"],
    )
    model_server = FederatedModelServer()
    base_model = build_base_model(config)
    client_manager = ClientManager(
        base_model=base_model,
        num_clients=config["num_clients"],
        gpu_count=config["gpu_count"],
        dtype=config["dtype"],
        local_epochs=config["local_epochs"],
        optimizer_kwargs={
            "lr": config["client_lr"],
            "weight_decay": config["client_weight_decay"],
        },
        local_update_threshold=config["client_local_update_threshold"],
        local_ema_alpha=config["client_local_ema_alpha"],
        lambda_proto=config["lambda_proto"],
        novelty_buffer_size=config["novelty_buffer_size"],
        novelty_k=config["novelty_k"],
        kmeans_max_iters=config["kmeans_max_iters"],
        kmeans_tol=config["kmeans_tol"],
    )
    gpad_loss = build_gpad_loss(config)

    training_history = initialize_history()
    current_global_weights = extract_trainable_state_dict(base_model)
    global_prototypes: Optional[torch.Tensor] = None
    global_round_idx = 0

    for stage_idx, stage_spec in enumerate(stage_plan):
        stage_number = stage_idx + 1
        stage_kind = stage_spec["stage_kind"]
        stage_dataset_names = [
            stage_spec["datasets"][client_index]
            for client_index in sorted(stage_spec["datasets"])
        ]
        stage_label = ", ".join(
            f"client_{client_index}={DATASET_DISPLAY_NAMES[dataset_name]}"
            for client_index, dataset_name in enumerate(stage_dataset_names)
        )
        logger.info(
            "Starting stage %s/%s | kind=%s | %s",
            stage_number,
            config["num_stages"],
            stage_kind,
            stage_label,
        )

        stage_datasets: List[torch.utils.data.Dataset] = []
        for client_index, dataset_name in enumerate(stage_dataset_names):
            dataset = dataset_cache[
                f"train::{dataset_name}::{'merged' if stage_kind == 'retention_noise' else 'standard'}"
            ]
            stage_datasets.append(dataset)
            logger.info(
                "Client %s ready | dataset=%s | train_samples=%s",
                client_index,
                DATASET_DISPLAY_NAMES[dataset_name],
                len(dataset),
            )

        for stage_round in range(1, config["rounds_per_dataset"] + 1):
            global_round_idx += 1
            logger.info(
                "Round %s/%s starting | stage=%s | stage_round=%s/%s",
                global_round_idx,
                config["total_sequential_rounds"],
                stage_number,
                stage_round,
                config["rounds_per_dataset"],
            )

            # Create fresh DataLoaders each round to prevent BrokenPipeError
            # from stale persistent worker IPC state under memory pressure.
            stage_dataloaders = [
                create_standard_dataloader(
                    dataset=dataset,
                    batch_size=config["batch_size"],
                    num_workers=config["num_workers"],
                    pin_memory=config["pin_memory"],
                    shuffle=config["dataloader_shuffle"],
                    persistent_workers=False,
                    prefetch_factor=config["dataloader_prefetch_factor"],
                )
                for dataset in stage_datasets
            ]
            round_start = time.time()
            client_manager.sync_clients(current_global_weights)

            download_bytes = config["num_clients"] * (
                state_dict_num_bytes(current_global_weights)
                + tensor_num_bytes(global_prototypes)
            )

            client_results = client_manager.train_round(
                dataloaders=stage_dataloaders,
                global_prototypes=global_prototypes,
                gpad_loss_fn=gpad_loss,
            )

            client_payloads: List[Dict[str, Any]] = []
            upload_bytes = 0
            client_prototype_counts: List[int] = []

            for client_index, client in enumerate(client_manager.clients):
                if stage_round == 1:
                    # The first round of a new stage enriches the persistent
                    # local bank with fresh stage concepts instead of resetting
                    # older client memory.
                    local_prototypes = client.generate_prototypes(
                        stage_dataloaders[client_index],
                        K_init=config["k_init_prototypes"],
                    )
                else:
                    local_prototypes = client.get_local_prototypes()

                weights = client.get_trainable_state()
                payload: Dict[str, Any] = {
                    "client_id": f"client_{client_index}",
                    "weights": weights,
                }
                upload_bytes += state_dict_num_bytes(weights)

                if local_prototypes is not None and local_prototypes.numel() > 0:
                    # Prototype payloads move to CPU only for server exchange
                    # and byte accounting. The active training math itself
                    # stays on the client device in float32.
                    cpu_prototypes = local_prototypes.detach().cpu()
                    payload["protos"] = cpu_prototypes
                    upload_bytes += tensor_num_bytes(cpu_prototypes)
                    client_prototype_counts.append(int(cpu_prototypes.size(0)))
                else:
                    client_prototype_counts.append(0)

                client_payloads.append(payload)

            aggregated_prototypes, aggregated_weights = run_server_round(
                proto_manager=proto_bank,
                model_server=model_server,
                client_payloads=client_payloads,
                current_global_weights=current_global_weights,
                round_idx=global_round_idx,
                server_model_ema_alpha=config["server_model_ema_alpha"],
            )

            if aggregated_weights:
                base_model.load_state_dict(aggregated_weights, strict=False)
                current_global_weights = extract_trainable_state_dict(base_model)

            global_prototypes = aggregated_prototypes.detach().cpu()
            round_time = time.time() - round_start

            training_history["round_ids"].append(global_round_idx)
            training_history["round_times"].append(round_time)
            training_history["avg_total_loss"].append(
                average_client_metric(client_results, "loss")
            )
            training_history["avg_mae_loss"].append(
                average_client_metric(client_results, "mae_loss")
            )
            training_history["avg_gpad_loss"].append(
                average_client_metric(client_results, "gpad_loss")
            )
            training_history["avg_anchor_fraction"].append(
                average_client_metric(client_results, "anchored_fraction")
            )
            training_history["avg_local_match_fraction"].append(
                average_client_metric(client_results, "local_match_fraction")
            )
            training_history["avg_novel_fraction"].append(
                average_client_metric(client_results, "novel_fraction")
            )
            training_history["global_prototype_count"].append(int(global_prototypes.size(0)))
            training_history["client_prototype_counts"].append(client_prototype_counts)
            training_history["upload_bytes"].append(upload_bytes)
            training_history["download_bytes"].append(download_bytes)
            training_history["total_communication_bytes"].append(upload_bytes + download_bytes)
            training_history["task_classes"].append(stage_dataset_names)
            training_history["stage_kinds"].append(stage_kind)
            training_history["client_results"].append(
                {
                    "stage": stage_number,
                    "stage_kind": stage_kind,
                    "stage_round": stage_round,
                    "dataset_names": stage_dataset_names,
                    "results": client_results,
                }
            )

            print_round_summary(
                round_idx=global_round_idx,
                num_rounds=config["total_sequential_rounds"],
                client_results=client_results,
                proto_bank=proto_bank,
                round_time=round_time,
                upload_bytes=upload_bytes,
                download_bytes=download_bytes,
            )
            logger.info(
                "Round %s datasets complete | kind=%s | %s | client_prototypes=%s",
                global_round_idx,
                stage_kind,
                ", ".join(DATASET_DISPLAY_NAMES[name] for name in stage_dataset_names),
                client_prototype_counts,
            )

            # Shut down this round's DataLoader workers before the next round
            for dataloader in stage_dataloaders:
                shutdown_dataloader_workers(dataloader)
            del stage_dataloaders

            save_history(training_history, output_dirs["metrics"])
            plot_training_history(
                training_history,
                output_dirs["plots"],
                prefix="main",
            )

        logger.info(
            "Stage %s complete | kind=%s | releasing stage datasets and cached tensors",
            stage_number,
            stage_kind,
        )

        del stage_datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_history(training_history, output_dirs["metrics"])
        plot_training_history(
            training_history,
            output_dirs["plots"],
            prefix="main",
        )

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=base_model,
        proto_bank=proto_bank,
        config=config,
        is_final=True,
    )
    save_history(training_history, output_dirs["metrics"])
    plot_training_history(
        training_history,
        output_dirs["plots"],
        prefix="main",
    )
    logger.info(
        "Federated training complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
    )
    try:
        final_probe_results = evaluate_datasets(
            model=base_model,
            dataset_names=config["linear_probe_dataset_order"],
            config=config,
            dataset_cache=dataset_cache,
        )
        final_linear_probe_summary = summarize_final_linear_probe(
            evaluation_results=final_probe_results,
            dataset_order=config["linear_probe_dataset_order"],
        )
        logger.info(
            "Final linear probe | datasets=%s | avg_acc=%.4f",
            ", ".join(
                DATASET_DISPLAY_NAMES[name]
                for name in final_linear_probe_summary["dataset_order"]
            ),
            final_linear_probe_summary["average_accuracy"],
        )
        log_final_linear_probe_summary(
            active_logger=logger,
            run_label="Federated",
            final_linear_probe_summary=final_linear_probe_summary,
        )
        render_final_linear_probe_outputs(
            final_linear_probe_summary=final_linear_probe_summary,
            metrics_dir=output_dirs["probe_metrics"],
            plots_dir=output_dirs["probe_plots"],
            prefix="main",
        )
    except Exception as exc:
        save_json_payload(
            {
                "run_mode": "federated",
                "error": str(exc),
                "checkpoint_path": str(output_dirs["checkpoints"] / "final_model.pt"),
            },
            output_dirs["probe_metrics"],
            filename="main_final_linear_probe_error.json",
        )
        logger.exception(
            "Federated final linear probe failed after the final checkpoint was saved."
        )
        raise
    logger.info(
        "Training complete | rounds=%s | final_checkpoint=%s | probe_dir=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
        output_dirs["probe_root"],
    )


def main() -> None:
    """Dispatch the single repository entrypoint to the selected run mode."""
    if RUN_MODE == "federated":
        run_federated_experiment()
        return
    if RUN_MODE == "baseline":
        run_baseline_experiment()
        return
    raise ValueError(f"Unsupported RUN_MODE '{RUN_MODE}'. Choose 'federated' or 'baseline'.")


if __name__ == "__main__":
    main()
