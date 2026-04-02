"""Run the full active PODFCSSV experiment pipeline from one file.

This module is intentionally the executable source of truth for the repository.
It keeps the entire paper-facing workflow in one place so the current behavior
can be audited without following a chain of entrypoints.

The high-level flow is:
1. read ``RUN_MODE`` and decide whether to run the federated method or the
   sequential baseline,
2. resolve the real execution device and keep the active numeric path in
   ``torch.float32``,
3. build the shared adapter-injected ViT-MAE backbone,
4. construct the benchmark-plus-stress continual stage plan,
5. train through that stage stream while preserving the correct continual state
   for the selected mode,
6. after each stage, evaluate the benchmark datasets seen so far through
   frozen-feature linear probing,
7. summarize retention metrics and render plots, and
8. save one final checkpoint together with JSON histories at the end.

The file therefore owns configuration, dataset loading, training loops,
aggregation, retention analysis, plotting, and final export for the active
workflow.
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
    "batch_size": 96,
    "client_lr": 1e-4,
    "client_weight_decay": 0.05,
    "gpu_count": 0,
    "device": "cpu",
    "dtype": torch.float32,
    "pin_memory": True,
    "dataloader_shuffle": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 4,
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
    "linear_eval_batch_size": 256,
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

    Every run writes into three stable locations:
    1. ``checkpoints`` for model snapshots.
    2. ``metrics`` for JSON histories.
    3. ``plots`` for publication-oriented visualizations.
    """
    root = Path(save_dir)
    checkpoints_dir = root / "checkpoints"
    metrics_dir = root / "metrics"
    plots_dir = root / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "checkpoints": checkpoints_dir,
        "metrics": metrics_dir,
        "plots": plots_dir,
    }


def resolve_worker_count() -> int:
    """Choose one runtime worker count from the available CPU resources.

    The current pipeline derives dataloader workers from the host CPU instead
    of hard-coding separate values for training and evaluation. The rule is
    intentionally simple:
    1. read the visible CPU count,
    2. reserve one core for the main process, and
    3. keep at least one worker so the loaders remain asynchronous.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def build_base_model(config: Dict[str, Any]) -> nn.Module:
    """Build the shared adapter-injected MAE backbone used by all modes.

    The construction path is:
    1. Load the selected pretrained ViT-MAE checkpoint.
    2. Inject adapters into the upper half of the encoder.
    3. Move the model to the configured device and dtype.
    4. Log the full and trainable parameter counts.
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
    """Build the GPAD loss object from the current config values."""
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


def save_checkpoint(
    checkpoint_dir: Path,
    round_idx: int,
    base_model: nn.Module,
    config: Dict[str, Any],
    proto_bank: Optional[GlobalPrototypeBank] = None,
    training_history: Optional[Dict[str, Any]] = None,
    is_final: bool = False,
    include_training_history: bool = True,
) -> Path:
    """Save the final experiment snapshot in the repository checkpoint format.

    The active workflow writes checkpoints only after the whole run completes.
    Each saved file contains:
    1. the final round index,
    2. the trainable adapter state,
    3. the serialized runtime config, and
    4. the optional global prototype bank for federated runs.

    The saved payload is intended for later analysis and downstream evaluation
    of the finished run rather than for exact mid-training crash recovery.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "round": round_idx,
        "model_state_dict": extract_trainable_state_dict(base_model),
        "config": serialize_config(config),
    }
    if proto_bank is not None:
        checkpoint["global_prototypes"] = proto_bank.get_prototypes().detach().cpu()
    if include_training_history and training_history is not None:
        checkpoint["training_history"] = training_history
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

    The history dictionary stores two kinds of information:
    1. round-wise optimization and communication metrics collected during
       federated training, and
    2. stage-wise linear-probe retention summaries collected after each stage.

    Keeping both views in one structure ensures that JSON exports and plots
    stay synchronized.
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
        "used_for_linear_probe": [],
        "stage_evaluations": [],
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
    1. Benchmark stages that count toward the reported evaluation set.
    2. Stress stages that exist only to create extra forgetting pressure.
    3. An optional final stress stage after the last benchmark stage when the
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
                "use_for_linear_probe": True,
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
                    "use_for_linear_probe": False,
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
    4. Preserve the ``use_for_linear_probe`` flag so only benchmark datasets
       contribute to the reported evaluation set.
    """
    federated_stage_plan = build_federated_stage_plan(dict(MULTI_DATASET_CONFIG))
    baseline_stage_plan: List[Dict[str, Any]] = []

    for stage_spec in federated_stage_plan:
        for client_index in sorted(stage_spec["datasets"]):
            baseline_stage_plan.append(
                {
                    "stage_kind": stage_spec["stage_kind"],
                    "use_for_linear_probe": stage_spec["use_for_linear_probe"],
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
    """Create a deterministic fixed-size train/eval split for one combined dataset."""
    required_samples = train_samples + eval_samples
    dataset_size = len(dataset)
    if dataset_size < required_samples:
        raise ValueError(
            f"{DATASET_DISPLAY_NAMES[dataset_name]} has only {dataset_size} samples, "
            f"but the fixed split needs {required_samples} samples."
        )

    split_seed = stable_int_from_parts(seed, dataset_name, "fixed_split")
    generator = torch.Generator().manual_seed(split_seed)
    selected_indices = torch.randperm(dataset_size, generator=generator)[:required_samples].tolist()
    train_indices = selected_indices[:train_samples]
    eval_indices = selected_indices[train_samples:]
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
    """Load one dataset exactly the way the current pipeline expects it.

    All split policy decisions live here so the training and evaluation paths do
    not drift apart over time.

    The rules are:
    1. benchmark datasets use their normal train-side split in full,
    2. ``EuroSAT`` is handled as a fixed deterministic ``22000 / 5000`` split,
    3. stress datasets may merge all official splits into one self-supervised
       training pool when ``merge_all_training_splits`` is enabled, and
    4. optional deterministic sample shaping is applied only when an explicit
       cap or minimum is requested by the caller.
    """
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if train and merge_all_training_splits:
        # Stress datasets are used as broad self-supervised training pools, so
        # the training path intentionally consumes every official split they expose.
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
    """Load the held-out split used by the stage-wise benchmark evaluation.

    The reporting path is centralized here so all retention metrics are based
    on one consistent split policy. The current rules are:
    1. reuse the repository's fixed `EuroSAT` held-out split,
    2. use the official held-out split for datasets that already define one,
    3. use `Country211` validation rather than test in the active benchmark,
       and
    4. keep any special merge logic for held-out reporting in one place.
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


def create_standard_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> DataLoader:
    """Create one dataloader using the current project-wide loading policy.

    This helper keeps the low-level loading behavior aligned across training
    and evaluation:
    1. respect the caller's dataset, batch size, and shuffle policy,
    2. use the resolved runtime worker count,
    3. enable pinned memory only when the runtime configuration says it is
       useful,
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

    Both stage-wise evaluation paths use the encoder alone and set the MAE mask
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

    This is the frozen-feature stage used during built-in retention analysis:
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

    This function is the core of the active retention-evaluation path. For one
    dataset pair it:
    1. remaps labels onto a contiguous class range,
    2. builds a single linear classification head,
    3. trains that head on top of frozen encoder features,
    4. evaluates the head on the held-out split, and
    5. returns the held-out accuracy used in the forgetting summaries.

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
        train_loader_kwargs["persistent_workers"] = config["dataloader_persistent_workers"]
        train_loader_kwargs["prefetch_factor"] = config["dataloader_prefetch_factor"]
    train_loader = DataLoader(**train_loader_kwargs)

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


def evaluate_datasets(
    model: nn.Module,
    dataset_names: Sequence[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate one stage checkpoint on the currently seen benchmark datasets.

    This helper is called after every stage. For each benchmark dataset in the
    current seen set it:
    1. loads the train split used to fit the probe,
    2. loads the held-out split used for reporting,
    3. extracts full-image encoder features with masking disabled,
    4. trains a linear classifier on those frozen features, and
    5. records the held-out accuracy used by the retention summaries.

    Stress datasets never enter this reporting path directly.
    """
    results: Dict[str, float] = {}
    for dataset_name in dataset_names:
        # Only benchmark datasets are reported here. Stress datasets affect the
        # checkpoint through training, but they are excluded from the headline metrics.
        train_dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
        )
        test_dataset = load_named_evaluation_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            seed=config["seed"],
        )

        train_loader = create_standard_dataloader(
            dataset=train_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            persistent_workers=config["dataloader_persistent_workers"],
            prefetch_factor=config["dataloader_prefetch_factor"],
        )
        test_loader = create_standard_dataloader(
            dataset=test_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
            persistent_workers=config["dataloader_persistent_workers"],
            prefetch_factor=config["dataloader_prefetch_factor"],
        )

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

        results[dataset_name] = run_linear_probe(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            config=config,
        )

    return results


def evaluate_seen_datasets(
    model: nn.Module,
    seen_dataset_names: List[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate the benchmark datasets that have been seen so far."""
    return evaluate_datasets(model=model, dataset_names=seen_dataset_names, config=config)


def summarize_stage_evaluation(
    history: Dict[str, Any],
    stage_number: int,
    stage_kind: str,
    evaluated_dataset_names: Sequence[str],
    accuracies: Dict[str, float],
    history_key: str = "stage_evaluations",
) -> Dict[str, Any]:
    """Turn one stage's accuracies into the saved retention summary block.

    The forgetting curves are built here. For the current stage the function:
    1. records the linear-probe accuracy for each seen benchmark dataset,
    2. compares that score against the dataset's earlier stage history,
    3. derives forgetting, retention ratio, and backward-transfer values, and
    4. appends one compact summary object to the run history.

    The plotting and JSON-export code then reads directly from this summarized
    history instead of recomputing the retention math later.
    """
    previous_stage_evaluations = history.get(history_key, [])
    forgetting: Dict[str, float] = {}
    retention_ratio: Dict[str, float] = {}
    backward_transfer: Dict[str, float] = {}

    for dataset_name in evaluated_dataset_names:
        current_accuracy = float(accuracies.get(dataset_name, 0.0))
        previous_accuracies = [
            float(stage_evaluation["accuracies"][dataset_name])
            for stage_evaluation in previous_stage_evaluations
            if dataset_name in stage_evaluation["accuracies"]
        ]
        best_accuracy = max(previous_accuracies + [current_accuracy]) if previous_accuracies else current_accuracy
        first_accuracy = previous_accuracies[0] if previous_accuracies else current_accuracy

        forgetting[dataset_name] = float(best_accuracy - current_accuracy)
        retention_ratio[dataset_name] = float(current_accuracy / max(best_accuracy, 1e-8))
        backward_transfer[dataset_name] = float(current_accuracy - first_accuracy)

    average_accuracy = float(
        sum(float(accuracies.get(dataset_name, 0.0)) for dataset_name in evaluated_dataset_names)
        / max(len(evaluated_dataset_names), 1)
    )
    average_forgetting = float(
        sum(forgetting.values()) / max(len(forgetting), 1)
    )
    average_retention = float(
        sum(retention_ratio.values()) / max(len(retention_ratio), 1)
    )
    average_backward_transfer = float(
        sum(backward_transfer.values()) / max(len(backward_transfer), 1)
    )

    stage_summary = {
        "stage": stage_number,
        "stage_kind": stage_kind,
        "evaluated_datasets": list(evaluated_dataset_names),
        "accuracies": {dataset_name: float(accuracies[dataset_name]) for dataset_name in evaluated_dataset_names},
        "average_accuracy": average_accuracy,
        "forgetting": forgetting,
        "average_forgetting": average_forgetting,
        "retention_ratio": retention_ratio,
        "average_retention": average_retention,
        "backward_transfer": backward_transfer,
        "average_backward_transfer": average_backward_transfer,
    }
    history.setdefault(history_key, []).append(stage_summary)
    return stage_summary


def plot_stage_evaluation_heatmap(
    history: Dict[str, Any],
    dataset_order: Sequence[str],
    plots_dir: Path,
    prefix: str,
    history_key: str = "stage_evaluations",
    title: str = "Stage-Wise Benchmark Accuracy",
) -> Path:
    """Plot stage-by-dataset benchmark accuracy so forgetting is easy to inspect."""
    output_path = plots_dir / f"{prefix}_stage_accuracy_heatmap.png"
    stage_evaluations = history.get(history_key, [])
    if not stage_evaluations or not dataset_order:
        return output_path

    accuracy_matrix = np.full((len(stage_evaluations), len(dataset_order)), np.nan, dtype=float)
    stage_labels: List[str] = []
    for row_index, stage_evaluation in enumerate(stage_evaluations):
        stage_labels.append(
            f"S{stage_evaluation['stage']} ({stage_evaluation['stage_kind']})"
        )
        for column_index, dataset_name in enumerate(dataset_order):
            if dataset_name in stage_evaluation["accuracies"]:
                accuracy_matrix[row_index, column_index] = stage_evaluation["accuracies"][dataset_name]

    figure, axis = plt.subplots(figsize=(max(9, len(dataset_order) * 1.5), max(4, len(stage_evaluations) * 0.9)))
    masked_matrix = np.ma.masked_invalid(accuracy_matrix)
    image = axis.imshow(masked_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(dataset_order)))
    axis.set_xticklabels([DATASET_DISPLAY_NAMES[name] for name in dataset_order], rotation=45, ha="right")
    axis.set_yticks(range(len(stage_labels)))
    axis.set_yticklabels(stage_labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_stage_metric_curves(
    history: Dict[str, Any],
    plots_dir: Path,
    prefix: str,
    history_key: str = "stage_evaluations",
    title_prefix: str = "Linear Probe",
) -> Path:
    """Plot the stage-wise averages that summarize retention quality."""
    output_path = plots_dir / f"{prefix}_stage_metric_curves.png"
    stage_evaluations = history.get(history_key, [])
    if not stage_evaluations:
        return output_path

    stage_indices = [stage_evaluation["stage"] for stage_evaluation in stage_evaluations]
    average_accuracy = [stage_evaluation["average_accuracy"] for stage_evaluation in stage_evaluations]
    average_forgetting = [stage_evaluation["average_forgetting"] for stage_evaluation in stage_evaluations]
    average_retention = [stage_evaluation["average_retention"] for stage_evaluation in stage_evaluations]
    average_backward_transfer = [stage_evaluation["average_backward_transfer"] for stage_evaluation in stage_evaluations]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(stage_indices, average_accuracy, marker="o")
    axes[0, 0].set_title(f"{title_prefix} Average Benchmark Accuracy")
    axes[0, 0].set_xlabel("Stage")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0.0, 1.0)

    axes[0, 1].plot(stage_indices, average_forgetting, marker="o")
    axes[0, 1].set_title(f"{title_prefix} Average Forgetting")
    axes[0, 1].set_xlabel("Stage")
    axes[0, 1].set_ylabel("Forgetting")

    axes[1, 0].plot(stage_indices, average_retention, marker="o")
    axes[1, 0].set_title(f"{title_prefix} Average Retention Ratio")
    axes[1, 0].set_xlabel("Stage")
    axes[1, 0].set_ylabel("Retention")
    axes[1, 0].set_ylim(0.0, 1.05)

    axes[1, 1].plot(stage_indices, average_backward_transfer, marker="o")
    axes[1, 1].set_title(f"{title_prefix} Average Backward Transfer")
    axes[1, 1].set_xlabel("Stage")
    axes[1, 1].set_ylabel("BWT")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_final_forgetting_bars(
    history: Dict[str, Any],
    dataset_order: Sequence[str],
    plots_dir: Path,
    prefix: str,
    history_key: str = "stage_evaluations",
    title: str = "Final Forgetting by Dataset",
) -> Path:
    """Plot the final forgetting value for each benchmark dataset."""
    output_path = plots_dir / f"{prefix}_final_forgetting.png"
    stage_evaluations = history.get(history_key, [])
    if not stage_evaluations:
        return output_path

    final_stage = stage_evaluations[-1]
    forgetting_values = [
        float(final_stage["forgetting"].get(dataset_name, 0.0))
        for dataset_name in dataset_order
        if dataset_name in final_stage["accuracies"]
    ]
    plotted_dataset_names = [
        dataset_name
        for dataset_name in dataset_order
        if dataset_name in final_stage["accuracies"]
    ]
    if not plotted_dataset_names:
        return output_path

    figure, axis = plt.subplots(figsize=(max(8, len(plotted_dataset_names) * 1.3), 5))
    x_positions = np.arange(len(plotted_dataset_names))
    axis.bar(x_positions, forgetting_values, color="tab:orange")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([DATASET_DISPLAY_NAMES[name] for name in plotted_dataset_names], rotation=45, ha="right")
    axis.set_title(title)
    axis.set_ylabel("Forgetting")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def render_all_stage_evaluation_plots(
    history: Dict[str, Any],
    dataset_order: Sequence[str],
    plots_dir: Path,
    prefix: str,
) -> None:
    """Write the linear-probe retention plots used by the current pipeline."""
    plot_stage_evaluation_heatmap(
        history,
        dataset_order,
        plots_dir,
        prefix,
        history_key="stage_evaluations",
        title="Linear-Probe Stage Accuracy",
    )
    plot_stage_metric_curves(
        history,
        plots_dir,
        prefix,
        history_key="stage_evaluations",
        title_prefix="Linear Probe",
    )
    plot_final_forgetting_bars(
        history,
        dataset_order,
        plots_dir,
        prefix,
        history_key="stage_evaluations",
        title="Linear-Probe Final Forgetting by Dataset",
    )


def initialize_baseline_history() -> Dict[str, Any]:
    """Create the history structure written by the baseline mode.

    The baseline records three groups of information:
    1. Round-wise reconstruction metrics.
    2. Per-stage dataset summaries.
    3. Stage-wise linear-probe retention summaries.
    """
    return {
        "round_ids": [],
        "dataset_names": [],
        "dataset_rounds": [],
        "round_times": [],
        "avg_total_loss": [],
        "avg_mae_loss": [],
        "stage_summaries": [],
        "stage_evaluations": [],
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

    The baseline keeps the same backbone, data stream, and stage-wise
    evaluation as the federated run, but removes the federated-specific
    machinery. The execution flow is:
    1. build one adapter-injected ViT-MAE model,
    2. walk through the same benchmark-plus-stress stage order used by the
       federated run,
    3. optimize reconstruction loss only,
    4. preserve the model and optimizer state across every dataset transition,
    5. run stage-wise linear probing on the benchmark datasets seen so far, and
    6. save one final checkpoint together with the histories and plots.
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
    benchmark_dataset_order = build_dataset_order_by_stage(benchmark_sequence)
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
    config["linear_probe_dataset_order"] = list(benchmark_dataset_order)
    config["num_stages"] = len(baseline_stage_plan)
    config["total_sequential_rounds"] = config["num_stages"] * config["rounds_per_dataset"]

    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])

    model = build_base_model(config)
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["client_lr"],
        weight_decay=config["client_weight_decay"],
    )
    history = initialize_baseline_history()
    global_round_idx = 0
    seen_benchmark_dataset_names: List[str] = []

    baseline_logger.info(
        "Starting unified baseline | stages=%s | benchmark_datasets=%s | rounds_per_dataset=%s | batch_size=%s | linear_probe_epochs=%s | device=%s | output=%s",
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

        dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
            max_samples=None,
            min_samples=None,
            merge_all_training_splits=(stage_kind == "retention_noise"),
        )
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
                "use_for_linear_probe": stage_spec["use_for_linear_probe"],
                "train_sample_count": len(dataset),
                "average_round_loss": float(sum(stage_losses) / max(len(stage_losses), 1)),
                "last_round_loss": float(stage_losses[-1]),
            }
        )
        if stage_spec["use_for_linear_probe"] and dataset_name not in seen_benchmark_dataset_names:
            seen_benchmark_dataset_names.append(dataset_name)

        stage_accuracies = evaluate_seen_datasets(
            model=model,
            seen_dataset_names=seen_benchmark_dataset_names,
            config=config,
        )
        stage_evaluation = summarize_stage_evaluation(
            history=history,
            stage_number=stage_index,
            stage_kind=stage_kind,
            evaluated_dataset_names=seen_benchmark_dataset_names,
            accuracies=stage_accuracies,
            history_key="stage_evaluations",
        )
        baseline_logger.info(
            "Baseline stage %s linear probe | kind=%s | datasets=%s | avg_acc=%.4f | avg_forgetting=%.4f | avg_retention=%.4f | avg_bwt=%.4f",
            stage_index,
            stage_kind,
            ", ".join(DATASET_DISPLAY_NAMES[name] for name in seen_benchmark_dataset_names),
            stage_evaluation["average_accuracy"],
            stage_evaluation["average_forgetting"],
            stage_evaluation["average_retention"],
            stage_evaluation["average_backward_transfer"],
        )

        del dataloader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_history(history, output_dirs["metrics"], filename="baseline_training_history.json")
        plot_baseline_training_history(history, output_dirs["plots"])
        render_all_stage_evaluation_plots(
            history,
            config["linear_probe_dataset_order"],
            output_dirs["plots"],
            "baseline",
        )

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=model,
        config=config,
        training_history=history,
        is_final=True,
        include_training_history=False,
    )
    save_history(history, output_dirs["metrics"], filename="baseline_training_history.json")
    plot_baseline_training_history(history, output_dirs["plots"])
    render_all_stage_evaluation_plots(
        history,
        config["linear_probe_dataset_order"],
        output_dirs["plots"],
        "baseline",
    )
    baseline_logger.info(
        "Baseline complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
    )


def run_federated_experiment() -> None:
    """Run the full two-client federated continual-learning benchmark.

    This is the proposed method implemented by the repository. The execution
    flow is:
    1. build the shared adapter-injected ViT-MAE backbone and the global
       prototype bank,
    2. alternate benchmark stages with retention-stress stages,
    3. train one dataset per client for the configured number of rounds,
    4. upload only trainable adapter weights and local prototype banks,
    5. merge prototypes and aggregate adapter weights on the server,
    6. smooth the aggregated adapter state with server-side EMA,
    7. broadcast the updated global state back to the clients,
    8. preserve both global memory and client-local memory across dataset
       changes,
    9. run stage-wise linear probing on the benchmark datasets seen so far, and
    10. save the final checkpoint, communication histories, and plots.
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
    config["linear_probe_dataset_order"] = list(config["dataset_order_by_stage"])
    config["training_stage_plan"] = [
        {
            "stage_kind": stage_spec["stage_kind"],
            "use_for_linear_probe": stage_spec["use_for_linear_probe"],
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
    seen_benchmark_dataset_names: List[str] = []

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
            dataset = load_named_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                train=True,
                seed=config["seed"],
                max_samples=None,
                min_samples=None,
                merge_all_training_splits=(stage_kind == "retention_noise"),
            )
            stage_datasets.append(dataset)
            logger.info(
                "Client %s ready | dataset=%s | train_samples=%s",
                client_index,
                DATASET_DISPLAY_NAMES[dataset_name],
                len(dataset),
            )

        stage_dataloaders = [
            create_standard_dataloader(
                dataset=dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                shuffle=config["dataloader_shuffle"],
                persistent_workers=config["dataloader_persistent_workers"],
                prefetch_factor=config["dataloader_prefetch_factor"],
            )
            for dataset in stage_datasets
        ]

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
                    # The first round of each stage injects fresh stage concepts
                    # into the persistent local bank instead of resetting it.
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
                    # and byte accounting; the active training math stays on-device.
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
            training_history["used_for_linear_probe"].append(stage_spec["use_for_linear_probe"])
            training_history["client_results"].append(
                {
                    "stage": stage_number,
                    "stage_kind": stage_kind,
                    "stage_round": stage_round,
                    "dataset_names": stage_dataset_names,
                    "use_for_linear_probe": stage_spec["use_for_linear_probe"],
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

            save_history(training_history, output_dirs["metrics"])
            plot_training_history(
                training_history,
                output_dirs["plots"],
                prefix="main",
            )

        logger.info(
            "Stage %s complete | kind=%s | releasing stage dataloaders and cached tensors",
            stage_number,
            stage_kind,
        )

        if stage_spec["use_for_linear_probe"]:
            for dataset_name in stage_dataset_names:
                if dataset_name not in seen_benchmark_dataset_names:
                    seen_benchmark_dataset_names.append(dataset_name)

        if seen_benchmark_dataset_names:
            stage_accuracies = evaluate_seen_datasets(
                model=base_model,
                seen_dataset_names=seen_benchmark_dataset_names,
                config=config,
            )
            stage_evaluation = summarize_stage_evaluation(
                history=training_history,
                stage_number=stage_number,
                stage_kind=stage_kind,
                evaluated_dataset_names=seen_benchmark_dataset_names,
                accuracies=stage_accuracies,
                history_key="stage_evaluations",
            )
            logger.info(
                "Stage %s linear probe | kind=%s | datasets=%s | avg_acc=%.4f | avg_forgetting=%.4f | avg_retention=%.4f | avg_bwt=%.4f",
                stage_number,
                stage_kind,
                ", ".join(DATASET_DISPLAY_NAMES[name] for name in seen_benchmark_dataset_names),
                stage_evaluation["average_accuracy"],
                stage_evaluation["average_forgetting"],
                stage_evaluation["average_retention"],
                stage_evaluation["average_backward_transfer"],
            )

        del stage_dataloaders
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
        render_all_stage_evaluation_plots(
            training_history,
            config["linear_probe_dataset_order"],
            output_dirs["plots"],
            "main",
        )

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=base_model,
        proto_bank=proto_bank,
        training_history=training_history,
        config=config,
        is_final=True,
        include_training_history=False,
    )
    save_history(training_history, output_dirs["metrics"])
    plot_training_history(
        training_history,
        output_dirs["plots"],
        prefix="main",
    )
    render_all_stage_evaluation_plots(
        training_history,
        config["linear_probe_dataset_order"],
        output_dirs["plots"],
        "main",
    )
    logger.info(
        "Training complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
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
