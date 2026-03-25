"""Run the federated continual-learning experiment used as the main pipeline.

The current entrypoint is the 2-client, 6-dataset sequential benchmark:
1. Build one pretrained ViT-MAE backbone with injected residual adapters.
2. Place one client on each usable GPU when CUDA really works.
3. Feed each client one dataset per stage across three sequential stages.
4. Train locally with MAE reconstruction and GPAD.
5. Merge trainable adapter weights and local prototypes on the server.
6. Carry global and local memory forward into the next dataset stage.
7. Save checkpoints, histories, communication statistics, and plots.

Every training split is fitted to the same effective sample budget so both
clients spend roughly the same number of steps on each stage, and the image
pipeline intentionally avoids ImageNet-style normalization.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
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

CONFIG: Dict[str, Any] = {
    "seed": 42,
    "num_clients": 2,
    "local_epochs": 1,
    "batch_size": 64,
    "client_lr": 1e-4,
    "client_weight_decay": 0.05,
    "gpu_count": 0,
    "device": "cpu",
    "dtype": torch.float32,
    "num_workers": 2,
    "pin_memory": True,
    "dataloader_shuffle": True,
    "pretrained_model_name": "facebook/vit-mae-large",
    "embedding_dim": 1024,
    "image_size": 224,
    "adapter_bottleneck_dim": 256,
    "merge_threshold": 0.85,
    "server_ema_alpha": 0.1,
    "server_model_ema_alpha": 0.1,
    "max_global_prototypes": 2000,
    "gpad_base_tau": 0.8,
    "gpad_temp_gate": 0.2,
    "gpad_lambda_entropy": 0.3,
    "gpad_soft_assign_temp": 0.07,
    "gpad_epsilon": 1e-8,
    "lambda_proto": 0.01,
    "k_init_prototypes": 50,
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
    "num_workers": 2,
    "linear_eval_batch_size": 256,
    "linear_eval_epochs": 5,
    "linear_eval_lr": 1e-2,
    "linear_eval_weight_decay": 1e-4,
    "linear_eval_num_workers": 2,
    "linear_eval_train_samples": None,
    "linear_eval_test_samples": None,
    "min_linear_eval_train_samples": 0,
    "min_linear_eval_test_samples": 0,
    "max_global_prototypes": 2000,
    "train_samples_per_dataset": 10000,
    "min_train_samples_per_dataset": 1000,
    "save_dir": "multidataset_outputs_2client",
}

EUROSAT_TRAIN_SPLIT_SAMPLES = 10000
EUROSAT_EVAL_SPLIT_SAMPLES = 5000

CLIENT_DATASET_SEQUENCE: Dict[int, List[str]] = {
    0: ["eurosat", "oxfordiiitpet", "flowers102"],
    1: ["gtsrb", "fgvcaircraft", "dtd"],
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
    """Populate device information after checking whether CUDA kernels actually run."""
    usable_gpu_count = get_usable_cuda_device_count()
    if usable_gpu_count > 0:
        config["gpu_count"] = usable_gpu_count
        config["device"] = "cuda"
    else:
        config["gpu_count"] = 0
        config["device"] = "cpu"


def convert_to_rgb(image: Any) -> Any:
    """Convert input images to RGB before any resize or tensor conversion."""
    if hasattr(image, "convert"):
        return image.convert("RGB")
    return image


def serialize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config values into a JSON-safe representation for checkpoints."""
    serialized: Dict[str, Any] = {}
    for key, value in config.items():
        serialized[key] = str(value) if isinstance(value, torch.dtype) else value
    return serialized


def tensor_num_bytes(tensor: Optional[torch.Tensor]) -> int:
    """Return how many bytes one tensor would occupy on the wire."""
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def state_dict_num_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Return the total communication cost of one state-dict payload."""
    return sum(tensor_num_bytes(tensor) for tensor in state_dict.values())


def extract_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU copy of only the trainable parameters in the model."""
    trainable_state: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_state[name] = parameter.detach().cpu().clone()
    return trainable_state


def prepare_output_dirs(save_dir: str) -> Dict[str, Path]:
    """Create the checkpoint, metric, and plot directories for one run."""
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


def build_base_model(config: Dict[str, Any]) -> nn.Module:
    """Load pretrained ViT-MAE and inject adapters into the shared trainable layers."""
    model = ViTMAEForPreTraining.from_pretrained(config["pretrained_model_name"])
    model = inject_adapters(model, bottleneck_dim=config["adapter_bottleneck_dim"])
    model = model.to(device=config["device"], dtype=config["dtype"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
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
    """Return the average of one logged client metric across the current round."""
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
    """Save one checkpoint with the trainable weights and optional run metadata."""
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
    """Create the federated training summary figure used by the main run."""
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
    """Print the key round metrics shown after each communication round."""
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
    """Create the round-history structure used by the federated trainer."""
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
        "client_results": [],
    }


def validate_dataset_schedule(config: Dict[str, Any]) -> None:
    """Check that the hardcoded client schedule matches the requested run shape."""
    if len(CLIENT_DATASET_SEQUENCE) != config["num_clients"]:
        raise ValueError(
            "The client schedule does not match the configured client count. "
            f"Expected {config['num_clients']} clients but found {len(CLIENT_DATASET_SEQUENCE)}."
        )

    stage_counts = {len(dataset_names) for dataset_names in CLIENT_DATASET_SEQUENCE.values()}
    if len(stage_counts) != 1:
        raise ValueError("Every client must receive the same number of sequential datasets.")

    all_dataset_names = [
        dataset_name
        for client_index in sorted(CLIENT_DATASET_SEQUENCE)
        for dataset_name in CLIENT_DATASET_SEQUENCE[client_index]
    ]
    duplicate_names = sorted(
        dataset_name
        for dataset_name in set(all_dataset_names)
        if all_dataset_names.count(dataset_name) > 1
    )
    if duplicate_names:
        raise ValueError(f"Duplicate dataset entries found: {duplicate_names}")

    missing_display_names = sorted(
        dataset_name
        for dataset_name in all_dataset_names
        if dataset_name not in DATASET_DISPLAY_NAMES
    )
    if missing_display_names:
        raise ValueError(
            f"Missing display-name entries for datasets: {missing_display_names}"
        )


def get_num_stages() -> int:
    """Return how many sequential stages exist in the current client schedule."""
    return len(next(iter(CLIENT_DATASET_SEQUENCE.values())))


def get_stage_dataset_names(stage_index: int) -> List[str]:
    """Return the dataset names assigned to one stage in client order."""
    return [
        CLIENT_DATASET_SEQUENCE[client_index][stage_index]
        for client_index in sorted(CLIENT_DATASET_SEQUENCE)
    ]


def build_dataset_order_by_stage(
    dataset_sequence: Optional[Dict[Any, Sequence[str]]] = None,
) -> List[str]:
    """Flatten the client schedule into the stage-by-stage order used for logs and eval."""
    sequence = dataset_sequence or CLIENT_DATASET_SEQUENCE
    normalized_sequence = {
        int(client_index): list(dataset_names)
        for client_index, dataset_names in sequence.items()
    }
    stage_count = len(next(iter(normalized_sequence.values())))

    dataset_order: List[str] = []
    for stage_index in range(stage_count):
        for client_index in sorted(normalized_sequence):
            dataset_order.append(normalized_sequence[client_index][stage_index])
    return dataset_order


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
    """Fit one split to the configured sample budget in a deterministic way.

    The sample-budget rule is:
    1. Fail early if the split is below the required minimum.
    2. Keep the full split when no target sample count is requested.
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
    """Resize each image for ViT-MAE without applying dataset-specific normalization."""
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
) -> torch.utils.data.Dataset:
    """Load one named dataset and optionally fit it to a training or eval budget.

    This helper centralizes every dataset-specific split rule so the training
    and evaluation entrypoints all go through the same loading logic.
    """
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if dataset_name == "eurosat":
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
            split="train" if train else "test",
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
    """Load the official held-out split or split-combination used for evaluation."""
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
) -> DataLoader:
    """Create a dataloader with the shared project defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def pool_model_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Pool the last hidden state exactly the same way training code does."""
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

    This is the frozen-feature stage used by linear probing:
    1. Run the MAE encoder on full images.
    2. Pool the last hidden state into one embedding per image.
    3. L2-normalize the embedding.
    4. Return features and labels on CPU.
    """
    model.eval()
    feature_batches: List[torch.Tensor] = []
    label_batches: List[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device=device, dtype=dtype)
        outputs = model(images, output_hidden_states=True)
        embeddings = pool_model_hidden_states(outputs.hidden_states[-1])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        feature_batches.append(embeddings.cpu())
        label_batches.append(torch.as_tensor(labels, dtype=torch.long).cpu())

    if not feature_batches:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)

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
    """Train one linear probe on frozen features and return held-out accuracy."""
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["linear_eval_batch_size"],
        shuffle=True,
        num_workers=config["linear_eval_num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=False,
    )

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
    """Evaluate frozen representations on one or more datasets with linear probing."""
    results: Dict[str, float] = {}
    for dataset_name in dataset_names:
        train_dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
            max_samples=config.get("linear_eval_train_samples"),
            min_samples=config.get("min_linear_eval_train_samples"),
        )
        test_dataset = load_named_evaluation_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            seed=config["seed"],
            max_samples=config.get("linear_eval_test_samples"),
            min_samples=config.get("min_linear_eval_test_samples"),
        )

        train_loader = create_standard_dataloader(
            dataset=train_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
        )
        test_loader = create_standard_dataloader(
            dataset=test_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
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
    """Backward-compatible alias used by the standalone evaluation script."""
    return evaluate_datasets(model=model, dataset_names=seen_dataset_names, config=config)


def main() -> None:
    """Run the full 2-client sequential federated continual-learning benchmark."""
    config = dict(MULTI_DATASET_CONFIG)
    resolve_runtime_config(config)
    validate_dataset_schedule(config)

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

    config["client_dataset_sequence"] = {
        str(client_index): list(dataset_names)
        for client_index, dataset_names in CLIENT_DATASET_SEQUENCE.items()
    }
    config["dataset_order_by_stage"] = build_dataset_order_by_stage()
    config["num_stages"] = get_num_stages()
    config["total_sequential_rounds"] = config["num_stages"] * config["rounds_per_dataset"]

    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])

    # Build the shared global objects once before the dataset stages begin.
    logger.info(
        "Starting 2-client sequential run | clients=%s | stages=%s | rounds_per_stage=%s | total_rounds=%s | train_budget=%s | device=%s | output=%s",
        config["num_clients"],
        config["num_stages"],
        config["rounds_per_dataset"],
        config["total_sequential_rounds"],
        config["train_samples_per_dataset"],
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

    for stage_idx in range(config["num_stages"]):
        stage_number = stage_idx + 1
        stage_dataset_names = get_stage_dataset_names(stage_idx)
        stage_label = ", ".join(
            f"client_{client_index}={DATASET_DISPLAY_NAMES[dataset_name]}"
            for client_index, dataset_name in enumerate(stage_dataset_names)
        )
        logger.info("Starting stage %s/%s | %s", stage_number, config["num_stages"], stage_label)

        # Load exactly one dataset per client for the current stage.
        stage_datasets: List[torch.utils.data.Dataset] = []
        for client_index, dataset_name in enumerate(stage_dataset_names):
            dataset = load_named_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                train=True,
                seed=config["seed"],
                max_samples=config["train_samples_per_dataset"],
                min_samples=config["min_train_samples_per_dataset"],
            )
            stage_datasets.append(dataset)
            logger.info(
                "Client %s ready | dataset=%s | train_samples=%s",
                client_index,
                DATASET_DISPLAY_NAMES[dataset_name],
                len(dataset),
            )

        # Build one dataloader per client after the stage datasets are ready.
        stage_dataloaders = [
            create_standard_dataloader(
                dataset=dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=config["pin_memory"],
                shuffle=config["dataloader_shuffle"],
            )
            for dataset in stage_datasets
        ]

        # Run the configured number of communication rounds for this stage.
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
                # Stage round 1 refreshes prototypes from the current dataset; later rounds reuse local memory.
                if stage_round == 1:
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

            # Store the full round state so plots, JSON logs, and later analysis stay in sync.
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
            training_history["client_results"].append(
                {
                    "stage": stage_number,
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
                "Round %s datasets complete | %s | client_prototypes=%s",
                global_round_idx,
                ", ".join(DATASET_DISPLAY_NAMES[name] for name in stage_dataset_names),
                client_prototype_counts,
            )

            save_checkpoint(
                checkpoint_dir=output_dirs["checkpoints"],
                round_idx=global_round_idx,
                base_model=base_model,
                proto_bank=proto_bank,
                training_history=training_history,
                config=config,
                is_final=False,
                include_training_history=False,
            )
            save_history(training_history, output_dirs["metrics"])
            plot_training_history(
                training_history,
                output_dirs["plots"],
                prefix="main",
            )

        # Release only the stage-local dataloaders and datasets before moving to the next pair.
        logger.info("Stage %s complete | releasing stage dataloaders and cached tensors", stage_number)
        del stage_dataloaders
        del stage_datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    logger.info(
        "Training complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
    )


if __name__ == "__main__":
    main()
