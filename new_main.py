"""Sequential federated training entrypoint for the 8-client, 24-dataset run.

The baseline math from ``main.py`` stays the same:
1. Build one shared ViT-MAE backbone with trainable adapters.
2. Assign one client to each GPU.
3. Feed every client one dataset at a time for three sequential stages.
4. Run local MAE + GPAD updates on each client.
5. Aggregate adapter weights and local prototypes on the server.
6. Save round-level checkpoints, metrics, and plots.

This entrypoint changes the data schedule only. Each stage uses eight datasets
in parallel, one per client. Every training split is deterministically fitted
to the same number of samples so all clients run for the same number of steps
per round without ImageNet-style normalization.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from main import (
    CONFIG,
    average_client_metric,
    build_base_model,
    build_gpad_loss,
    convert_to_rgb,
    extract_trainable_state_dict,
    initialize_history,
    plot_training_history,
    prepare_output_dirs,
    print_round_summary,
    resolve_runtime_config,
    save_checkpoint,
    save_history,
    set_random_seed,
    state_dict_num_bytes,
    tensor_num_bytes,
)
from src.client import ClientManager
from src.server import FederatedModelServer, GlobalPrototypeBank, run_server_round

logger = logging.getLogger("PODFCSSV_NewMain")

MULTI_DATASET_CONFIG: Dict[str, Any] = {
    **CONFIG,
    "num_clients": 8,
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
    "save_dir": "multidataset_outputs_8client",
}

CLIENT_DATASET_SEQUENCE: Dict[int, List[str]] = {
    0: ["eurosat", "gtsrb", "stl10"],
    1: ["pcam", "svhn", "lfwpeople"],
    2: ["fer2013", "stanfordcars", "cifar10"],
    3: ["fgvcaircraft", "country211", "fashionmnist"],
    4: ["dtd", "caltech101", "renderedsst2"],
    5: ["oxfordiiitpet", "caltech256", "usps"],
    6: ["flowers102", "sun397", "emnistletters"],
    7: ["food101", "cifar100", "omniglot"],
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


def validate_dataset_schedule(config: Dict[str, Any]) -> None:
    """Check that the client schedule matches the requested run shape."""
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
        raise ValueError(
            "Each dataset should appear only once in the 8-client schedule. "
            f"Duplicate entries: {duplicate_names}"
        )

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
    """Return the number of sequential stages in the current client schedule."""
    return len(next(iter(CLIENT_DATASET_SEQUENCE.values())))


def get_stage_dataset_names(stage_index: int) -> List[str]:
    """Return the datasets assigned to one stage in client order."""
    return [
        CLIENT_DATASET_SEQUENCE[client_index][stage_index]
        for client_index in sorted(CLIENT_DATASET_SEQUENCE)
    ]


def build_dataset_order_by_stage(
    dataset_sequence: Optional[Dict[Any, Sequence[str]]] = None,
) -> List[str]:
    """Flatten a client schedule into the stage order used across the run."""
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
    """Create a reproducible integer seed from the base seed and text parts."""
    joined = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_deterministic_split_indices(
    dataset_size: int,
    seed: int,
    train_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Create a deterministic train/test split for datasets without built-in splits."""
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
    """Fit a split to a fixed sample budget with deterministic sub/over-sampling."""
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
    """Resize each image for ViT-MAE without applying dataset normalization."""
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


def load_named_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    train: bool,
    seed: int,
    max_samples: Optional[int] = None,
    min_samples: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """Load one named dataset and optionally cap it to a fixed sample budget."""
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if dataset_name == "eurosat":
        dataset = build_split_subset(
            datasets.EuroSAT(root=str(root), download=True, transform=transform),
            dataset_name=dataset_name,
            train=train,
            seed=seed,
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
            download=True,
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
    """Mirror the training-time embedding pooling for later linear evaluation."""
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
    """Extract normalized embeddings and labels from a dataset split."""
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
    """Train a linear probe on frozen features and return test accuracy."""
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
    """Evaluate frozen representations on a list of datasets with linear probing."""
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
        test_dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=False,
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
    """Run the 8-client sequential continual-learning schedule end to end."""
    config = dict(MULTI_DATASET_CONFIG)
    resolve_runtime_config(config)
    validate_dataset_schedule(config)

    if config["gpu_count"] not in (0, config["num_clients"]):
        raise ValueError(
            "This entrypoint expects either CPU execution or exactly one GPU per client. "
            f"Detected {config['gpu_count']} GPUs for {config['num_clients']} clients."
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

    logger.info(
        "Starting 8-client sequential run | clients=%s | stages=%s | rounds_per_stage=%s | total_rounds=%s | train_budget=%s | device=%s | output=%s",
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

        for client in client_manager.clients:
            client.reset_local_memory()

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
                prefix="new_main",
            )

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
