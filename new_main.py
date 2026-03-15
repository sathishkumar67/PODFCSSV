"""Sequential 10-dataset continual training entrypoint.

This script keeps the same federated-learning math as ``main.py`` but replaces
the Tiny ImageNet schedule with a deliberately diverse two-client benchmark.

The continual schedule is:
1. There are exactly two clients.
2. Each client owns five datasets.
3. Every client finishes the configured rounds on its current dataset before
   moving to the next dataset in its sequence.
4. The global adapter weights, local prototypes, novelty buffers, and global
   prototype bank persist across dataset boundaries.

The dataset mix spans several domains so the experiment showcases robustness
under strong distribution shifts:
1. Satellite imagery.
2. Medical histopathology.
3. Country and geography recognition.
4. Fine-grained aircraft recognition.
5. Texture recognition.
6. Pet breeds.
7. Flower species.
8. Food categories.
9. Traffic signs.
10. Street-view house numbers.

The script also adds publication-oriented reporting:
1. Per-round training metrics and communication tracking.
2. Per-stage linear-probe evaluation on every seen dataset.
3. Final accuracy and forgetting summaries.
4. Plots saved to disk for later inclusion in the paper.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from main import (
    CONFIG,
    VIT_MAE_IMAGE_MEAN,
    VIT_MAE_IMAGE_STD,
    average_client_metric,
    build_base_model,
    build_gpad_loss,
    convert_to_rgb,
    extract_trainable_state_dict,
    initialize_history,
    plot_training_history,
    prepare_output_dirs,
    resolve_runtime_config,
    save_checkpoint,
    save_history,
    serialize_config,
    set_random_seed,
    state_dict_num_bytes,
    tensor_num_bytes,
)
from src.client import ClientManager
from src.server import FederatedModelServer, GlobalPrototypeBank, run_server_round

logger = logging.getLogger("PODFCSSV_NewMain")

MULTI_DATASET_CONFIG: Dict[str, Any] = {
    **CONFIG,
    "num_clients": 2,
    "rounds_per_dataset": 3,
    "linear_eval_batch_size": 256,
    "linear_eval_epochs": 5,
    "linear_eval_lr": 1e-2,
    "linear_eval_weight_decay": 1e-4,
    "linear_eval_num_workers": 2,
    "save_dir": "multidataset_outputs",
}

CLIENT_DATASET_SEQUENCE: Dict[int, List[str]] = {
    0: ["eurosat", "pcam", "country211", "fgvcaircraft", "dtd"],
    1: ["oxfordiiitpet", "flowers102", "food101", "gtsrb", "svhn"],
}

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "eurosat": "EuroSAT",
    "pcam": "PCAM",
    "country211": "Country211",
    "fgvcaircraft": "FGVC Aircraft",
    "dtd": "DTD",
    "oxfordiiitpet": "Oxford-IIIT Pet",
    "svhn": "SVHN",
    "flowers102": "Flowers102",
    "food101": "Food101",
    "gtsrb": "GTSRB",
}


def build_dataset_transform(image_size: int) -> transforms.Compose:
    """Build the ViT-MAE-compatible transform used for every sequential dataset.

    The transform mirrors the preprocessing used by the baseline entrypoint:
    1. Convert every image to RGB.
    2. Resize to the configured model input size.
    3. Convert to a tensor in ``[0, 1]``.
    4. Normalize with the ImageNet mean and standard deviation expected by the
       pre-trained MAE backbone.
    """
    return transforms.Compose(
        [
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MAE_IMAGE_MEAN, std=VIT_MAE_IMAGE_STD),
        ]
    )


def load_named_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int,
    train: bool,
    seed: int,
) -> torch.utils.data.Dataset:
    """Load one of the ten sequential datasets.

    All datasets come from ``torchvision`` and were chosen to create strong
    cross-domain shifts relative to the Tiny ImageNet baseline while covering
    satellite, medical, geographic, fine-grained, texture, and traffic domains.
    """
    transform = build_dataset_transform(image_size)
    root = Path(data_root) / "multidataset" / dataset_name

    if dataset_name == "eurosat":
        full_dataset = datasets.EuroSAT(
            root=str(root),
            download=True,
            transform=transform,
        )
        train_indices, test_indices = build_deterministic_split_indices(
            dataset_size=len(full_dataset),
            seed=seed,
            train_fraction=0.8,
        )
        return Subset(full_dataset, train_indices if train else test_indices)
    if dataset_name == "pcam":
        return datasets.PCAM(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "country211":
        return datasets.Country211(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "fgvcaircraft":
        return datasets.FGVCAircraft(
            root=str(root),
            split="trainval" if train else "test",
            annotation_level="variant",
            download=True,
            transform=transform,
        )
    if dataset_name == "dtd":
        return datasets.DTD(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "oxfordiiitpet":
        return datasets.OxfordIIITPet(
            root=str(root),
            split="trainval" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "svhn":
        return datasets.SVHN(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "flowers102":
        return datasets.Flowers102(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "food101":
        return datasets.Food101(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )
    if dataset_name == "gtsrb":
        return datasets.GTSRB(
            root=str(root),
            split="train" if train else "test",
            download=True,
            transform=transform,
        )

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def build_deterministic_split_indices(
    dataset_size: int,
    seed: int,
    train_fraction: float,
) -> Tuple[List[int], List[int]]:
    """Split a dataset deterministically when torchvision provides no split.

    EuroSAT ships as a single image folder in torchvision, so the code creates
    a reproducible train/test partition using the repository seed.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_size = int(dataset_size * train_fraction)
    return indices[:train_size], indices[train_size:]


def create_standard_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    """Create a standard dataloader with the repository's preferred settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def pool_model_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply the same embedding pooling rule used by the training clients."""
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
    """Extract normalized embeddings and labels for linear evaluation."""
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
    """Map arbitrary dataset labels to ``0..num_classes-1``."""
    unique_labels = torch.unique(torch.cat([train_labels, test_labels], dim=0)).tolist()
    label_mapping = {original_label: new_index for new_index, original_label in enumerate(unique_labels)}

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
    """Train a linear classifier on frozen embeddings and return test accuracy."""
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


def evaluate_seen_datasets(
    model: nn.Module,
    seen_dataset_names: List[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run linear-probe evaluation on every dataset seen so far."""
    results: Dict[str, float] = {}
    for dataset_name in seen_dataset_names:
        train_dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
        )
        test_dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=False,
            seed=config["seed"],
        )

        train_loader = create_standard_dataloader(
            train_dataset,
            batch_size=config["linear_eval_batch_size"],
            num_workers=config["linear_eval_num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=False,
        )
        test_loader = create_standard_dataloader(
            test_dataset,
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


def compute_forgetting(
    stage_evaluations: List[Dict[str, Any]],
    dataset_first_seen_stage: Dict[str, int],
) -> Tuple[Dict[str, float], float]:
    """Compute per-dataset and average forgetting from stage evaluations."""
    forgetting: Dict[str, float] = {}
    if not stage_evaluations:
        return forgetting, 0.0

    for dataset_name, first_stage in dataset_first_seen_stage.items():
        accuracies = [
            stage_result["accuracies"][dataset_name]
            for stage_result in stage_evaluations
            if stage_result["stage"] >= first_stage
            and dataset_name in stage_result["accuracies"]
        ]
        if accuracies:
            forgetting[dataset_name] = max(accuracies) - accuracies[-1]

    average_forgetting = (
        sum(forgetting.values()) / len(forgetting) if forgetting else 0.0
    )
    return forgetting, average_forgetting


def save_evaluation_history(
    evaluation_history: Dict[str, Any],
    metrics_dir: Path,
) -> Path:
    """Save stage-evaluation results as JSON."""
    evaluation_path = metrics_dir / "evaluation_history.json"
    with evaluation_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_history, handle, indent=2)
    return evaluation_path


def plot_accuracy_heatmap(
    evaluation_history: Dict[str, Any],
    plots_dir: Path,
) -> Path:
    """Plot a stage-by-dataset heatmap of linear-probe accuracy."""
    stage_results = evaluation_history.get("stage_results", [])
    dataset_names = evaluation_history.get("dataset_order", [])
    if not stage_results or not dataset_names:
        return plots_dir / "new_main_accuracy_heatmap.png"

    matrix = np.full((len(stage_results), len(dataset_names)), np.nan, dtype=np.float32)
    for stage_index, stage_result in enumerate(stage_results):
        for dataset_index, dataset_name in enumerate(dataset_names):
            if dataset_name in stage_result["accuracies"]:
                matrix[stage_index, dataset_index] = stage_result["accuracies"][dataset_name]

    figure, axis = plt.subplots(figsize=(max(10, len(dataset_names) * 1.1), 6))
    masked_matrix = np.ma.masked_invalid(matrix)
    heatmap = axis.imshow(masked_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_title("Linear-Probe Accuracy Across Stages")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Stage")
    axis.set_xticks(range(len(dataset_names)))
    axis.set_xticklabels(
        [DATASET_DISPLAY_NAMES[name] for name in dataset_names],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(stage_results)))
    axis.set_yticklabels([f"Stage {stage_result['stage']}" for stage_result in stage_results])
    figure.colorbar(heatmap, ax=axis, label="Accuracy")
    figure.tight_layout()

    figure_path = plots_dir / "new_main_accuracy_heatmap.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def plot_final_accuracy_bar(
    evaluation_history: Dict[str, Any],
    plots_dir: Path,
) -> Path:
    """Plot final linear-probe accuracy for each dataset."""
    final_accuracy = evaluation_history.get("final_accuracy", {})
    if not final_accuracy:
        return plots_dir / "new_main_final_accuracy.png"

    dataset_names = list(final_accuracy.keys())
    accuracies = [final_accuracy[name] for name in dataset_names]

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [DATASET_DISPLAY_NAMES[name] for name in dataset_names],
        accuracies,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Final Linear-Probe Accuracy")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Accuracy")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    figure_path = plots_dir / "new_main_final_accuracy.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def plot_forgetting_bar(
    evaluation_history: Dict[str, Any],
    plots_dir: Path,
) -> Path:
    """Plot forgetting for each dataset."""
    forgetting = evaluation_history.get("forgetting", {})
    if not forgetting:
        return plots_dir / "new_main_forgetting.png"

    dataset_names = list(forgetting.keys())
    forgetting_values = [forgetting[name] for name in dataset_names]

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [DATASET_DISPLAY_NAMES[name] for name in dataset_names],
        forgetting_values,
    )
    axis.set_title("Forgetting After Sequential Multi-Dataset Training")
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Forgetting")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    figure_path = plots_dir / "new_main_forgetting.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def initialize_evaluation_history(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the history container used for stage evaluation."""
    return {
        "config": serialize_config(config),
        "dataset_order": [
            dataset_name
            for stage_pair in zip(
                CLIENT_DATASET_SEQUENCE[0],
                CLIENT_DATASET_SEQUENCE[1],
            )
            for dataset_name in stage_pair
        ],
        "stage_results": [],
        "dataset_first_seen_stage": {},
        "final_accuracy": {},
        "forgetting": {},
        "average_forgetting": 0.0,
    }


def main() -> None:
    """Run the 10-dataset sequential continual-learning experiment."""
    config = dict(MULTI_DATASET_CONFIG)
    resolve_runtime_config(config)
    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])

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
    evaluation_history = initialize_evaluation_history(config)
    current_global_weights = extract_trainable_state_dict(base_model)
    global_prototypes: torch.Tensor | None = None
    global_round_idx = 0

    num_stages = len(CLIENT_DATASET_SEQUENCE[0])
    for stage_idx in range(num_stages):
        stage_number = stage_idx + 1
        stage_dataset_names = [
            CLIENT_DATASET_SEQUENCE[0][stage_idx],
            CLIENT_DATASET_SEQUENCE[1][stage_idx],
        ]
        logger.info(
            "Starting stage %s/%s | client_0=%s | client_1=%s",
            stage_number,
            num_stages,
            DATASET_DISPLAY_NAMES[stage_dataset_names[0]],
            DATASET_DISPLAY_NAMES[stage_dataset_names[1]],
        )

        stage_datasets = [
            load_named_dataset(
                dataset_name=dataset_name,
                data_root=config["data_root"],
                image_size=config["image_size"],
                train=True,
                seed=config["seed"],
            )
            for dataset_name in stage_dataset_names
        ]
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

        for dataset_name in stage_dataset_names:
            evaluation_history["dataset_first_seen_stage"].setdefault(
                dataset_name,
                stage_number,
            )

        for stage_round in range(1, config["rounds_per_dataset"] + 1):
            global_round_idx += 1
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
                if global_round_idx == 1:
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

            save_checkpoint(
                checkpoint_dir=output_dirs["checkpoints"],
                round_idx=global_round_idx,
                base_model=base_model,
                proto_bank=proto_bank,
                training_history=training_history,
                config=config,
                is_final=False,
            )
            save_history(training_history, output_dirs["metrics"])
            plot_training_history(
                training_history,
                output_dirs["plots"],
                prefix="new_main",
            )

        seen_dataset_names = []
        for client_index in range(config["num_clients"]):
            seen_dataset_names.extend(CLIENT_DATASET_SEQUENCE[client_index][:stage_number])
        seen_dataset_names = list(dict.fromkeys(seen_dataset_names))

        stage_accuracies = evaluate_seen_datasets(
            model=base_model,
            seen_dataset_names=seen_dataset_names,
            config=config,
        )
        average_accuracy = (
            sum(stage_accuracies.values()) / len(stage_accuracies)
            if stage_accuracies
            else 0.0
        )
        evaluation_history["stage_results"].append(
            {
                "stage": stage_number,
                "dataset_names": stage_dataset_names,
                "accuracies": stage_accuracies,
                "average_accuracy": average_accuracy,
            }
        )

        forgetting, average_forgetting = compute_forgetting(
            stage_evaluations=evaluation_history["stage_results"],
            dataset_first_seen_stage=evaluation_history["dataset_first_seen_stage"],
        )
        evaluation_history["final_accuracy"] = stage_accuracies
        evaluation_history["forgetting"] = forgetting
        evaluation_history["average_forgetting"] = average_forgetting

        save_evaluation_history(evaluation_history, output_dirs["metrics"])
        plot_accuracy_heatmap(evaluation_history, output_dirs["plots"])
        plot_final_accuracy_bar(evaluation_history, output_dirs["plots"])
        plot_forgetting_bar(evaluation_history, output_dirs["plots"])

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=base_model,
        proto_bank=proto_bank,
        training_history=training_history,
        config=config,
        is_final=True,
    )


if __name__ == "__main__":
    main()
