"""
Federated Continual Self-Supervised Learning (FCSSL) - Main Orchestrator

This module serves as the primary entry point for simulating a highly realistic,
non-IID federated continual learning environment. The framework integrates several
state-of-the-art paradigms:
    1. Federated Learning (FL): Distributed training without data sharing.
    2. Continual Learning (CL): Sequential task learning without catastrophic forgetting.
    3. Self-Supervised Learning (SSL): Masked Autoencoders (MAE) for representations.
    4. Parameter-Efficient Fine-Tuning (PEFT): Information Bottleneck Adapters.

Theoretical Foundation
----------------------
Traditional FL struggles with statistical heterogeneity (non-IID data) and concept
drift over time. This framework addresses these challenges via:
    - Dirichlet Data Partitioning: Simulates heavy label skew across edge devices.
    - GPAD (Gated Prototype Anchored Distillation): A novel regularization term that
      anchors local representations to a Global Prototype Bank, selectively penalizing
      drift for known concepts while allowing free representation learning for novel
      out-of-distribution (OOD) data.
    - Partial Aggregation: Only the lightweight Adapter modules (~1% parameters) are
      aggregated via FedAvg, strictly preserving the dense semantic priors frozen inside
      the pretrained ViT-MAE backbone.

Execution Flow
--------------
    [Phase 1] Environment & Configuration Verification
    [Phase 2] Component Initialization (Servers, Clients, Models)
    [Phase 3] Non-IID Dirichlet Scheduling (Task Generation)
    [Phase 4] Federated Optimization Loop
              ├─ Broadcast Global Protos (Server -> Client)
              ├─ Local SSL + GPAD Optimization (Client GPUs)
              ├─ Local Prototype Routing & Clustering (Client)
              └─ FedAvg & Prototype Merging (Client -> Server)
    [Phase 5] Global Checkpointing & Evaluation

Author: AI Engineering & Research Team
Target Publication: Q1 High-Impact AI/ML Venues
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from transformers import ViTMAEForPreTraining

from src.server import GlobalPrototypeBank, FederatedModelServer, run_server_round
from src.client import ClientManager
from src.loss import GPADLoss
from src.mae_with_adapter import inject_adapters

# ==============================================================================
# 0. Global Logging Infrastructure
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)-12s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FCSSL_Orchestrator")


# ==============================================================================
# 1. Centralized Hyperparameter Registry (CONFIG)
# ==============================================================================
# This dictionary contains all tunable hyperparameters for the entire ecosystem.
# Centralizing parameters prevents decoupled logic and drastically simplifies
# hyperparameter optimization (HPO) for journal rebuttals.

CONFIG = {
    # ── System Runtime Parameters ─────────────────────────────────────────────
    # Deterministic control ensures fully reproducible Dirichlet data splits
    # and network initializations.
    "seed": 42,
    
    # Total participating nodes in the federated network.
    "num_clients": 2,
    
    # ── Task & Data Heterogeneity (Dirichlet Shift) ───────────────────────────
    # Number of distinct sequential tasks (communication rounds).
    "num_rounds": 5,
    
    # Number of new visual concepts (classes) introduced globally per round.
    "classes_per_task": 40,
    
    # Base expected classes per client. Used to compute the probability of a
    # client receiving any samples from a given class (controls matrix sparsity).
    "classes_per_client_base": 20,
    
    # Concentration parameter (α) for the Dirichlet distribution.
    # Defines the degree of non-IID label skew. As α → 0, data becomes heavily
    # imbalanced (each client gets only a few classes). As α → ∞, data forms
    # an IID uniform distribution. Standard for Q1 benchmarks is 0.5.
    "dirichlet_alpha": 0.5,

    # ── Client Optimization Hyperparameters ───────────────────────────────────
    # Local optimization steps. Minimizing this (e.g., 1 epoch) reduces
    # client-drift in feature space, a known critical issue in non-IID FL.
    "local_epochs": 1,
    
    # Batch size for local SGD updates.
    "batch_size": 64,
    
    # Learning rate tailored for the lightweight adapters. Usually 10x larger
    # than backbone LRs because adapters learn rapidly from zero-init.
    "client_lr": 1e-4,
    
    # L2 regularization factor to bound adapter weight norms.
    "client_weight_decay": 0.05,

    # ── Hardware & Scaling Specifications ─────────────────────────────────────
    # Auto-detected runtime hardware maps.
    "gpu_count": 0,
    "device": "cpu",
    
    # Float precision. Allows seamless scaling to float16 for massive ViTs.
    "dtype": torch.float32,
    "num_workers": 4,
    "pin_memory": True,
    "dataloader_shuffle": True,

    # ── Model Architecture (PEFT Backbone) ────────────────────────────────────
    "pretrained_model_name": "facebook/vit-mae-base",
    "embedding_dim": 768,
    "image_size": 224,
    
    # Dimensionality of the low-rank adapter projections. Small bottlenecks
    # act as a strong structural prior against overfitting.
    "adapter_bottleneck_dim": 256,

    # ── Server Aggregation & Global Momentum ──────────────────────────────────
    # Sim threshold required for the server to Merge an external prototype 
    # into an existing cluster rather than Adding it as a novel concept.
    "merge_threshold": 0.85,
    
    # Smoothing parameters for historical prototype and weight trajectory.
    "server_ema_alpha": 0.1,
    "server_model_ema_alpha": 0.1,
    "max_global_prototypes": 500,

    # ── GPAD (Gated Prototype Anchored Distillation) ──────────────────────────
    # Tau defines the rigid cosine boundary for deciding if a feature vector
    # matches a global concept.
    "gpad_base_tau": 0.8,
    
    # Temperature (T) controls the steepness of the continuous sigmoid gate.
    "gpad_temp_gate": 0.2,
    
    # Controls how harshly the threshold adapts based on assignment entropy.
    "gpad_lambda_entropy": 0.3,
    
    # Defines the sharpness of the probability mass applied to global targets.
    "gpad_soft_assign_temp": 0.07,
    "gpad_epsilon": 1e-8,
    
    # Scalar weighting λ multiplying the GPAD loss backward signal.
    "lambda_proto": 0.01,

    # ── Client Local Novelty & Clustering ─────────────────────────────────────
    "k_init_prototypes": 50,
    "client_local_update_threshold": 0.85,
    "client_local_ema_alpha": 0.05,
    "novelty_buffer_size": 256,
    "novelty_k": 10,
    "kmeans_max_iters": 100,
    "kmeans_tol": 1e-4,

    # ── System I/O ────────────────────────────────────────────────────────────
    "data_root": "./data",
    "save_dir": "checkpoints",
}


# ==============================================================================
# 2. Data Synthesis & Dirichlet Mapping
# ==============================================================================

def load_tinyimagenet(data_root: str, image_size: int = 224) -> datasets.ImageFolder:
    """
    Acquires and formats the Tiny ImageNet distribution for ViT-MAE intake.

    Data must be aggressively upscaled from 64x64 native resolution to 224x224
    to satisfy the fixed 16x16 patch projection matrix frozen inside the 
    pretrained ViT-MAE backbone.

    Args:
        data_root (str): Output directory path for the dataset payload.
        image_size (int): Expected spatial dimension of the visual tensors.

    Returns:
        datasets.ImageFolder: A PyTorch accessible dataset container mapping
            raw upscaled tensors to numeric integer labels.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")

    if not os.path.isdir(train_dir):
        logger.info(f"Initiating remote transfer of Tiny ImageNet to: {data_root}")
        _download_tinyimagenet(data_root)

    dataset = datasets.ImageFolder(train_dir, transform=transform)
    logger.info(f"[Tiny ImageNet Loaded] Data Volume: {len(dataset)} | Classes: {len(dataset.classes)}")
    return dataset


def _download_tinyimagenet(data_root: str) -> None:
    """
    Executes a direct binary HTTP transfer of the Tiny ImageNet payload
    from the Stanford university endpoint, unpacking the archive into Memory.
    """
    import urllib.request
    import zipfile

    os.makedirs(data_root, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    logger.info(f"Streaming bytes from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    logger.info(f"Unpacking archive headers into {data_root}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_root)

    os.remove(zip_path)


def create_task_schedule(
    dataset: datasets.ImageFolder,
    num_rounds: int,
    classes_per_task: int,
    num_clients: int,
    classes_per_client_base: int,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> List[List[Dict[int, float]]]:
    """
    Formulates a chronologically sequential, heavily non-IID task distribution
    using the Dirichlet statistical framework (Dir(α)).

    Mathematical Formulation:
    -----------------------
    For each global class c ∈ C_t introduced in task t, we sample a vector
    p_c ~ Dir(α). The component p_{c,k} determines the exact continuous
    proportion of available data for class c allocated to client k. By limiting
    the number of clients k that participate in drawing p_c, we enforce harsh
    structural sparsity mimicking isolated domain streams.

    Args:
        dataset: Target dataset containing global class distributions.
        num_rounds: Equivalent to the number of temporal shifts (tasks).
        classes_per_task: |C_t|, cardinality of the task subspace.
        num_clients: K, the total number of federation nodes.
        classes_per_client_base: Proxy defining the target sparsity ratio.
        dirichlet_alpha: α, governing the entropy of the allocation spread.
        seed: Determinism anchor for strict experimental reproducibility.

    Returns:
        A nested scheduling tensor T of shape [Rounds, Clients] where each
        T[r, k] maps subset class indices uniquely to scalar proportion bounds.
    """
    all_classes = list(range(len(dataset.classes)))
    rng = torch.Generator().manual_seed(seed)
    
    # Establish a randomized trajectory of global concepts avoiding ordering bias
    perm = torch.randperm(len(all_classes), generator=rng).tolist()
    shuffled_classes = [all_classes[i] for i in perm]
    
    np.random.seed(seed)

    schedule = []
    for round_idx in range(num_rounds):
        start = round_idx * classes_per_task
        end = start + classes_per_task
        task_classes = shuffled_classes[start:end]

        round_schedule = [{} for _ in range(num_clients)]
        
        for cls_idx in task_classes:
            # Control structural sparsity: isolate the class to exactly N clients
            client_ratio = max(
                1, 
                min(num_clients, int(num_clients * (classes_per_client_base / classes_per_task)))
            )
            
            # Uniformly draw indices of participating clients without replacement
            selected_clients = np.random.choice(num_clients, client_ratio, replace=False)
            
            # Generative draw over the probability simplex Δ^{K-1}
            proportions = np.random.dirichlet([dirichlet_alpha] * client_ratio)
            
            for c_idx, prop in zip(selected_clients, proportions):
                # Discard trivial density mass to preserve true non-IID hardness
                if prop > 0.01:
                    round_schedule[c_idx][cls_idx] = float(prop)

        schedule.append(round_schedule)

    return schedule


def create_client_dataloaders(
    dataset: datasets.ImageFolder,
    client_class_proportions: List[Dict[int, float]],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> List[DataLoader]:
    """
    Instantiates physical PyTorch generators traversing the Dirichlet partitions.

    This function isolates absolute memory indices corresponding to the raw labels,
    shuffles the global index block to remove structural locality biases, and 
    symmetrically splices the block according to the Dirichlet probability mass p_{c,k} 
    computed in the scheduling orchestrator.

    Args:
        dataset: The massive global memory mapping.
        client_class_proportions: T[r, :] containing exact p_{c,k} assignments.
        seed: Epoch/Round deterministic mutation seed.

    Returns:
        List containing isolated subset generators for local client gradients.
    """
    dataloaders = []
    targets = torch.tensor(dataset.targets)
    client_sample_indices = [[] for _ in range(len(client_class_proportions))]

    # Extract all unique concepts activated globally during this specific interval
    all_task_classes = set()
    for c_dict in client_class_proportions:
        all_task_classes.update(c_dict.keys())

    np.random.seed(seed)

    for cls_idx in all_task_classes:
        # Resolve raw integer address boundaries for the target class
        mask = (targets == cls_idx)
        cls_indices = torch.where(mask)[0].tolist()
        np.random.shuffle(cls_indices)
        
        total_samples = len(cls_indices)
        current_offset = 0
        
        # Sequentially map the shuffled spatial indices purely based on density
        for c_idx, c_dict in enumerate(client_class_proportions):
            if cls_idx in c_dict:
                prop = c_dict[cls_idx]
                num_samples_for_client = int(total_samples * prop)
                
                slice_end = min(current_offset + num_samples_for_client, total_samples)
                client_sample_indices[c_idx].extend(cls_indices[current_offset:slice_end])
                current_offset = slice_end

    # Encapsulate isolated indices inside secure memory-pinned dataloader contexts
    for c_idx, indices in enumerate(client_sample_indices):
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        dataloaders.append(loader)

    return dataloaders


# ==============================================================================
# 3. Model State Archival & Analytics
# ==============================================================================

def save_checkpoint(
    save_dir: str,
    round_idx: int,
    base_model: nn.Module,
    proto_bank: GlobalPrototypeBank,
    training_history: Dict[str, Any],
    is_final: bool = False,
) -> str:
    """
    Flushes all active multi-dimensional state tracking to non-volatile disk.
    
    Critically, only delta parameter matrices (the adapter weights mapping to
    <requires_grad=True>) are flushed, drastically mitigating the exponential 
    storage requirements of classical deep-model FL.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter state dict aggressively for topological footprint compression
    trainable_state = {
        k: v.cpu() for k, v in base_model.state_dict().items()
        if v.requires_grad or "adapter" in k.lower()
    }
    if len(trainable_state) == 0:
        trainable_state = {k: v.cpu() for k, v in base_model.state_dict().items()}

    checkpoint = {
        "round": round_idx,
        "model_state_dict": trainable_state,
        "global_prototypes": (
            proto_bank.prototypes.cpu() if proto_bank.prototypes is not None else None
        ),
        "training_history": training_history,
        "config": {k: str(v) for k, v in CONFIG.items()},
    }

    filename = "final_model.pt" if is_final else f"round_{round_idx}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return filepath


def print_round_summary(
    round_idx: int,
    num_rounds: int,
    losses: List[float],
    proto_bank: GlobalPrototypeBank,
    round_time: float,
    training_history: Dict[str, Any],
) -> None:
    """
    Projects cumulative step-wise metrics into terminal standard out formats.
    """
    bank_size = proto_bank.prototypes.shape[0] if proto_bank.prototypes is not None else 0
    max_bank = CONFIG["max_global_prototypes"]

    print(f"\n{'━' * 60}")
    print(f"  Federated Step [{round_idx}/{num_rounds}] Computation Complete")
    print(f"{'━' * 60}")
    for i, loss in enumerate(losses):
        print(f"  > Edge Node {i} Loss:    {loss:.6f}")
    print(f"  > Aggregate Network Loss: {sum(losses) / len(losses):.6f}")
    print(f"  > Global Semantic Matrix: {bank_size}/{max_bank} Nodes Embedded")
    print(f"  > Communication Time:     {round_time:.1f}s")
    print(f"{'━' * 60}\n")


# ==============================================================================
# 4. Global Architecture Orchestrator (Main Execution Graph)
# ==============================================================================

def main():
    """
    Traverses the topological graph of the entire federated continual learning 
    experiment payload, strictly dictating operations sequences to emulate 
    edge-to-server mathematical barriers. 
    
    Phases:
        [1] CUDA mapping and algorithmic determinism bindings.
        [2] Heavy memory allocations for Base Model and architectural mutations 
            (layer-wise adapter projections).
        [3] Dirichlet dataset generations.
        [4] Network synchronizations handling gradient trajectories and 
            representation bank embeddings.
    """
    logger.info("Initiating High-Fidelity Federated Continual Learning Graph...")

    # ── [Phase 1] Compute Hardware Target Definitions
    if torch.cuda.is_available():
        CONFIG["gpu_count"] = torch.cuda.device_count()
        CONFIG["device"] = "cuda"
        logger.info(f"Targeting active hardware stack: {CONFIG['device']} ({CONFIG['gpu_count']}x Accelerators detected).")
    else:
        CONFIG["device"] = "cpu"

    if CONFIG["seed"] is not None:
        torch.manual_seed(CONFIG["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG["seed"])

    # ── [Phase 2] Component Topologies
    proto_bank = GlobalPrototypeBank(
        embedding_dim=CONFIG["embedding_dim"],
        merge_threshold=CONFIG["merge_threshold"],
        ema_alpha=CONFIG["server_ema_alpha"],
        device=CONFIG["device"],
        max_prototypes=CONFIG["max_global_prototypes"],
    )

    fed_server = FederatedModelServer()

    logger.info("Instantiating generic global architectural foundations...")
    base_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    
    # Mutate standard architecture via bottleneck injections, freezing dense weights
    base_model = inject_adapters(base_model, bottleneck_dim=CONFIG["adapter_bottleneck_dim"])
    base_model = base_model.to(device=CONFIG["device"], dtype=CONFIG["dtype"])

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    logger.info(f"Topological Mutation Success | Param Mass: {total_params:,} | Active Delta Rank: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")

    client_manager = ClientManager(
        base_model=base_model,
        num_clients=CONFIG["num_clients"],
        gpu_count=CONFIG["gpu_count"],
        dtype=CONFIG["dtype"],
        optimizer_kwargs={"lr": CONFIG["client_lr"], "weight_decay": CONFIG["client_weight_decay"]},
        local_update_threshold=CONFIG["client_local_update_threshold"],
        local_ema_alpha=CONFIG["client_local_ema_alpha"],
        lambda_proto=CONFIG["lambda_proto"],
        novelty_buffer_size=CONFIG["novelty_buffer_size"],
        novelty_k=CONFIG["novelty_k"],
        kmeans_max_iters=CONFIG["kmeans_max_iters"],
        kmeans_tol=CONFIG["kmeans_tol"],
    )

    gpad_loss = GPADLoss(
        base_tau=CONFIG["gpad_base_tau"],
        temp_gate=CONFIG["gpad_temp_gate"],
        lambda_entropy=CONFIG["gpad_lambda_entropy"],
        soft_assign_temp=CONFIG["gpad_soft_assign_temp"],
        epsilon=CONFIG["gpad_epsilon"],
    )

    # ── [Phase 3] Spatial Dirichlet Mapping 
    dataset = load_tinyimagenet(CONFIG["data_root"], CONFIG["image_size"])
    task_schedule = create_task_schedule(
        dataset=dataset, num_rounds=CONFIG["num_rounds"], classes_per_task=CONFIG["classes_per_task"],
        num_clients=CONFIG["num_clients"], classes_per_client_base=CONFIG["classes_per_client_base"],
        dirichlet_alpha=CONFIG["dirichlet_alpha"], seed=CONFIG["seed"],
    )

    for r, round_sched in enumerate(task_schedule):
        for c, class_props in enumerate(round_sched):
            logger.info(f"Target Subspace [Shift {r+1} | Node {c}]: {len(class_props.keys())} dimensional categories loaded via Dirichlet(α).")

    # ── [Phase 4] Dynamic Synchronous Federation Loop
    training_history = {
        "round_losses": [], "avg_losses": [], "proto_bank_sizes": [],
        "round_times": [], "task_classes": []
    }
    global_protos = None

    for round_idx in range(1, CONFIG["num_rounds"] + 1):
        round_start = time.time()
        logger.info(f"\n{'='*40}\nEXECUTING NON-STATIONARY SHIFT ROUTINE {round_idx}/{CONFIG['num_rounds']}\n{'='*40}")

        # Broadcast State Vectors
        if round_idx > 1:
            logger.info(f"Pushing rank-{len(global_protos)} memory payload globally.")

        client_class_proportions = task_schedule[round_idx - 1]
        dataloaders = create_client_dataloaders(
            dataset=dataset, client_class_proportions=client_class_proportions,
            batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"], shuffle=CONFIG["dataloader_shuffle"],
            seed=CONFIG["seed"] + round_idx,
        )

        # SGD Opt + Local Novelty Discovery via MAE and GPAD Gradients
        losses = client_manager.train_round(dataloaders, global_prototypes=global_protos, gpad_loss_fn=gpad_loss)
        
        client_payloads = []
        if round_idx == 1:
            # Baseline zero-shot clustering formulation for absolute novel manifolds
            for i, client in enumerate(client_manager.clients):
                local_protos = client.generate_prototypes(dataloaders[i], K_init=CONFIG["k_init_prototypes"])
                weights = {k: v.cpu() for k, v in client.model.state_dict().items() if client.model.get_parameter(k).requires_grad}
                client_payloads.append({"client_id": f"client_{i}", "protos": local_protos.cpu(), "weights": weights})
        else:
            # Iterative non-stationary extraction tracking dynamic parameter shift
            for i, client in enumerate(client_manager.clients):
                local_protos = client.get_local_prototypes()
                weights = {k: v.cpu() for k, v in client.model.state_dict().items() if client.model.get_parameter(k).requires_grad}
                payload = {"client_id": f"client_{i}", "weights": weights}
                if local_protos is not None:
                    payload["protos"] = local_protos.cpu()
                client_payloads.append(payload)

        # Apply Global Momentum bounds & Vector Algebra (FedAvg + EMA Prototype Merges)
        global_protos, global_weights = run_server_round(
            proto_manager=proto_bank, model_server=fed_server,
            client_payloads=client_payloads, current_global_weights=base_model.state_dict(),
            round_idx=round_idx, server_model_ema_alpha=CONFIG["server_model_ema_alpha"],
        )
        base_model.load_state_dict(global_weights, strict=False)

        round_time = time.time() - round_start
        training_history["round_losses"].append(losses)
        training_history["avg_losses"].append(sum(losses) / len(losses))
        training_history["proto_bank_sizes"].append(global_protos.shape[0] if global_protos is not None else 0)
        training_history["round_times"].append(round_time)
        training_history["task_classes"].append([list(cp.keys()) for cp in client_class_proportions])

        print_round_summary(round_idx, CONFIG["num_rounds"], losses, proto_bank, round_time, training_history)
        save_checkpoint(CONFIG["save_dir"], round_idx, base_model, proto_bank, training_history)

    # ── [Phase 5] Termination & Output
    save_checkpoint(CONFIG["save_dir"], CONFIG["num_rounds"], base_model, proto_bank, training_history, is_final=True)
    
    total_time = sum(training_history["round_times"])
    logger.info(f"Network Convergence Secured | Total Active State Time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
