"""
Federated Continual Self-Supervised Learning — Main Orchestrator.

This script is the top-level entry point that executes the complete
lifecycle of a Federated Learning (FL) system designed for Continual
Self-Supervised Learning on the Tiny ImageNet dataset. It combines a
pretrained ViT-MAE backbone with IBA adapters as the model architecture,
MAE as the self-supervised pretext task, and Gated Prototype Anchored
Distillation (GPAD) as a forgetting-prevention mechanism.

System Architecture
-------------------
The system consists of three logical actors:

1. **Server** (central coordinator):
   - Maintains a Global Prototype Bank — a dynamically growing collection
     of L2-normalized prototype vectors on the unit hypersphere, where each
     vector represents a distinct visual concept discovered across the
     federation.
   - Aggregates client model weights via Federated Averaging (FedAvg),
     computing the element-wise arithmetic mean of all client state dicts.

2. **Clients** (edge devices):
   - Each client holds a private local dataset and an independent deep copy
     of the global model (ViTMAEForPreTraining with IBA adapters).
   - Privacy-preserving: clients NEVER share raw data — only compact
     prototype vectors and model weights are communicated over the network.
   - Training uses per-embedding routing to classify each feature vector as
     either "anchored" (known concept → GPAD loss) or "non-anchored"
     (novel concept → local prototype update or novelty buffer).

3. **Orchestrator** (this script):
   - Manages the round-based communication loop between server and clients.
   - Handles broadcasting, collection, and state updates.
   - Centrally defines ALL hyperparameters in the CONFIG dictionary.

Model Pipeline
--------------
The model is a real ``ViTMAEForPreTraining`` loaded from HuggingFace with
IBA adapters injected into every encoder layer via ``inject_adapters()``.
The backbone is frozen — only adapter parameters (~1% of total) are
trained. The model expects image tensors of shape ``[B, 3, 224, 224]``.

Continual Learning Design
--------------------------
Tiny ImageNet has 200 classes. We split them into 5 sequential tasks of
40 classes each. Each round introduces a new task. Within each task, the
40 classes are divided between the 2 clients (20 each, non-IID), simulating
realistic federated continual learning where:
- Different clients see different data distributions (non-IID).
- New visual concepts arrive over time (continual).
- Raw data never leaves the client (federated).

Hyperparameter Centralization
-----------------------------
Every tunable value across the entire pipeline is centralized in the
``CONFIG`` dictionary defined at the module level. This avoids magic numbers
scattered across source files and enables straightforward hyperparameter
sweeps. Each entry includes a descriptive comment and valid range for tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import logging
import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# ==========================================================================
# Project Imports
# ==========================================================================
# Import the project's server, client, and loss modules. These contain the
# core FL components: prototype bank, FedAvg server, client manager, and
# the GPAD distillation loss.
from src.server import GlobalPrototypeBank, FederatedModelServer, run_server_round, GlobalModel
from src.client import ClientManager
from src.loss import GPADLoss

# Import the pretrained ViT-MAE backbone and adapter injection utility.
# ViTMAEForPreTraining provides the full Masked Autoencoder with encoder +
# decoder. inject_adapters() freezes the backbone and adds ~1% trainable
# IBA adapter parameters.
from transformers import ViTMAEForPreTraining
from src.mae_with_adapter import inject_adapters

# Torchvision for dataset loading and image transforms.
from torchvision import transforms, datasets


# ==========================================================================
# Logging Configuration
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DistillFed")


# ==========================================================================
# HYPERPARAMETERS & CONFIGURATION
#
# Every tunable value across the entire pipeline is centralized here.
# This avoids magic numbers scattered in source files and makes
# hyperparameter sweeps straightforward.
#
# Each entry includes:
#   - A descriptive comment explaining the parameter's role.
#   - The valid range (as a comment) for future hyperparameter tuning.
#   - The default value chosen based on initial experimentation.
# ==========================================================================

CONFIG = {
    # ── System ────────────────────────────────────────────────────────────────
    # Random seed for reproducibility (set to None to disable)
    "seed": 42,

    # How many federated clients to simulate in this run
    "num_clients": 2,

    # Total number of server-client communication rounds to simulate.
    # Each round corresponds to one continual learning task.
    # 5 rounds × 40 classes per task = 200 classes (all of Tiny ImageNet).
    "num_rounds": 5,

    # Number of local training epochs each client runs per round.
    # In real federated settings, each client trains for 1 epoch to minimize
    # communication overhead and client drift.
    # Range: 1–10
    "local_epochs": 1,

    # Number of GPUs available (0 = CPU-only sequential mode)
    # This value is auto-detected at runtime if CUDA is available
    "gpu_count": 0,

    # Primary computation device for training and inference.
    # Set to "cuda" to use the first available GPU, or "cuda:N" for a
    # specific GPU. Auto-updated at runtime when CUDA is detected.
    "device": "cpu",

    # Floating-point precision used for model parameters and input tensors.
    # torch.float32 (single precision) is the safe default for T4 GPUs.
    "dtype": torch.float32,

    # ── Model & Data ─────────────────────────────────────────────────────────
    # Pretrained model name/path for HuggingFace ViT-MAE backbone
    "pretrained_model_name": "facebook/vit-mae-base",

    # Dimensionality of the feature embedding space.
    # ViT-Base hidden size is 768. This must match the pretrained model.
    "embedding_dim": 768,

    # Input image size expected by the ViT-MAE model.
    # ViT-MAE-Base uses 224×224 images (16×16 patches → 196 patch tokens).
    "image_size": 224,

    # Mini-batch size for local training and prototype extraction.
    # 16 is a good balance for T4 GPUs (16GB VRAM) with float32.
    "batch_size": 64,

    # Whether to shuffle the DataLoader between epochs
    "dataloader_shuffle": True,

    # Number of DataLoader worker processes for parallel data loading.
    "num_workers": 4,

    # Enable pinned memory for faster CPU→GPU data transfer.
    "pin_memory": True,

    # ── Continual Learning Task Schedule ──────────────────────────────────────
    # Number of classes introduced per round (task).
    # 40 classes × 5 rounds = 200 (all of Tiny ImageNet).
    "classes_per_task": 40,

    # Number of classes assigned to each client per task (non-IID split).
    # 20 classes × 2 clients = 40 classes per task.
    "classes_per_client": 20,

    # Root directory for the Tiny ImageNet dataset.
    # The dataset will be downloaded here on first run.
    "data_root": "./data",

    # ── Checkpointing ────────────────────────────────────────────────────────
    # Directory to save model checkpoints and training history.
    "save_dir": "checkpoints",

    # ── Adapter (mae_with_adapter.py) ────────────────────────────────────────
    # Bottleneck dimension of the IBA adapters injected into the ViT encoder.
    # Smaller = fewer params / faster, larger = more capacity.
    # Typical values: 32–128. At dim=64 with ViT-Base the adapters add ~1 %
    # trainable parameters.
    "adapter_bottleneck_dim": 256,

    # Dropout rate for IBA adapters (regularization)
    # Range: 0.0–0.5
    "adapter_dropout": 0.1,

    # ── Global Prototype Management (server.py) ──────────────────────────────
    # Server-side global merge threshold: cosine similarity required to merge
    # a local prototype into an existing global one via EMA.
    #
    # IMPORTANT — ViT-MAE pretrained features on the unit sphere are very dense:
    # cosine similarity between different-class centroids routinely exceeds 0.5.
    # A high threshold (e.g. 0.6) causes ALL prototypes to merge into 1.
    # Use 0.15 so only near-identical prototypes merge; everything else is added.
    # Range: 0.05–0.4 for pretrained ViT-MAE features.
    "merge_threshold": 0.15,

    # Server-side EMA alpha for global prototype updates
    # Lower values = slower, more stable updates
    # Range: 0.01–0.2
    "server_ema_alpha": 0.1,

    # Maximum capacity of the global prototype bank.
    # New prototypes are not added once this limit is reached.
    # Range: 20–200
    "max_global_prototypes": 500,

    # ── GPAD Distillation Loss (loss.py) ─────────────────────────────────────
    # Base similarity threshold for global anchoring in GPAD
    # Higher = stricter gating, fewer anchors activated
    # Range: 0.3–0.7
    "gpad_base_tau": 0.4,

    # Sigmoid gate temperature for steepness control in GPAD
    # Lower = sharper (near step-function), higher = smoother
    # Range: 0.05–0.5
    "gpad_temp_gate": 0.1,

    # Uncertainty scaling factor for the entropy-based adaptive threshold
    # Higher lambda = threshold rises more steeply as assignment entropy increases
    # Range: 0.1–0.5
    "gpad_lambda_entropy": 0.3,

    # Temperature for the soft assignment distribution in GPAD
    # Controls sharpness of the softmax used in entropy calculation
    # Range: 0.05–0.5
    "gpad_soft_assign_temp": 0.1,

    # Numerical epsilon for GPAD loss computation (prevents div-by-zero)
    "gpad_epsilon": 1e-8,

    # ── Client Local Training (client.py) ────────────────────────────────────
    # Number of prototype centroids each client generates via K-Means (Round 1)
    # Range: 5–50
    "k_init_prototypes": 50,

    # Optimizer learning rate for local client training
    "client_lr": 1e-4,

    # AdamW weight decay for L2 regularization
    "client_weight_decay": 0.05,

    # Local merge threshold: cosine-similarity for online EMA prototype updates.
    # Only samples more similar than this to their nearest local prototype
    # trigger an update — prevents noisy refinements.
    # Must match the same reasoning as merge_threshold: ViT-MAE pretrained
    # embeddings are dense on the unit sphere, so this must also be low.
    # Range: 0.05–0.3 for pretrained ViT-MAE features.
    "client_local_update_threshold": 0.2,

    # EMA interpolation factor for local non-anchored updates and
    # local buffer centroid merges.
    # 0 = no update, 1 = full replacement.
    # Range: 0.05–0.3
    "client_local_ema_alpha": 0.1,

    # ── GPAD Loss Weighting ──────────────────────────────────────────────────
    # Weight of GPAD distillation loss relative to the main reconstruction loss.
    # total_loss = mae_loss + lambda_proto * gpad_loss.
    # Range: 0.001–0.1
    "lambda_proto": 0.01,

    # ── Novelty Buffer (client.py) ───────────────────────────────────────────
    # Novelty buffer capacity before triggering clustering.
    # Number of truly novel embeddings (failing both global and local
    # threshold checks) to accumulate before triggering a fresh K-Means
    # clustering to discover new visual concepts.
    # Options: 128, 256, 512
    "novelty_buffer_size": 256,

    # K for the buffer K-Means clustering.
    # This is independent of k_init_prototypes (Round 1 full clustering).
    # Range: 3–10
    "novelty_k": 10,

    # ── K-Means (client.py) ──────────────────────────────────────────────────
    # Maximum number of K-Means iterations before stopping
    "kmeans_max_iters": 100,

    # Convergence tolerance for K-Means (centroid shift below this = converged)
    "kmeans_tol": 1e-4,
}


# ==========================================================================
# DATA PIPELINE — Tiny ImageNet with Continual Learning Task Scheduling
# ==========================================================================


def load_tinyimagenet(data_root: str, image_size: int = 224):
    """
    Load the Tiny ImageNet training dataset with transforms.

    Tiny ImageNet contains 200 classes, each with 500 training images of
    64×64 pixels. Images are resized to ``image_size`` (224) to match
    the ViT-MAE's expected input resolution.

    Transforms applied:
    - Resize to (image_size, image_size): Upscale 64×64 → 224×224
    - ToTensor: Convert PIL Image → torch.Tensor [C, H, W] with [0, 1] range

    No augmentation or normalization is applied — the raw resized pixel
    values are fed directly to the ViT-MAE model.

    Parameters
    ----------
    data_root : str
        Root directory where the dataset is (or will be) stored.
    image_size : int
        Target spatial resolution. Default: 224.

    Returns
    -------
    datasets.ImageFolder
        The full Tiny ImageNet training dataset with transforms applied.
        Each sample is a (image_tensor, class_label) pair where
        image_tensor has shape [3, image_size, image_size].
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")

    if not os.path.isdir(train_dir):
        logger.info(
            f"Tiny ImageNet not found at {train_dir}. "
            f"Downloading to {data_root}..."
        )
        _download_tinyimagenet(data_root)

    dataset = datasets.ImageFolder(train_dir, transform=transform)
    logger.info(
        f"Loaded Tiny ImageNet: {len(dataset)} images, "
        f"{len(dataset.classes)} classes"
    )
    return dataset


def _download_tinyimagenet(data_root: str) -> None:
    """
    Download and extract Tiny ImageNet 200 to the given root directory.

    Downloads the official Tiny ImageNet zip file from Stanford's CS231N
    server and extracts it to ``data_root/tiny-imagenet-200/``.

    Parameters
    ----------
    data_root : str
        Root directory to store the dataset.
    """
    import urllib.request
    import zipfile

    os.makedirs(data_root, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")

    logger.info(f"Downloading Tiny ImageNet from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    logger.info(f"Extracting to {data_root}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_root)

    # Clean up the zip file to save disk space.
    os.remove(zip_path)
    logger.info("Tiny ImageNet download and extraction complete.")


def create_task_schedule(
    dataset: datasets.ImageFolder,
    num_rounds: int,
    classes_per_task: int,
    num_clients: int,
    classes_per_client: int,
    seed: int = 42,
) -> List[List[List[int]]]:
    """
    Create the continual learning task schedule for all rounds and clients.

    Splits the 200 Tiny ImageNet classes into sequential tasks, then
    partitions each task's classes across clients in a non-IID fashion.
    Returns a nested list structure where ``schedule[round][client]``
    contains the list of class indices assigned to that client in that round.

    Continual Learning Design
    -------------------------
    - 200 classes are shuffled (deterministically via seed) to randomize
      which concepts appear together.
    - Split into ``num_rounds`` tasks of ``classes_per_task`` classes each.
    - Within each task, classes are divided equally among ``num_clients``
      clients, giving each client ``classes_per_client`` unique classes.
    - This creates non-IID data distributions: no two clients see the same
      classes in the same round.

    Parameters
    ----------
    dataset : ImageFolder
        The full Tiny ImageNet dataset.
    num_rounds : int
        Number of communication rounds (= number of tasks).
    classes_per_task : int
        Total classes introduced per round.
    num_clients : int
        Number of federated clients.
    classes_per_client : int
        Classes assigned to each client per round.
    seed : int
        Random seed for class shuffling.

    Returns
    -------
    List[List[List[int]]]
        ``schedule[round_idx][client_idx]`` = list of class indices.
    """
    # Get all unique class indices and shuffle them deterministically.
    all_classes = list(range(len(dataset.classes)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_classes), generator=rng).tolist()
    shuffled_classes = [all_classes[i] for i in perm]

    schedule = []
    for round_idx in range(num_rounds):
        # Extract this round's classes from the shuffled order.
        start = round_idx * classes_per_task
        end = start + classes_per_task
        task_classes = shuffled_classes[start:end]

        # Divide task classes across clients (non-IID).
        round_schedule = []
        for client_idx in range(num_clients):
            c_start = client_idx * classes_per_client
            c_end = c_start + classes_per_client
            client_classes = task_classes[c_start:c_end]
            round_schedule.append(client_classes)

        schedule.append(round_schedule)

    return schedule


def create_client_dataloaders(
    dataset: datasets.ImageFolder,
    client_classes: List[List[int]],
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> List[DataLoader]:
    """
    Create one DataLoader per client for the current round's task.

    Filters the full dataset to include only samples from each client's
    assigned classes, then wraps them in a DataLoader.

    Parameters
    ----------
    dataset : ImageFolder
        The full Tiny ImageNet training dataset.
    client_classes : List[List[int]]
        ``client_classes[i]`` = list of class indices assigned to client i.
    batch_size : int
        Mini-batch size per client.
    num_workers : int
        DataLoader workers. Default: 4.
    pin_memory : bool
        Pin CUDA memory for faster transfer. Default: True.
    shuffle : bool
        Shuffle the data each epoch. Default: True.

    Returns
    -------
    List[DataLoader]
        One DataLoader per client, each containing only samples from
        that client's assigned classes.
    """
    dataloaders = []

    # Build a mapping: class_idx → list of sample indices for fast lookup.
    # dataset.targets is a list of integer class labels for each sample.
    targets = torch.tensor(dataset.targets)

    for classes in client_classes:
        # Create a boolean mask for all samples belonging to this client's classes.
        mask = torch.zeros(len(dataset), dtype=torch.bool)
        for cls_idx in classes:
            mask |= (targets == cls_idx)

        # Get the indices of matching samples.
        indices = torch.where(mask)[0].tolist()

        # Create a Subset and wrap in a DataLoader.
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


# ==========================================================================
# CHECKPOINTING — Save model, prototypes, and training history
# ==========================================================================


def save_checkpoint(
    save_dir: str,
    round_idx: int,
    base_model: nn.Module,
    proto_bank: GlobalPrototypeBank,
    training_history: Dict[str, Any],
    is_final: bool = False,
) -> str:
    """
    Save a training checkpoint to disk.

    Each checkpoint contains:
    - Model state dict (trainable adapter weights only for efficiency).
    - Global prototype bank tensor.
    - Full training history (losses, timings, bank sizes).

    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints.
    round_idx : int
        Current round number (used in filename).
    base_model : nn.Module
        The global model to save.
    proto_bank : GlobalPrototypeBank
        The global prototype bank.
    training_history : Dict[str, Any]
        Training metrics accumulated over all rounds.
    is_final : bool
        If True, saves as 'final_model.pt' instead of 'round_N.pt'.

    Returns
    -------
    str
        Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract only trainable (adapter) weights to save disk space.
    # Frozen backbone weights are identical to the pretrained checkpoint
    # and don't need to be saved.
    trainable_state = {
        k: v.cpu() for k, v in base_model.state_dict().items()
        if v.requires_grad or "adapter" in k.lower()
    }

    # Fallback: if the above filtering is too aggressive, save all weights.
    # This handles cases where requires_grad state may not be preserved
    # in the state_dict values.
    if len(trainable_state) == 0:
        trainable_state = {
            k: v.cpu() for k, v in base_model.state_dict().items()
        }

    checkpoint = {
        "round": round_idx,
        "model_state_dict": trainable_state,
        "global_prototypes": (
            proto_bank.prototypes.cpu() if proto_bank.prototypes is not None
            else None
        ),
        "training_history": training_history,
        "config": {k: str(v) for k, v in CONFIG.items()},
    }

    filename = "final_model.pt" if is_final else f"round_{round_idx}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

    # Also save training history as JSON for easy visualization.
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return filepath


# ==========================================================================
# TRAINING PROGRESS DISPLAY
# ==========================================================================


def print_round_summary(
    round_idx: int,
    num_rounds: int,
    losses: List[float],
    proto_bank: GlobalPrototypeBank,
    round_time: float,
    training_history: Dict[str, Any],
) -> None:
    """
    Print a formatted summary of the completed round.

    Displays loss per client, global prototype bank size, round timing,
    and cumulative statistics.

    Parameters
    ----------
    round_idx : int
        Current round number.
    num_rounds : int
        Total number of rounds.
    losses : List[float]
        Loss per client for this round.
    proto_bank : GlobalPrototypeBank
        The global prototype bank (to show capacity).
    round_time : float
        Wall-clock time for this round in seconds.
    training_history : Dict[str, Any]
        Full training history for cumulative stats.
    """
    bank_size = (
        proto_bank.prototypes.shape[0] if proto_bank.prototypes is not None
        else 0
    )
    max_bank = CONFIG["max_global_prototypes"]

    print(f"\n{'━' * 60}")
    print(f"  Round {round_idx}/{num_rounds} Complete")
    print(f"{'━' * 60}")
    for i, loss in enumerate(losses):
        print(f"  Client {i} Loss: {loss:.6f}")
    print(f"  Avg Loss:       {sum(losses) / len(losses):.6f}")
    print(f"  Proto Bank:     {bank_size}/{max_bank}")
    print(f"  Round Time:     {round_time:.1f}s")
    print(f"{'━' * 60}\n")


# ==========================================================================
# MAIN ORCHESTRATOR
# ==========================================================================


def main():
    """
    Run the complete Federated Continual Self-Supervised Learning pipeline
    on Tiny ImageNet with 2 T4 GPUs.

    This function orchestrates the full FL pipeline end-to-end using a real
    ViTMAEForPreTraining backbone with IBA adapters. It performs five phases:

    Phase 1 — Environment Setup:
        Detect GPUs, configure execution mode, set random seed.

    Phase 2 — Component Initialization:
        Instantiate GlobalPrototypeBank, FederatedModelServer, the real
        ViT-MAE model with adapters, ClientManager, and GPADLoss.

    Phase 3 — Data Setup:
        Load Tiny ImageNet (200 classes, 100k images), create the continual
        learning task schedule (5 tasks × 40 classes, non-IID split).

    Phase 4 — Federated Training Loop:
        For each round r = 1, ..., 5:
            (A) Broadcast global prototypes to clients.
            (B) Create task-specific dataloaders (non-IID class split).
            (C) Clients train locally (MAE + GPAD from Round 2).
            (D) Extract/collect local prototypes.
            (E) Server aggregates prototypes and weights.
            (F) Load FedAvg weights, save checkpoint.

    Phase 5 — Final Save:
        Save the final model, prototypes, and training history.
    """
    logger.info("Initializing Federated Continual Learning Pipeline...")

    # ── Phase 1: Environment Setup ────────────────────────────────────────────
    # Automatically detect available CUDA GPUs and configure the execution
    # mode. With 2 T4 GPUs, each client gets its own GPU for parallel training.
    if torch.cuda.is_available():
        CONFIG["gpu_count"] = torch.cuda.device_count()
        CONFIG["device"] = "cuda"
        logger.info(
            f"Detected {CONFIG['gpu_count']} GPU(s). "
            f"Device set to '{CONFIG['device']}'. "
            f"Parallel mode enabled if Clients <= GPUs."
        )
    else:
        CONFIG["device"] = "cpu"
        logger.info(
            f"No CUDA GPUs found. Device='{CONFIG['device']}' (Sequential Mode)."
        )

    logger.info(f"Dtype: {CONFIG['dtype']}")

    # Set random seed for reproducibility. This ensures deterministic weight
    # initialization, data shuffling, and class scheduling across runs.
    if CONFIG["seed"] is not None:
        torch.manual_seed(CONFIG["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG["seed"])
        logger.info(f"Random seed set to {CONFIG['seed']}")

    # ── Phase 2: Component Initialization ─────────────────────────────────────
    # All components read their hyperparameters from the centralized CONFIG
    # dictionary, ensuring consistency and enabling easy sweeps.

    # 2A. Server-Side Global Prototype Bank
    # Manages the shared knowledge base of visual concepts. Local prototypes
    # from all clients are merged into this bank each round using the
    # Merge-or-Add strategy with EMA. The bank lives on the configured
    # device and has a capacity limit (max_global_prototypes).
    proto_bank = GlobalPrototypeBank(
        embedding_dim=CONFIG["embedding_dim"],
        merge_threshold=CONFIG["merge_threshold"],
        ema_alpha=CONFIG["server_ema_alpha"],
        device=CONFIG["device"],
        max_prototypes=CONFIG["max_global_prototypes"],
    )

    # 2B. Server-Side Model Aggregator (FedAvg)
    # Computes the element-wise arithmetic mean of all client model state
    # dicts to produce a global consensus model.
    fed_server = FederatedModelServer()

    # 2C. Global Model — Real ViTMAEForPreTraining with IBA Adapters
    # Load the pretrained ViT-MAE backbone from HuggingFace Hub and inject
    # IBA adapters into every encoder layer. The backbone is frozen — only
    # adapter parameters (~1% of total) are trainable.
    logger.info(
        f"Loading pretrained model: {CONFIG['pretrained_model_name']}..."
    )
    base_model = ViTMAEForPreTraining.from_pretrained(
        CONFIG["pretrained_model_name"]
    )

    # Inject IBA adapters: freezes all backbone parameters and adds
    # lightweight bottleneck adapters after each encoder layer.
    base_model = inject_adapters(
        base_model,
        bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
    )
    logger.info("Model loaded and adapters injected successfully.")

    # 2D. Client Manager
    # Factory that spawns N independent FederatedClient instances, each with
    # a deep copy of the base model. Handles device assignment (1:1 GPU
    # mapping) and dispatches training commands in parallel (multi-GPU).
    client_manager = ClientManager(
        base_model=base_model,
        num_clients=CONFIG["num_clients"],
        gpu_count=CONFIG["gpu_count"],
        dtype=CONFIG["dtype"],
        optimizer_kwargs={
            "lr": CONFIG["client_lr"],
            "weight_decay": CONFIG["client_weight_decay"],
        },
        local_update_threshold=CONFIG["client_local_update_threshold"],
        local_ema_alpha=CONFIG["client_local_ema_alpha"],
        lambda_proto=CONFIG["lambda_proto"],
        novelty_buffer_size=CONFIG["novelty_buffer_size"],
        novelty_k=CONFIG["novelty_k"],
        kmeans_max_iters=CONFIG["kmeans_max_iters"],
        kmeans_tol=CONFIG["kmeans_tol"],
    )

    # 2E. GPAD Distillation Loss
    # The Gated Prototype Anchored Distillation loss module. Used from
    # Round 2 onwards to regularize client features against the global
    # prototype bank, preventing catastrophic forgetting.
    gpad_loss = GPADLoss(
        base_tau=CONFIG["gpad_base_tau"],
        temp_gate=CONFIG["gpad_temp_gate"],
        lambda_entropy=CONFIG["gpad_lambda_entropy"],
        soft_assign_temp=CONFIG["gpad_soft_assign_temp"],
        epsilon=CONFIG["gpad_epsilon"],
    )

    # ── Phase 3: Data Setup — Tiny ImageNet ───────────────────────────────────
    # Load the full training dataset and create the continual learning
    # task schedule. 200 classes → 5 tasks × 40 classes.
    logger.info("Loading Tiny ImageNet dataset...")
    dataset = load_tinyimagenet(
        data_root=CONFIG["data_root"],
        image_size=CONFIG["image_size"],
    )

    # Create the task schedule: which classes each client sees in each round.
    task_schedule = create_task_schedule(
        dataset=dataset,
        num_rounds=CONFIG["num_rounds"],
        classes_per_task=CONFIG["classes_per_task"],
        num_clients=CONFIG["num_clients"],
        classes_per_client=CONFIG["classes_per_client"],
        seed=CONFIG["seed"],
    )

    # Log the task schedule for transparency.
    for r, round_sched in enumerate(task_schedule):
        for c, classes in enumerate(round_sched):
            logger.info(
                f"  Round {r+1} | Client {c}: "
                f"{len(classes)} classes → {classes}"
            )

    # ── Phase 4: Federated Training Loop ──────────────────────────────────────
    # Training history for tracking and visualization.
    training_history = {
        "round_losses": [],       # List of [client_0_loss, client_1_loss, ...]
        "avg_losses": [],         # Average loss per round
        "proto_bank_sizes": [],   # Global bank size after each round
        "round_times": [],        # Wall-clock time per round
        "task_classes": [],       # Classes introduced per round
    }

    global_protos = None

    # Main training loop with tqdm progress over rounds.
    round_pbar = tqdm(
        range(1, CONFIG["num_rounds"] + 1),
        desc="Federated Rounds",
        unit="round",
        position=0,
    )

    for round_idx in round_pbar:
        round_start = time.time()

        round_pbar.set_description(
            f"Round {round_idx}/{CONFIG['num_rounds']}"
        )

        logger.info(f"\n{'='*40}")
        logger.info(f"STARTING ROUND {round_idx}/{CONFIG['num_rounds']}")
        logger.info(f"{'='*40}")

        # ── Step A: Server Broadcast ─────────────────────────────────────
        # Send the current global prototype bank to all clients so they can
        # use it for GPAD anchoring. In Round 1, no prototypes exist yet.
        if round_idx > 1:
            logger.info(
                f"Broadcasting {len(global_protos)} Global Prototypes to Clients."
            )

        # ── Step B: Create Task-Specific DataLoaders ─────────────────────
        # Each round introduces a new task with new classes. Each client
        # gets a disjoint subset of the task's classes (non-IID).
        client_classes = task_schedule[round_idx - 1]
        dataloaders = create_client_dataloaders(
            dataset=dataset,
            client_classes=client_classes,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"],
            shuffle=CONFIG["dataloader_shuffle"],
        )

        for i, (dl, classes) in enumerate(zip(dataloaders, client_classes)):
            logger.info(
                f"  Client {i}: {len(dl.dataset)} samples "
                f"from {len(classes)} classes"
            )

        # ── Step C: Client Local Training ────────────────────────────────
        # Each client trains on its private data for one epoch:
        #   Round 1:  Loss = MAE reconstruction loss only.
        #   Round >1: Loss = MAE + lambda_proto × GPAD (anchored embeddings).
        logger.info(">> Clients Training...")
        losses = client_manager.train_round(
            dataloaders,
            global_prototypes=global_protos,
            gpad_loss_fn=gpad_loss,
        )
        logger.info(f"Client Losses: {losses}")

        # ── Step D: Local Prototype Extraction ───────────────────────────
        # Round 1:  Run full K-Means from scratch to produce initial
        #           client prototypes (k_init_prototypes clusters).
        # Round >1: Prototypes are maintained online via per-embedding
        #           routing (EMA updates + novelty buffer clustering).
        client_payloads = []

        if round_idx == 1:
            # --- Round 1: Full K-Means prototype initialization ---
            logger.info(">> Generating Initial Local Prototypes (K-Means)...")
            for i, client in enumerate(client_manager.clients):
                local_protos = client.generate_prototypes(
                    dataloaders[i], K_init=CONFIG["k_init_prototypes"]
                )
                # Move weights to CPU for server-side aggregation.
                weights = {
                    k: v.cpu() for k, v in client.model.state_dict().items()
                }
                payload = {
                    "client_id": f"client_{i}",
                    "protos": local_protos.cpu(),
                    "weights": weights,
                }
                client_payloads.append(payload)
        else:
            # --- Round > 1: Collect online-maintained prototypes ---
            logger.info(">> Collecting Local Prototypes (from routing/buffer)...")
            for i, client in enumerate(client_manager.clients):
                local_protos = client.get_local_prototypes()
                weights = {
                    k: v.cpu() for k, v in client.model.state_dict().items()
                }
                payload = {
                    "client_id": f"client_{i}",
                    "weights": weights,
                }
                # Include prototypes only if the client has generated them.
                if local_protos is not None:
                    payload["protos"] = local_protos.cpu()
                    logger.info(
                        f"  Client {i}: {local_protos.shape[0]} local protos, "
                        f"buffer={len(client.novelty_buffer)}"
                    )
                else:
                    logger.info(f"  Client {i}: No local prototypes yet")
                client_payloads.append(payload)

        # ── Step E: Server Aggregation ───────────────────────────────────
        # The server receives all client payloads and executes two tasks:
        #   (1) Merge local prototypes into the Global Prototype Bank.
        #   (2) Average client model weights using FedAvg.
        logger.info(">> Server Aggregation...")
        server_result = run_server_round(
            proto_manager=proto_bank,
            model_server=fed_server,
            client_payloads=client_payloads,
        )

        # Extract the updated global state from the server's result.
        global_protos = server_result["global_prototypes"]
        global_weights = server_result["global_weights"]

        # ── Step F: Global Model Update + Checkpoint ─────────────────────
        # Load the FedAvg-aggregated weights into the base model. We use
        # strict=False because clients send all weights but frozen backbone
        # weights should not cause issues.
        base_model.load_state_dict(global_weights, strict=False)

        # Record round timing.
        round_time = time.time() - round_start

        # Update training history.
        bank_size = (
            global_protos.shape[0] if global_protos is not None else 0
        )
        training_history["round_losses"].append(losses)
        training_history["avg_losses"].append(sum(losses) / len(losses))
        training_history["proto_bank_sizes"].append(bank_size)
        training_history["round_times"].append(round_time)
        training_history["task_classes"].append(
            [cls_list for cls_list in client_classes]
        )

        # Print formatted round summary.
        print_round_summary(
            round_idx=round_idx,
            num_rounds=CONFIG["num_rounds"],
            losses=losses,
            proto_bank=proto_bank,
            round_time=round_time,
            training_history=training_history,
        )

        # Update tqdm postfix with key metrics.
        round_pbar.set_postfix({
            "avg_loss": f"{training_history['avg_losses'][-1]:.4f}",
            "bank": bank_size,
            "time": f"{round_time:.0f}s",
        })

        # Save checkpoint after every round.
        save_checkpoint(
            save_dir=CONFIG["save_dir"],
            round_idx=round_idx,
            base_model=base_model,
            proto_bank=proto_bank,
            training_history=training_history,
        )

        # Log round completion.
        if global_protos is not None:
            logger.info(
                f"Round {round_idx} Complete. "
                f"Global Bank Size: {global_protos.shape[0]}"
            )
        else:
            logger.error("Round Complete but Global Protos is None!")

    # ── Phase 5: Final Save ───────────────────────────────────────────────────
    # Save the final model checkpoint with all training history.
    save_checkpoint(
        save_dir=CONFIG["save_dir"],
        round_idx=CONFIG["num_rounds"],
        base_model=base_model,
        proto_bank=proto_bank,
        training_history=training_history,
        is_final=True,
    )

    # Print final training summary.
    total_time = sum(training_history["round_times"])
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total Rounds:    {CONFIG['num_rounds']}")
    print(f"  Total Time:      {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Final Avg Loss:  {training_history['avg_losses'][-1]:.6f}")
    print(f"  Final Bank Size: {training_history['proto_bank_sizes'][-1]}")
    print(f"  Checkpoints:     {CONFIG['save_dir']}/")
    print(f"{'═' * 60}\n")

    logger.info("\nPipeline Finished Successfully.")


# ==========================================================================
# Entry Point
# ==========================================================================

if __name__ == "__main__":
    main()