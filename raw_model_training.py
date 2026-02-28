"""
Raw Model Training Baseline — Incremental Class-by-Class MAE Training.

This script trains a ViT-MAE model with IBA adapters on Tiny ImageNet
WITHOUT any federated learning infrastructure, GPAD loss, or prototype
management. It serves as a naive continual learning baseline to compare
against the full PODFCSSV pipeline.

Training Strategy
-----------------
The model trains incrementally on one class at a time:
  - Class 0: train until all 500 images are seen (1 epoch)
  - Class 1: train until all 500 images are seen (1 epoch)
  - ...
  - Class 199: final class

This simulates the worst-case continual learning scenario where the model
has NO mechanism to prevent catastrophic forgetting. As training progresses
on later classes, the model will gradually forget representations learned
from earlier classes.

Model Architecture
------------------
Same as the federated pipeline:
  - ViT-MAE-Base backbone (frozen, ~111M params)
  - IBA adapters (trainable, ~1.2M params, ~1% of total)
  - MAE masked image reconstruction loss only

The comparison between this baseline and the PODFCSSV pipeline quantifies
the benefit of the federated prototype anchoring system for mitigating
catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import logging
import os
import json
import time
from tqdm import tqdm

# Project imports — only the model and adapter, no server/client/loss.
from transformers import ViTMAEForPreTraining
from src.mae_with_adapter import inject_adapters

# Torchvision for dataset loading and transforms.
from torchvision import transforms, datasets


# ==========================================================================
# Logging Configuration
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RawTraining")


# ==========================================================================
# HYPERPARAMETERS & CONFIGURATION
#
# Mirrors the main.py CONFIG for fair comparison, but strips out all
# federated/GPAD/prototype-related parameters.
# ==========================================================================

CONFIG = {
    # ── System ────────────────────────────────────────────────────────────────
    # Random seed for reproducibility
    "seed": 42,

    # Primary device — auto-detected at runtime
    "device": "cpu",

    # Floating-point precision (float32 for T4 GPUs)
    "dtype": torch.float32,

    # ── Model ────────────────────────────────────────────────────────────────
    # Pretrained model name from HuggingFace Hub
    "pretrained_model_name": "facebook/vit-mae-base",

    # Embedding dimension (must match pretrained model)
    "embedding_dim": 768,

    # Input image size for ViT-MAE
    "image_size": 224,

    # IBA adapter bottleneck dimension
    # Range: 32–128
    "adapter_bottleneck_dim": 64,

    # IBA adapter dropout rate
    # Range: 0.0–0.5
    "adapter_dropout": 0.0,

    # ── Training ─────────────────────────────────────────────────────────────
    # Mini-batch size for training
    "batch_size": 16,

    # Number of epochs to train per class
    # Each class has 500 images → ~31 steps per epoch at batch_size=16
    "epochs_per_class": 1,

    # Optimizer learning rate (same as federated client_lr)
    "lr": 1e-4,

    # AdamW weight decay (same as federated client_weight_decay)
    "weight_decay": 0.05,

    # DataLoader worker processes
    "num_workers": 4,

    # Pin CUDA memory for faster transfer
    "pin_memory": True,

    # ── Dataset ──────────────────────────────────────────────────────────────
    # Root directory for Tiny ImageNet
    "data_root": "./data",

    # Total number of classes in Tiny ImageNet
    "num_classes": 200,

    # ── Checkpointing ────────────────────────────────────────────────────────
    # Directory to save model checkpoints and training history
    "save_dir": "checkpoints_raw",

    # Save a checkpoint every N classes (to avoid excessive I/O)
    "save_every_n_classes": 10,
}


# ==========================================================================
# DATA PIPELINE
# ==========================================================================


def load_tinyimagenet(data_root: str, image_size: int = 224):
    """
    Load the Tiny ImageNet training dataset with minimal transforms.

    Transforms: Resize(224) + ToTensor() only — no augmentation or
    normalization, matching the federated pipeline for fair comparison.

    Parameters
    ----------
    data_root : str
        Root directory where the dataset is stored.
    image_size : int
        Target spatial resolution. Default: 224.

    Returns
    -------
    datasets.ImageFolder
        The full Tiny ImageNet training dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")

    if not os.path.isdir(train_dir):
        logger.info(
            f"Tiny ImageNet not found at {train_dir}. Downloading..."
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
    Download and extract Tiny ImageNet 200 from Stanford CS231N.

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

    os.remove(zip_path)
    logger.info("Tiny ImageNet download and extraction complete.")


def create_class_dataloader(
    dataset: datasets.ImageFolder,
    class_idx: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader containing only samples from a single class.

    Parameters
    ----------
    dataset : ImageFolder
        The full Tiny ImageNet dataset.
    class_idx : int
        The class index to filter for.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin CUDA memory.

    Returns
    -------
    DataLoader
        DataLoader with only samples from the specified class.
    """
    targets = torch.tensor(dataset.targets)
    indices = torch.where(targets == class_idx)[0].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ==========================================================================
# CHECKPOINTING
# ==========================================================================


def save_checkpoint(
    save_dir: str,
    class_idx: int,
    model: nn.Module,
    training_history: dict,
    is_final: bool = False,
) -> str:
    """
    Save a training checkpoint containing adapter weights and history.

    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints.
    class_idx : int
        Current class index (used in filename).
    model : nn.Module
        The model to save.
    training_history : dict
        Training metrics accumulated over all classes.
    is_final : bool
        If True, saves as 'final_model.pt'.

    Returns
    -------
    str
        Path to the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save all weights (frozen backbone is identical to pretrained, but
    # we save everything for simplicity — adapter keys are mixed in).
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "class_idx": class_idx,
        "model_state_dict": state_dict,
        "training_history": training_history,
        "config": {k: str(v) for k, v in CONFIG.items()},
    }

    filename = "final_model.pt" if is_final else f"class_{class_idx:03d}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

    # Also save training history as JSON for easy visualization.
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return filepath


# ==========================================================================
# TRAINING LOOP
# ==========================================================================


def train_one_class(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: torch.dtype,
    class_idx: int,
    epochs: int = 1,
) -> float:
    """
    Train the model on a single class for the specified number of epochs.

    Uses only the MAE masked image reconstruction loss — no GPAD,
    no prototype routing, no federated aggregation. This is vanilla
    self-supervised training on a single class.

    Parameters
    ----------
    model : nn.Module
        The ViTMAE model with adapters.
    dataloader : DataLoader
        DataLoader for the current class.
    optimizer : Optimizer
        AdamW optimizer (updates only adapter parameters).
    device : str
        Device string ('cuda' or 'cpu').
    dtype : torch.dtype
        Floating-point dtype.
    class_idx : int
        Current class index (for logging).
    epochs : int
        Number of epochs to train on this class.

    Returns
    -------
    float
        Average MAE loss over all batches in all epochs.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
        for batch in dataloader:
            # Extract images (ignore labels — self-supervised).
            images = batch[0].to(dtype).to(device)

            # Forward pass: ViT-MAE returns reconstruction loss.
            outputs = model(images)
            loss = outputs.loss

            # Backward pass: gradients flow only through adapters
            # (backbone is frozen).
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


# ==========================================================================
# MAIN
# ==========================================================================


def main():
    """
    Run incremental class-by-class MAE training on Tiny ImageNet.

    This is the naive continual learning baseline:
    - 1 model, no federation
    - Train on class 0, then class 1, ..., then class 199
    - MAE reconstruction loss only (no GPAD / prototype anchoring)
    - Frozen ViT backbone + trainable IBA adapters

    The model will exhibit catastrophic forgetting as it trains on later
    classes, since there is no mechanism to preserve earlier representations.
    """
    logger.info("Initializing Raw Model Training Baseline...")

    # ── Environment Setup ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        CONFIG["device"] = "cuda"
        gpu_count = torch.cuda.device_count()
        logger.info(
            f"Detected {gpu_count} GPU(s). Using '{CONFIG['device']}'."
        )
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        CONFIG["device"] = "cpu"
        logger.info("No CUDA GPUs found. Using CPU.")

    # Set random seed for reproducibility.
    if CONFIG["seed"] is not None:
        torch.manual_seed(CONFIG["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CONFIG["seed"])
        logger.info(f"Random seed set to {CONFIG['seed']}")

    device = CONFIG["device"]
    dtype = CONFIG["dtype"]

    # ── Model Initialization ──────────────────────────────────────────────────
    # Load pretrained ViT-MAE and inject IBA adapters (identical to the
    # federated pipeline for fair comparison).
    logger.info(
        f"Loading pretrained model: {CONFIG['pretrained_model_name']}..."
    )
    model = ViTMAEForPreTraining.from_pretrained(
        CONFIG["pretrained_model_name"]
    )
    model = inject_adapters(
        model,
        bottleneck_dim=CONFIG["adapter_bottleneck_dim"],
    )
    model = model.to(device)
    logger.info("Model loaded, adapters injected, moved to device.")

    # Create optimizer — only adapter parameters are trainable.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    logger.info(
        f"Optimizer: AdamW | lr={CONFIG['lr']} | "
        f"weight_decay={CONFIG['weight_decay']} | "
        f"trainable params: {sum(p.numel() for p in trainable_params):,}"
    )

    # ── Dataset Loading ───────────────────────────────────────────────────────
    logger.info("Loading Tiny ImageNet dataset...")
    dataset = load_tinyimagenet(
        data_root=CONFIG["data_root"],
        image_size=CONFIG["image_size"],
    )

    # Get class order — shuffle deterministically for reproducibility.
    num_classes = len(dataset.classes)
    rng = torch.Generator().manual_seed(CONFIG["seed"])
    class_order = torch.randperm(num_classes, generator=rng).tolist()
    logger.info(
        f"Training order: {num_classes} classes, shuffled with seed={CONFIG['seed']}"
    )

    # ── Training History ──────────────────────────────────────────────────────
    training_history = {
        "class_losses": [],          # Loss for each class
        "class_indices": [],         # Class index order
        "class_times": [],           # Time per class
        "cumulative_avg_loss": [],   # Running average loss
    }

    # ── Incremental Class-by-Class Training ───────────────────────────────────
    total_start = time.time()
    running_loss_sum = 0.0

    # Main progress bar over all classes.
    class_pbar = tqdm(
        enumerate(class_order),
        total=num_classes,
        desc="Incremental Training",
        unit="class",
        position=0,
    )

    for step, class_idx in class_pbar:
        class_start = time.time()

        # Create DataLoader for this single class.
        dataloader = create_class_dataloader(
            dataset=dataset,
            class_idx=class_idx,
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            pin_memory=CONFIG["pin_memory"],
        )

        # Train on this class.
        avg_loss = train_one_class(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
            class_idx=class_idx,
            epochs=CONFIG["epochs_per_class"],
        )

        class_time = time.time() - class_start

        # Update running statistics.
        running_loss_sum += avg_loss
        cumulative_avg = running_loss_sum / (step + 1)

        # Record history.
        training_history["class_losses"].append(avg_loss)
        training_history["class_indices"].append(class_idx)
        training_history["class_times"].append(class_time)
        training_history["cumulative_avg_loss"].append(cumulative_avg)

        # Update tqdm with current metrics.
        class_pbar.set_postfix({
            "class": class_idx,
            "loss": f"{avg_loss:.4f}",
            "avg": f"{cumulative_avg:.4f}",
            "time": f"{class_time:.1f}s",
        })

        # Periodic logging every 10 classes for visibility.
        if (step + 1) % 10 == 0:
            elapsed = time.time() - total_start
            logger.info(
                f"Progress: {step+1}/{num_classes} classes | "
                f"Last loss: {avg_loss:.6f} | "
                f"Cumulative avg: {cumulative_avg:.6f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # Save checkpoint periodically.
        if (step + 1) % CONFIG["save_every_n_classes"] == 0:
            save_checkpoint(
                save_dir=CONFIG["save_dir"],
                class_idx=step,
                model=model,
                training_history=training_history,
            )

    # ── Final Save ────────────────────────────────────────────────────────────
    save_checkpoint(
        save_dir=CONFIG["save_dir"],
        class_idx=num_classes - 1,
        model=model,
        training_history=training_history,
        is_final=True,
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    final_avg = training_history["cumulative_avg_loss"][-1]

    print(f"\n{'=' * 60}")
    print(f"  RAW MODEL TRAINING COMPLETE (Baseline)")
    print(f"{'=' * 60}")
    print(f"  Classes Trained:   {num_classes}")
    print(f"  Total Time:        {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Final Avg Loss:    {final_avg:.6f}")
    print(f"  First 10 Losses:   {[f'{l:.4f}' for l in training_history['class_losses'][:10]]}")
    print(f"  Last 10 Losses:    {[f'{l:.4f}' for l in training_history['class_losses'][-10:]]}")
    print(f"  Checkpoints:       {CONFIG['save_dir']}/")
    print(f"{'=' * 60}\n")

    logger.info("Raw Model Training Finished Successfully.")


# ==========================================================================
# Entry Point
# ==========================================================================

if __name__ == "__main__":
    main()
