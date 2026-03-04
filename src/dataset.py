import os
import torch
import logging
from typing import List, Dict, Tuple, Any
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

logger = logging.getLogger("DistillFed.Dataset")

# ============================================================================
# ViT-MAE Standard Transforms
# ============================================================================

def get_transforms(image_size: int, is_train: bool) -> transforms.Compose:
    """
    Standard robust augmentation and resizing pipeline for ViT-MAE.
    Includes Grayscale to RGB conversion to ensure 3-channel input for all.
    """
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3), # Force 3 channels (e.g., MNIST -> RGB)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(base_transforms)

# ============================================================================
# 20-Dataset Factory
# ============================================================================

def fetch_dataset(name: str, root: str, is_train: bool, transform: Any) -> Tuple[Dataset, int]:
    """
    Fetches one of 20 out-of-distribution datasets. Handles varying torchvision API parameters
    (e.g., train=True vs split='train') and returns the Dataset object along with its num_classes.
    """
    name = name.lower()
    os.makedirs(root, exist_ok=True)
    
    # 1. CIFAR-10
    if name == "cifar10":
        ds = datasets.CIFAR10(root, train=is_train, download=True, transform=transform)
        return ds, 10
    # 2. CIFAR-100
    elif name == "cifar100":
        ds = datasets.CIFAR100(root, train=is_train, download=True, transform=transform)
        return ds, 100
    # 3. SVHN
    elif name == "svhn":
        split = 'train' if is_train else 'test'
        ds = datasets.SVHN(root, split=split, download=True, transform=transform)
        return ds, 10
    # 4. Oxford Flowers-102
    elif name == "flowers102":
        split = 'train' if is_train else 'test'
        ds = datasets.Flowers102(root, split=split, download=True, transform=transform)
        # Fix labels to be 0-indexed if necessary (torchvision usually handles this, but some versions don't)
        return ds, 102
    # 5. Food-101
    elif name == "food101":
        split = 'train' if is_train else 'test'
        ds = datasets.Food101(root, split=split, download=True, transform=transform)
        return ds, 101
    # 6. Oxford-IIIT Pet
    elif name == "pets37":
        split = 'trainval' if is_train else 'test'
        ds = datasets.OxfordIIITPet(root, split=split, download=True, transform=transform)
        return ds, 37
    # 7. DTD (Describable Textures Dataset)
    elif name == "dtd":
        split = 'train' if is_train else 'test'
        ds = datasets.DTD(root, split=split, download=True, transform=transform)
        return ds, 47
    # 8. GTSRB (German Traffic Sign Recognition Benchmark)
    elif name == "gtsrb":
        split = 'train' if is_train else 'test'
        ds = datasets.GTSRB(root, split=split, download=True, transform=transform)
        return ds, 43
    # 9. MNIST
    elif name == "mnist":
        ds = datasets.MNIST(root, train=is_train, download=True, transform=transform)
        return ds, 10
    # 10. FashionMNIST
    elif name == "fmnist":
        ds = datasets.FashionMNIST(root, train=is_train, download=True, transform=transform)
        return ds, 10
    # 11. KMNIST
    elif name == "kmnist":
        ds = datasets.KMNIST(root, train=is_train, download=True, transform=transform)
        return ds, 10
    # 12. USPS
    elif name == "usps":
        ds = datasets.USPS(root, train=is_train, download=True, transform=transform)
        return ds, 10
    # 13. STL-10
    elif name == "stl10":
        split = 'train' if is_train else 'test'
        ds = datasets.STL10(root, split=split, download=True, transform=transform)
        return ds, 10
    # 14. FGVC Aircraft
    elif name == "aircraft":
        split = 'trainval' if is_train else 'test'
        ds = datasets.FGVCAircraft(root, split=split, download=True, transform=transform)
        return ds, 100
    # 15. Country211
    elif name == "country211":
        split = 'train' if is_train else 'test'
        ds = datasets.Country211(root, split=split, download=True, transform=transform)
        return ds, 211
    # 16. RenderedSST2
    elif name == "sst2":
        split = 'train' if is_train else 'test'
        ds = datasets.RenderedSST2(root, split=split, download=True, transform=transform)
        return ds, 2
    # 17. FER2013 (Facial Expressions)
    elif name == "fer2013":
        split = 'train' if is_train else 'test'
        ds = datasets.FER2013(root, split=split, download=True, transform=transform)
        return ds, 7
    # 18. SUN397
    elif name == "sun397":
        # SUN397 has no strictly defined train/test split in the base API, so we load the whole 
        # thing and manually split it reliably via a seeded generator
        ds_full = datasets.SUN397(root, download=True, transform=transform)
        gen = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(ds_full))
        test_len = len(ds_full) - train_len
        ds_train, ds_test = random_split(ds_full, [train_len, test_len], generator=gen)
        return ds_train if is_train else ds_test, 397
    # 19. Caltech101
    elif name == "caltech101":
        # Same manual split logic
        ds_full = datasets.Caltech101(root, download=True, transform=transform)
        gen = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(ds_full))
        test_len = len(ds_full) - train_len
        ds_train, ds_test = random_split(ds_full, [train_len, test_len], generator=gen)
        return ds_train if is_train else ds_test, 101
    # 20. EuroSAT
    elif name == "eurosat":
        # Same manual split logic
        ds_full = datasets.EuroSAT(root, download=True, transform=transform)
        gen = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(ds_full))
        test_len = len(ds_full) - train_len
        ds_train, ds_test = random_split(ds_full, [train_len, test_len], generator=gen)
        return ds_train if is_train else ds_test, 10
    else:
        raise ValueError(f"Dataset '{name}' is not one of the 20 supported Multi-Domain datasets.")

# ============================================================================
# Target Abstraction
# ============================================================================

def _get_dataset_targets(dataset: Dataset) -> np.ndarray:
    """
    Safely extracts the target labels array from any torchvision dataset.
    Because every dataset hides its labels differently (targets, labels, _labels, etc).
    """
    if isinstance(dataset, Subset):
        parent_targets = _get_dataset_targets(dataset.dataset)
        return np.array([parent_targets[i] for i in dataset.indices])
    
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    if hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    
    # Fallback: iterate (slow, but guaranteed to work)
    logger.warning("Target extraction fell back to iteration. This might take a few seconds.")
    targets = [dataset[i][1] for i in range(len(dataset))]
    return np.array(targets)


# ============================================================================
# Multi-Domain Continual Learning Manager
# ============================================================================

class MultiDomainDatasetManager:
    """
    Manages the 1-Dataset-per-Client architecture.
    Each client is assigned a completely distinct dataset (e.g., Client A = CIFAR100, Client B = SVHN).
    That dataset is then partitioned into sequential tasks based on classes (Continual Learning).
    """
    def __init__(
        self,
        client_datasets: List[str],
        data_root: str,
        image_size: int,
        num_rounds: int,
        val_split_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            client_datasets: List of dataset names. Length perfectly equals num_clients.
            data_root: Where to download datasets.
            image_size: Target image resolution for transforms.
            num_rounds: Total server tasks. We slice the dataset's classes by this number.
            val_split_ratio: How much of the train set to isolate for local validation.
            seed: Random seed for reproducibility.
        """
        self.num_clients = len(client_datasets)
        self.num_rounds = num_rounds
        self.val_split_ratio = val_split_ratio
        
        # State
        # Structure: self.client_data[client_id] = {"train_ds": ds, "test_ds": ds, "num_classes": N, "client_tasks": []}
        self.client_data = {}
        
        train_transform = get_transforms(image_size, is_train=True)
        test_transform = get_transforms(image_size, is_train=False)
        
        for client_id, ds_name in enumerate(client_datasets):
            logger.info(f"Preparing dataset '{ds_name}' for Client {client_id}...")
            
            # 1. Fetch
            train_full, num_classes = fetch_dataset(ds_name, data_root, True, train_transform)
            test_full, _ = fetch_dataset(ds_name, data_root, False, test_transform)
            
            # 2. Extract targets for splitting
            train_targets = _get_dataset_targets(train_full)
            test_targets = _get_dataset_targets(test_full)
            
            # 3. Create Class-Incremental Task Schedule
            # Divide the available classes into `num_rounds` disjoint subsets.
            np.random.seed(seed + client_id)
            all_classes = np.random.permutation(num_classes)
            classes_per_round = np.array_split(all_classes, num_rounds)
            
            # Create a task lookup list where client_tasks[r] = {train_indices, val_indices, test_indices}
            client_tasks = []
            
            for r in range(num_rounds):
                round_classes = set(classes_per_round[r].tolist())
                
                # Find indices of samples belonging to this round's classes
                task_train_full_idx = np.where(np.isin(train_targets, list(round_classes)))[0]
                task_test_idx = np.where(np.isin(test_targets, list(round_classes)))[0]
                
                # Split train into train & val
                val_len = int(self.val_split_ratio * len(task_train_full_idx))
                # Shuffle before splitting
                shuffled_idx = np.random.permutation(task_train_full_idx)
                task_val_idx = shuffled_idx[:val_len]
                task_train_idx = shuffled_idx[val_len:]
                
                client_tasks.append({
                    "classes": list(round_classes),
                    "train_indices": task_train_idx.tolist(),
                    "val_indices": task_val_idx.tolist(),
                    "test_indices": task_test_idx.tolist()
                })
                
            self.client_data[client_id] = {
                "name": ds_name,
                "train_ds": train_full,
                "test_ds": test_full,
                "num_classes": num_classes,
                "tasks": client_tasks
            }
        
        # 4. Enforce Dynamic Truncation (Perfectly Balanced Local Datasets)
        # Find the minimum available task sizes across all clients for EACH round,
        # then violently truncate the larger ones down to that minimum.
        for r in range(num_rounds):
            min_train = min(len(self.client_data[c]["tasks"][r]["train_indices"]) for c in range(self.num_clients))
            min_val = min(len(self.client_data[c]["tasks"][r]["val_indices"]) for c in range(self.num_clients))
            
            for c in range(self.num_clients):
                # Truncate by dropping excess samples at the end (already shuffled)
                self.client_data[c]["tasks"][r]["train_indices"] = self.client_data[c]["tasks"][r]["train_indices"][:min_train]
                self.client_data[c]["tasks"][r]["val_indices"] = self.client_data[c]["tasks"][r]["val_indices"][:min_val]
        
        for client_id in range(self.num_clients):
            num_classes = self.client_data[client_id]["num_classes"]
            ds_name = self.client_data[client_id]["name"]
            logger.info(f"Client {client_id} [{ds_name}]: {num_classes} total classes split into {num_rounds} dynamically balanced tasks.")

    def get_client_loaders(self, client_id: int, round_idx: int, batch_size: int, num_workers: int, pin_memory: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns the (train, val, test) DataLoaders for a specific client during a specific round.
        Because we use Class-Incremental slices, the data is entirely novel for each round.
        """
        c_data = self.client_data[client_id]
        task_info = c_data["tasks"][round_idx]
        
        train_subset = Subset(c_data["train_ds"], task_info["train_indices"])
        val_subset = Subset(c_data["train_ds"], task_info["val_indices"])
        test_subset = Subset(c_data["test_ds"], task_info["test_indices"])
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,          # Only shuffle train
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
        
    def get_client_info(self, client_id: int) -> dict:
        """Returns metadata about the client's dataset."""
        return {
            "name": self.client_data[client_id]["name"],
            "total_classes": self.client_data[client_id]["num_classes"]
        }
