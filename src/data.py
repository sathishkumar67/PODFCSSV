
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
import os

class FederatedDataManager:
    def __init__(self, 
                 dataset_name='cifar100', 
                 num_clients=10, 
                 num_tasks=5, 
                 alpha=0.5, 
                 seed=42,
                 batch_size=32):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.seed = seed
        self.batch_size = batch_size
        
        self.class_order = None
        self.tasks = [] # List of lists of classes
        self.client_data_indices = defaultdict(lambda: defaultdict(list)) # {task_id: {client_id: [indices]}}
        
        # Load Data
        self._load_dataset()
        
        # Create Tasks
        self._create_tasks()
        
        # Partition Data
        self._partition_data()

    def _load_dataset(self):
        # Transformations
        # Check if we should use 224 (ViT) or 32 (CIFAR original) -> Guide says resize to 224 for ViT
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        data_root = './data'
        os.makedirs(data_root, exist_ok=True)

        if self.dataset_name == 'cifar100':
            self.train_set = datasets.CIFAR100(root=data_root, train=True, download=True, transform=self.transform)
            self.test_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=self.transform)
        else:
            raise NotImplementedError("Only CIFAR-100 is supported for this demo.")
        
        self.targets = np.array(self.train_set.targets)
        self.classes = self.train_set.classes

    def _create_tasks(self):
        """
        Splits classes into `num_tasks` tasks.
        """
        np.random.seed(self.seed)
        num_classes = len(self.classes)
        self.class_order = np.random.permutation(num_classes)
        
        classes_per_task = num_classes // self.num_tasks
        for i in range(self.num_tasks):
            task_classes = self.class_order[i * classes_per_task : (i + 1) * classes_per_task]
            self.tasks.append(task_classes.tolist())
            print(f"[Task {i}] Assigned {len(task_classes)} classes.")

    def _partition_data(self):
        """
        Partition data using Dirichlet distribution.
        """
        print(f"[Data] Partitioning among {self.num_clients} clients (Alpha={self.alpha})...")
        np.random.seed(self.seed)
        
        for task_id, task_classes in enumerate(self.tasks):
            # 1. Identify all indices belonging to this task
            # Faster lookup
            task_indices_mask = np.isin(self.targets, task_classes)
            task_indices_all = np.where(task_indices_mask)[0]
            
            # 2. Distribute per class to ensure label presence (if alpha is high) or imbalance (if low)
            # We iterate per class to apply Dirichlet
            
            client_indices = [[] for _ in range(self.num_clients)]
            
            for k in task_classes:
                idx_k = np.where(self.targets == k)[0]
                np.random.shuffle(idx_k)
                
                # Dirichlet logic
                # proportions ~ Dir(alpha)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                
                # Normalize just in case, though dirichlet sums to 1
                proportions = np.array([p * (len(idx_j) < len(task_indices_all) / self.num_clients * 1.5) for p, idx_j in zip(proportions, client_indices)])
                proportions = proportions / proportions.sum()
                
                # Split indices
                split_indices = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_split = np.split(idx_k, split_indices)
                
                for client_id, indices in enumerate(idx_split):
                    client_indices[client_id].extend(indices.tolist())
            
            # Assign to self.client_data_indices
            for client_id, indices in enumerate(client_indices):
                if len(indices) == 0:
                     print(f"[Warning] Client {client_id} has no data for Task {task_id}")
                self.client_data_indices[task_id][client_id] = indices
        
        print("[Data] Partitioning complete.")

    def get_dataloader(self, client_id, task_id, mode='train'):
        if mode == 'train':
            indices = self.client_data_indices[task_id][client_id]
            dataset = Subset(self.train_set, indices)
            # Drop last to avoid single-sample batches messing up BatchNorm or similar
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif mode == 'test':
            # Test set for the task
            task_classes = self.tasks[task_id]
            # Indices for these classes in test set
            test_targets = np.array(self.test_set.targets)
            indices = np.where(np.isin(test_targets, task_classes))[0]
            dataset = Subset(self.test_set, indices)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError("Mode must be 'train' or 'test'")

if __name__ == "__main__":
    dm = FederatedDataManager()
