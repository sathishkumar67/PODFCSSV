import torch
import torchvision
import numpy as np
import json
import os
from torchvision import transforms

# 2 clients and 2 tasks (50 classes per task)
def setup_cifar100_continual_federated(
    root='./data', 
    num_clients=2,      # CHANGED: Set to 2 Clients
    num_tasks=2,        # CHANGED: Set to 2 Tasks (100 classes / 2 tasks = 50 classes per task)
    alpha=0.5, 
    seed=42
):
    """
    Downloads CIFAR-100 and creates a JSON file mapping:
    Client ID -> Task ID -> List of Image Indices
    """
    print(f"Initializing Data Split: {num_clients} Clients, {num_tasks} Tasks (50 classes/task), Alpha={alpha}")
    
    # 1. Fix Seeds for Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Download CIFAR-100
    # Ensure data directory exists
    os.makedirs(root, exist_ok=True)
    
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    
    targets = np.array(train_dataset.targets)
    classes = np.arange(100)
    
    # 3. Define Tasks (Class Incremental)
    # With num_tasks=2, this creates:
    # Task 0: Classes 0-49
    # Task 1: Classes 50-99
    classes_per_task = 100 // num_tasks
    task_splits = {t: classes[t*classes_per_task : (t+1)*classes_per_task] for t in range(num_tasks)}
    
    # Structure to save: client_data[client_id][task_id] = [indices]
    client_data = {cid: {tid: [] for tid in range(num_tasks)} for cid in range(num_clients)}
    test_data = {tid: [] for tid in range(num_tasks)}

    # 4. Perform Dirichlet Split per Task
    for task_id, task_classes in task_splits.items():
        print(f"Processing Task {task_id} (Classes {task_classes[0]}-{task_classes[-1]})...")
        
        # Save Test Data for this task (Global test set for evaluation)
        test_indices = [i for i, t in enumerate(test_dataset.targets) if t in task_classes]
        test_data[task_id] = test_indices

        # Split Training Data
        for c in task_classes:
            # Get all indices for this specific class
            idx_k = np.where(targets == c)[0]
            np.random.shuffle(idx_k)
            
            # Generate Dirichlet distribution for this class across clients
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Normalize proportions to strictly sum to 1 (handling floating point issues)
            proportions = proportions / proportions.sum()
            
            # Calculate split points based on proportions
            # Logic: Cumulative sum of proportions * total items -> cast to int for indices
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Split the indices
            idx_batch = np.split(idx_k, split_points)
            
            # Assign to clients
            for client_id in range(num_clients):
                client_data[client_id][task_id].extend(idx_batch[client_id].tolist())

    # 5. Save Metadata
    save_path = os.path.join(root, 'federated_splits.json')
    meta_data = {
        'client_data': client_data,
        'test_data': test_data,
        'task_config': {t: task_splits[t].tolist() for t in task_splits}
    }
    
    with open(save_path, 'w') as f:
        json.dump(meta_data, f)
        
    print(f"✅ Data preparation complete. Splits saved to {save_path}")
    
    # Validation Print
    print("\n--- Split Statistics ---")
    for cid in range(num_clients):
        print(f"Client {cid}:")
        for tid in range(num_tasks):
            count = len(client_data[cid][tid])
            print(f"  Task {tid}: {count} samples (Classes {task_splits[tid][0]}-{task_splits[tid][-1]})")

if __name__ == "__main__":
    setup_cifar100_continual_federated()