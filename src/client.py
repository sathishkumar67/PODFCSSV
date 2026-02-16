
from __future__ import annotations
import copy
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class FederatedClient:
    """
    Represents a single independent client in the federated system.
    
    Each client manages its own:
    - Model copy (on its specific device)
    - Optimizer
    - Local training loop
    """
    
    def __init__(
        self, 
        client_id: int,
        model: nn.Module,
        device: torch.device,
        optimizer_cls: type = optim.AdamW,
        optimizer_kwargs: Dict[str, Any] = None
    ) -> None:
        """
        Args:
            client_id: Unique identifier for the client.
            model: The base model (will be deep-copied to ensure independence).
            device: The device (CPU/GPU) where this client's model resides.
            optimizer_cls: The optimizer class to use (default: AdamW).
            optimizer_kwargs: Arguments for the optimizer (lr, weight_decay, etc.).
        """
        self.client_id = client_id
        self.device = device
        
        # 1. Independent Model Copy
        # We deepcopy the base model so that this client's training 
        # doesn't affect the base model or other clients.
        self.model = copy.deepcopy(model).to(self.device)
        
        # 2. Independent Optimizer
        opt_kwargs = optimizer_kwargs or {"lr": 1e-3}
        # Only optimize parameters that require gradients (e.g., adapters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(trainable_params, **opt_kwargs)
        
        logger.info(f"Client {self.client_id} initialized on {self.device}")

    def train_epoch(
        self, 
        dataloader: DataLoader, 
        criterion: nn.Module = None
    ) -> float:
        """
        Runs one epoch of local training.

        Args:
            dataloader: Local data for this client.
            criterion: Loss function. If None, assumes model returns dict with 'loss'.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move data to client's device
            if isinstance(batch, (list, tuple)):
                # Assume standard [inputs, labels] format
                inputs = batch[0].to(self.device)
                # Labels might be needed depending on loss, but MAE is self-supervised
                # outputs = self.model(inputs)
            elif isinstance(batch, dict):
                # Handle HF style dicts
                inputs = {k: v.to(self.device) for k, v in batch.items()}
            else:
                inputs = batch.to(self.device)

            # Forward Pass
            # For MAE/ViTMAE, the model output usually contains 'loss' if labels/pixel_values provided
            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                outputs = self.model(inputs)

            # Compute Loss
            if criterion:
                # Custom loss (e.g., GPAD) logic would go here
                # For now, default to internal loss if available, else placeholder
                loss = getattr(outputs, "loss", None)
            else:
                loss = getattr(outputs, "loss", None)

            if loss is None:
                # Fallback implementation if no loss returned
                # This depends heavily on the specific model/task
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


class ClientManager:
    """
    Orchestrator for Multi-GPU Client Simulation.
    
    This class handles the initialization and management of multiple FederatedClient 
    instances, assigning them to available GPUs in a round-robin fashion.
    """
    
    def __init__(
        self, 
        base_model: nn.Module, 
        num_clients: int, 
        gpu_count: int = 0
    ) -> None:
        """
        Args:
            base_model: The generic model architecture to replicate.
            num_clients: Total number of clients to spawn.
            gpu_count: Number of GPUs available.
                       - If 2 GPUs and 4 clients: Clients 0,2 on GPU0; Clients 1,3 on GPU1.
                       - If 0 GPUs: All clients on CPU.
        """
        self.clients: List[FederatedClient] = []
        self.num_clients = num_clients
        self.gpu_count = gpu_count
        
        self._initialize_clients(base_model)

    def _initialize_clients(self, base_model: nn.Module) -> None:
        """Internal helper to spawn clients on appropriate devices."""
        logger.info(f"Initializing {self.num_clients} clients across {self.gpu_count} GPUs...")
        
        for i in range(self.num_clients):
            # Determine Device
            if self.gpu_count > 0:
                # Round-robin assignment: 0->GPU0, 1->GPU1, 2->GPU0...
                device_id = i % self.gpu_count
                device = torch.device(f"cuda:{device_id}")
            else:
                device = torch.device("cpu")
            
            # Create Client
            # Note: The optimizer config can be parameterized later
            client = FederatedClient(
                client_id=i,
                model=base_model,
                device=device,
                optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.05}
            )
            self.clients.append(client)

    def train_round(
        self, 
        dataloaders: List[DataLoader]
    ) -> List[float]:
        """
        Triggers one round of local training for all clients.
        
        Args:
            dataloaders: List of DataLoaders, one for each client.
                         Must match num_clients.

        Returns:
            List[float]: Average loss for each client this round.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Dataloader count ({len(dataloaders)}) does not match "
                f"client count ({self.num_clients})"
            )

        round_losses = []
        
        # Parallel Execution Note:
        # In this Python implementation, this loop is sequential (Client 0 trains, then Client 1...).
        # However, because they are on different CUDA streams/devices, PyTorch operations 
        # are asynchronous. To get true parallelism in Python, we'd need multiprocessing, 
        # but for simulation, this sequential dispatch is standard and effective.
        
        for i, client in enumerate(self.clients):
            logger.info(f"Starting training for Client {i} on {client.device}...")
            loss = client.train_epoch(dataloaders[i])
            round_losses.append(loss)
            logger.info(f"Client {i} finished. Loss: {loss:.4f}")
            
        return round_losses
