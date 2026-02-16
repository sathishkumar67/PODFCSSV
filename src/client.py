
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
        global_prototypes: torch.Tensor = None,
        gpad_loss_fn: nn.Module = None
    ) -> float:
        """
        Runs one epoch of local training.
        
        Phased Logic:
        - If global_prototypes is None (Round 1): Train with MAE Reconstruction Loss only.
        - If global_prototypes is provided (Round > 1): Train with MAE + GPAD Loss.

        Args:
            dataloader: Local data for this client.
            global_prototypes: Global prototypes (M, D) for GPAD.
            gpad_loss_fn: Loss function instance for GPAD.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move data to client's device
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device).float()
            elif isinstance(batch, dict):
                inputs = {k: v.to(self.device).float() for k, v in batch.items()}
            else:
                inputs = batch.to(self.device).float()

            # Forward Pass
            # We need hidden_states if doing GPAD
            output_hidden = (global_prototypes is not None)
            
            if isinstance(inputs, dict):
                outputs = self.model(**inputs, output_hidden_states=output_hidden)
            else:
                outputs = self.model(inputs, output_hidden_states=output_hidden)

            # 1. Base MAE Loss
            mae_loss = getattr(outputs, "loss", None)
            if mae_loss is None:
                # Fallback if model doesn't compute loss internally (unlikely for ViTMAEForPreTraining)
                mae_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            final_loss = mae_loss

            # 2. GPAD Loss (if applicable)
            if global_prototypes is not None and gpad_loss_fn is not None:
                # Extract embeddings from last hidden state
                if hasattr(outputs, "hidden_states"):
                    hidden = outputs.hidden_states[-1] 
                elif isinstance(outputs, tuple):
                    # Check tuple structure for HF models (usually loss, logits, hidden_states...)
                    # Index depends on specific model return signature
                    hidden = outputs[-1] # Risky, better to rely on object or attribute
                else:
                    hidden = outputs # Fallback
                
                # Pool: (B, L, D) -> (B, D)
                if len(hidden.shape) == 3:
                    embeddings = hidden.mean(dim=1)
                else:
                    embeddings = hidden
                
                # Compute GPAD
                # Ensure global prototypes are on same device
                protos_device = global_prototypes.to(self.device)
                gpad = gpad_loss_fn(embeddings, protos_device)
                
                final_loss = final_loss + gpad

            # Backward Pass
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss


    @torch.no_grad()
    def generate_prototypes(self, dataloader: DataLoader, K_init: int = 10) -> torch.Tensor:
        """
        Generates local prototypes using K-Means clustering on feature embeddings.

        Args:
            dataloader: Data to extract features from.
            K_init: Number of clusters (prototypes) to form.

        Returns:
            torch.Tensor: Local prototypes (K_init, D).
        """
        self.model.eval()
        all_features = []
        
        # 1. Feature Extraction (Forward Pass)
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
            elif isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
            else:
                inputs = batch.to(self.device)
            
            # Extract features from the model
            # Assuming model returns an object with 'hidden_states' or similar, 
            # or for ViTMAE, we might need to tap into the encoder output.
            # For simplicity, let's assume the model returns a direct embedding or 'last_hidden_state'
            # If standard ViTMAE, outputs.last_hidden_state is (B, L, D). We usually pool it (e.g. CLS or mean).
            # Let's assume Mean Pooling for prototype generation if sequence provided.
            
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs, output_hidden_states=True)
                else:
                    # Some models (like standard HF ViT) output tuple if no keys
                    outputs = self.model(inputs, output_hidden_states=True)

            # Handle HF Output
            if hasattr(outputs, "hidden_states"):
                # Use the last encoder layer's hidden state
                hidden = outputs.hidden_states[-1] 
            elif isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs
                
            # Pooling: (B, L, D) -> (B, D)
            if len(hidden.shape) == 3:
                # Mean pooling over sequence length
                features = hidden.mean(dim=1) 
            else:
                features = hidden
                
            all_features.append(features)

        # Concatenate all features: (N_samples, D)
        embeddings = torch.cat(all_features, dim=0)
        
        # 2. Normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 3. K-Means Clustering (Simple PyTorch Implementation)
        return self._kmeans(embeddings, K=K_init)

    def _kmeans(self, X: torch.Tensor, K: int, max_iters: int = 100) -> torch.Tensor:
        """
        Simple K-Means implementation in PyTorch.
        """
        N, D = X.shape
        
        # Initialize centroids randomly from data
        indices = torch.randperm(N)[:K]
        centroids = X[indices].clone()
        
        for _ in range(max_iters):
            # Compute distances: ||x - c||^2
            # (Using cosine distance since inputs are normalized)
            # dist = 1 - cos_sim
            
            # Normalize centroids to keep consistent with embedding space (unit sphere)
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
            
            # Similarity matrix: (N, K)
            sims = torch.mm(X, centroids.t())
            # Distance is monotonic with 1-sim, so maximizing sim is minimizing dist
            
            # Assign clusters
            _, labels = sims.max(dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                cluster_mask = (labels == k)
                if cluster_mask.sum() > 0:
                    new_centroids[k] = X[cluster_mask].mean(dim=0)
                else:
                    # Re-initialize empty cluster
                    new_idx = torch.randint(0, N, (1,)).item()
                    new_centroids[k] = X[new_idx]
            
            # Check convergence
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            if center_shift < 1e-4:
                break
                
        return torch.nn.functional.normalize(centroids, p=2, dim=1)


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
        logger.info(f"Initializing {self.num_clients} clients...")
        
        # Enforce 1:1 Mapping rule if GPUs are available
        if self.gpu_count > 0:
            if self.num_clients != self.gpu_count:
                raise ValueError(
                    f"Strict 1:1 Client-GPU mapping required. "
                    f"Requested {self.num_clients} clients but found {self.gpu_count} GPUs."
                )
            logger.info(f"Parallel Mode: Mapped {self.num_clients} clients to {self.gpu_count} GPUs.")
        else:
            logger.info(f"Sequential Mode: Running {self.num_clients} clients on CPU.")

        for i in range(self.num_clients):
            # Determine Device
            if self.gpu_count > 0:
                # 1:1 Mapping: Client i -> GPU i
                device = torch.device(f"cuda:{i}")
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
        dataloaders: List[DataLoader],
        global_prototypes: torch.Tensor = None,
        gpad_loss_fn: nn.Module = None
    ) -> List[float]:
        """
        Triggers one round of local training for all clients.
        
        Execution Strategy:
        - If GPUs > 0: Parallel execution via ThreadPool (1 client per GPU).
        - If GPUs == 0: Sequential execution on CPU.
        
        Args:
            dataloaders: List of DataLoaders, one for each client.
                        Must match num_clients.
            global_prototypes: Global prototypes (if r > 1).
            gpad_loss_fn: GPAD Loss instance (if r > 1).

        Returns:
            List[float]: Average loss for each client this round.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Dataloader count ({len(dataloaders)}) does not match "
                f"client count ({self.num_clients})"
            )

        round_losses = [0.0] * self.num_clients
        
        if self.gpu_count > 0:
            # Parallel Execution for GPUs
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                logger.info(f"Spawning {self.num_clients} training threads (1 per GPU)...")
                futures = {}
                for i, client in enumerate(self.clients):
                    futures[executor.submit(
                        client.train_epoch, 
                        dataloaders[i], 
                        global_prototypes=global_prototypes, 
                        gpad_loss_fn=gpad_loss_fn
                    )] = i
                
                for future in futures:
                    client_idx = futures[future]
                    try:
                        loss = future.result()
                        round_losses[client_idx] = loss
                        logger.info(f"Client {client_idx} (GPU {client_idx}) finished. Loss: {loss:.4f}")
                    except Exception as e:
                        logger.error(f"Client {client_idx} failed: {e}")
                        round_losses[client_idx] = float('nan')
        else:
            # Sequential Execution for CPU
            logger.info(f"Running sequential training on CPU for {self.num_clients} clients...")
            for i, client in enumerate(self.clients):
                try:
                    loss = client.train_epoch(
                        dataloaders[i],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn
                    )
                    round_losses[i] = loss
                    logger.info(f"Client {client_idx} (CPU) finished. Loss: {loss:.4f}")
                except Exception as e:
                    logger.error(f"Client {i} failed: {e}")
                    round_losses[i] = float('nan')
            
        return round_losses