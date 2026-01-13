import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional, Union



class LocalPrototypeManager:
    """
    Manages local prototypes (centroids) for a specific client or node.
    
    This class handles the initialization, storage, and updating of prototype 
    vectors based on feature embeddings. It uses K-Means clustering to 
    identify representative prototypes from the data.

    Attributes:
        K (int): The number of prototypes to maintain.
        alpha_ema (float): The decay rate for Exponential Moving Average updates.
        novelty_threshold (float): Threshold distance to consider an embedding 'novel'.
        embedding_dim (int): The dimensionality of the feature embeddings.
        prototypes (Optional[torch.Tensor]): The current prototype vectors [K, embedding_dim].
        prototype_counts (Optional[torch.Tensor]): The number of samples assigned to each prototype [K].
        novelty_buffer (List[torch.Tensor]): A buffer to store embeddings detected as novel.
    """

    def __init__(
        self, 
        K: int = 20, 
        alpha_ema: float = 0.1, 
        novelty_threshold: float = 0.4, 
        embedding_dim: int = 768
    ):
        """
        Initializes the LocalPrototypeManager.

        Args:
            K (int): Number of prototypes (clusters). Defaults to 20.
            alpha_ema (float): Momentum factor for EMA updates (0 < alpha < 1). Defaults to 0.1.
            novelty_threshold (float): Cosine distance threshold for novelty detection. Defaults to 0.4.
            embedding_dim (int): Size of the embedding vector. Defaults to 768.
        """
        self.K = K
        self.alpha_ema = alpha_ema
        self.novelty_threshold = novelty_threshold
        self.embedding_dim = embedding_dim
        
        # State variables
        self.prototypes: Optional[torch.Tensor] = None
        self.prototype_counts: Optional[torch.Tensor] = None
        self.novelty_buffer: List[torch.Tensor] = []
    
    def compute_local_prototypes_kmeans(
        self, 
        embeddings: torch.Tensor, 
        K: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes local prototypes using K-Means clustering on the provided embeddings.
        
        This method normalizes embeddings (forcing spherical K-Means behavior),
        performs clustering using Scikit-Learn, and calculates the mean vector
        for each cluster.

        Args:
            embeddings (torch.Tensor): Input tensor of shape [N, embedding_dim].
            K (int): The number of clusters to form. Defaults to 20.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - prototypes: Tensor of shape [K, embedding_dim].
                - counts: Tensor of shape [K] containing sample counts per cluster.
        
        Raises:
            ValueError: If the input embeddings tensor is empty.
        """
        if embeddings.numel() == 0:
            raise ValueError("Embeddings tensor is empty. Cannot compute prototypes.")

        # Ensure we don't try to find more clusters than samples
        num_samples = embeddings.size(0)
        actual_K = min(K, num_samples)
        
        # Capture the device to ensure outputs match input device
        device = embeddings.device
        
        # 1. Normalize embeddings (L2 norm) so they lie on the hypersphere.
        # This allows K-Means (Euclidean distance) to approximate Cosine Similarity.
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # 2. Perform K-means clustering (requires CPU numpy array)
        # using 'k-means++' for better initialization.
        embeddings_np = embeddings_norm.detach().cpu().numpy()
        
        kmeans = KMeans(
            n_clusters=actual_K, 
            init='k-means++',
            n_init=10, 
            random_state=42
        )
        labels = kmeans.fit_predict(embeddings_np)
        
        # 3. Compute cluster means (Prototypes)
        prototypes_list = []
        counts_list = []
        
        # Convert labels back to tensor for masking
        labels_tensor = torch.tensor(labels, device=device)
        
        for k in range(K):
            if k < actual_K:
                mask = (labels_tensor == k)
                count = mask.sum().item()
                
                if count > 0:
                    # Calculate mean of assigned embeddings
                    proto = embeddings_norm[mask].mean(dim=0)
                    # Re-normalize to maintain unit length
                    proto = F.normalize(proto, dim=0)
                    
                    prototypes_list.append(proto)
                    counts_list.append(count)
                else:
                    # Handle empty cluster (rare with k-means++, but possible)
                    prototypes_list.append(self._get_random_prototype(device))
                    counts_list.append(0)
            else:
                # If N < K, fill remaining slots with random initialized vectors
                prototypes_list.append(self._get_random_prototype(device))
                counts_list.append(0)
        
        # Stack lists into tensors
        self.prototypes = torch.stack(prototypes_list)       # [K, embedding_dim]
        self.prototype_counts = torch.tensor(counts_list, device=device) # [K]
        
        return self.prototypes, self.prototype_counts
    
    def initialize_from_first_batch(self, embeddings: torch.Tensor) -> None:
        """
        Initializes prototypes using the first available batch of data.

        Args:
            embeddings (torch.Tensor): A batch of embeddings [N, embedding_dim].
        """
        # Validate dimensionality
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch. Expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        self.compute_local_prototypes_kmeans(embeddings, K=self.K)

    def _get_random_prototype(self, device: torch.device) -> torch.Tensor:
        """
        Helper method to generate a random normalized prototype.
        
        Args:
            device (torch.device): The device to place the tensor on.

        Returns:
            torch.Tensor: Normalized random vector of shape [embedding_dim].
        """
        rand_proto = torch.randn(self.embedding_dim, device=device)
        return F.normalize(rand_proto, dim=0)
    

class LocalPrototypeManagerV2(LocalPrototypeManager):
    """
    Advanced Local Prototype Manager.

    Extends the basic manager to support:
    1. **EMA Updates**: Gradually shifts existing prototypes towards new data.
    2. **Novelty Detection**: Identifies embeddings that don't fit existing clusters.
    3. **Dynamic Expansion**: Spawns new prototypes when enough novel data accumulates.
    4. **Consolidation**: Merges redundant prototypes to prevent memory explosion.
    """

    def __init__(self, client_id: str = "Unknown", **kwargs):
        """
        Args:
            client_id (str): Identifier for logging.
            **kwargs: Arguments passed to the parent LocalPrototypeManager.
        """
        super().__init__(**kwargs)
        self.client_id = client_id
        
        # Ensure novelty buffer is initialized if parent didn't do it
        if not hasattr(self, 'novelty_buffer'):
            self.novelty_buffer: List[List[float]] = []

    @torch.no_grad()
    def update_prototypes(
        self, 
        new_embeddings: torch.Tensor, 
        round_id: int, 
        task_id: int,
        global_prototypes: torch.Tensor, 
        global_confidence: torch.Tensor
    ) -> None:
        """
        Updates the local prototype bank based on the results of the current training round.

        Args:
            new_embeddings (torch.Tensor): Features extracted from current local data [N, Dim].
            round_id (int): Current federated round number.
            task_id (int): Current task identifier (for task-incremental scenarios).
            global_prototypes (torch.Tensor): Global reference prototypes [K_global, Dim].
            global_confidence (torch.Tensor): Global confidence scores.
        """
        if new_embeddings.numel() == 0:
            return

        # 1. Device Management
        # Ensure operations happen on the device where prototypes are stored
        device = self.prototypes.device if self.prototypes is not None else new_embeddings.device
        new_embeddings = new_embeddings.to(device)
        
        # Normalize for Cosine Similarity
        new_embeddings_norm = F.normalize(new_embeddings, dim=1)
        
        # Handle Cold Start (if prototypes somehow don't exist yet)
        if self.prototypes is None:
            self.initialize_from_first_batch(new_embeddings)
            return

        # Step 1: Assign new embeddings to existing local prototypes
        # [N, Dim] @ [Dim, K] -> [N, K]
        sims = torch.mm(new_embeddings_norm, self.prototypes.T)
        best_sims, assigned_idx = sims.max(dim=1)  # [N], [N]
        
        # Step 2: EMA Update for Existing Prototypes
        # We iterate only over prototypes that actually received samples to save compute
        present_indices = torch.unique(assigned_idx)
        alpha = self.alpha_ema
        
        for k in present_indices:
            mask = (assigned_idx == k)
            count = mask.sum().item()
            
            # Calculate mean of assigned samples
            z_mean = new_embeddings_norm[mask].mean(dim=0)
            
            # EMA Update Formula: Old * (1-a) + New * a
            self.prototypes[k] = (1 - alpha) * self.prototypes[k] + alpha * z_mean
            
            # Re-normalize to stay on the hypersphere
            self.prototypes[k] = F.normalize(self.prototypes[k], dim=0)
            
            # Update counts
            self.prototype_counts[k] += count

        # Step 3: Detect Novelty
        # Identify samples that are too far from ANY existing local prototype
        novel_mask = best_sims < self.novelty_threshold
        num_novel = novel_mask.sum().item()
        
        if num_novel > 0:
            # Extract novel embeddings
            # We convert to Python list (float) to save GPU memory while buffering
            novel_embeddings = new_embeddings_norm[novel_mask].cpu().tolist()
            self.novelty_buffer.extend(novel_embeddings)
            
            print(f"[Client {self.client_id}] Detected {num_novel} novel samples. Buffer size: {len(self.novelty_buffer)}")
        
        # Step 4: Dynamic Expansion (Create new prototypes)
        # Trigger creation if buffer is large enough
        if len(self.novelty_buffer) > 50:
            self._expand_prototypes(device)

        # Step 5: Periodic Consolidation
        # Cleanup redundant prototypes every 10 rounds
        if round_id > 0 and round_id % 10 == 0:
            self.consolidate_prototypes()

    def _expand_prototypes(self, device: torch.device) -> None:
        """Helper method to cluster the novelty buffer and append new prototypes."""
        
        # Convert buffer to tensor for clustering
        # Note: We create this on CPU first because Sklearn requires CPU
        novel_tensor_cpu = torch.tensor(self.novelty_buffer)
        
        # Determine number of new clusters (heuristic: 1 cluster per 10 samples, max 3)
        num_new_clusters = min(3, len(novel_tensor_cpu) // 10)
        
        if num_new_clusters > 0:
            # KMeans runs on CPU (NumPy)
            kmeans = KMeans(n_clusters=num_new_clusters, n_init=5, random_state=42)
            labels = kmeans.fit_predict(novel_tensor_cpu.numpy())
            
            new_protos_list = []
            
            for cluster_id in range(num_new_clusters):
                mask = (labels == cluster_id)
                # Compute mean of cluster
                cluster_data = novel_tensor_cpu[mask]
                if len(cluster_data) > 0:
                    proto = cluster_data.mean(dim=0)
                    proto = F.normalize(proto, dim=0)
                    new_protos_list.append(proto)
            
            if new_protos_list:
                # Move new prototypes to the correct device (GPU)
                new_protos_tensor = torch.stack(new_protos_list).to(device)
                
                # Append to existing bank
                self.prototypes = torch.cat([self.prototypes, new_protos_tensor], dim=0)
                
                # Initialize counts (giving them a 'warm start' weight of 10)
                new_counts = torch.full((len(new_protos_list),), 10.0, device=device)
                self.prototype_counts = torch.cat([self.prototype_counts, new_counts], dim=0)
                
                print(f"[Client {self.client_id}] Expanded capacity! Created {len(new_protos_list)} new prototypes.")

        # Clear buffer after processing
        self.novelty_buffer = []

    @torch.no_grad()
    def consolidate_prototypes(self, threshold: float = 0.95) -> None:
        """
        ENHANCEMENT 2: Removes redundant prototypes.
        
        If two prototypes are very similar (cosine sim > threshold), 
        the one with fewer counts is removed.

        Args:
            threshold (float): Cosine similarity threshold for merging.
        """
        if self.prototypes is None or len(self.prototypes) < 2:
            return

        # Compute Similarity Matrix: [K, K]
        sims = torch.mm(self.prototypes, self.prototypes.T)
        
        to_remove = set()
        num_protos = len(self.prototypes)
        
        # Greedy consolidation
        for i in range(num_protos):
            if i in to_remove:
                continue
                
            for j in range(i + 1, num_protos):
                if j in to_remove:
                    continue
                
                if sims[i, j] > threshold:
                    # Merge logic: Remove the one with fewer historical samples
                    if self.prototype_counts[j] > self.prototype_counts[i]:
                        to_remove.add(i)
                        break # i is removed, stop checking i against others
                    else:
                        to_remove.add(j)
        
        if not to_remove:
            return

        # Create keep mask
        keep_mask = torch.tensor(
            [k not in to_remove for k in range(num_protos)], 
            device=self.prototypes.device, 
            dtype=torch.bool
        )
        
        # Apply mask
        old_count = len(self.prototypes)
        self.prototypes = self.prototypes[keep_mask]
        self.prototype_counts = self.prototype_counts[keep_mask]
        
        print(f"[Client {self.client_id}] Consolidated prototypes: {old_count} → {len(self.prototypes)}")