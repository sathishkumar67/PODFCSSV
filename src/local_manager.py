import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional

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