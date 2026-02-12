
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class PrototypeManager:
    def __init__(self, 
                 num_prototypes=20, 
                 merge_threshold=0.85, 
                 embedding_dim=768, 
                 device='cuda'):
        self.num_prototypes = num_prototypes
        self.merge_threshold = merge_threshold
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.global_prototypes = []  # List of tensors
        self.global_counts = []      # List of integers (support)

    def compute_local_prototypes(self, embeddings):
        """
        Step 1: Local Prototype Computation via K-Means
        
        Args:
            embeddings (torch.Tensor): [N, D] local embeddings
            
        Returns:
            prototypes (torch.Tensor): [K, D]
            counts (torch.Tensor): [K]
        """
        if embeddings.size(0) < self.num_prototypes:
            # Fallback if not enough samples
            return embeddings, torch.ones(embeddings.size(0), device=self.device)
            
        # 1. Normalize
        z_norm = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        
        # 2. K-Means
        kmeans = KMeans(n_clusters=self.num_prototypes, n_init=10, random_state=42)
        kmeans.fit(z_norm)
        
        # 3. Get centers and normalize
        centers = torch.tensor(kmeans.cluster_centers_, device=self.device)
        centers = F.normalize(centers, p=2, dim=1)
        
        # 4. Get counts
        labels = kmeans.labels_
        counts = np.bincount(labels, minlength=self.num_prototypes)
        counts = torch.tensor(counts, device=self.device, dtype=torch.float32)
        
        return centers, counts

    def merge_global_prototypes(self, client_prototypes_list, client_counts_list):
        """
        Step 2: Global Prototype Aggregation
        
        Args:
            client_prototypes_list: List of [K, D] tensors
            client_counts_list: List of [K] tensors
            
        Returns:
            merged_prototypes (torch.Tensor): [M, D]
        """
        # 1. Concatenate all
        P_all = torch.cat(client_prototypes_list, dim=0) # [Total_K, D]
        C_all = torch.cat(client_counts_list, dim=0)     # [Total_K]
        
        # 2. Iterative Merging based on Cosine Similarity
        # Greedy approach:
        # a. Sort by count (descending) to prioritize robust prototypes? 
        #    Or just iterate. The guide says "For each prototype p... find similar... merge"
        
        merged_protos = []
        
        # Mask to keep track of merged
        active_mask = torch.ones(P_all.size(0), dtype=torch.bool, device=self.device)
        
        indices = torch.argsort(C_all, descending=True) # Process most supported first
        
        for i in indices:
            if not active_mask[i]:
                continue
                
            p = P_all[i]
            
            # Calculate sim to all OTHER active prototypes
            # (In naive implementation, we might just look at all remaining)
            
            # We want to find all q s.t. cos(p, q) > threshold
            # And merge them into p
            
            sims = torch.mm(P_all, p.unsqueeze(1)).squeeze() # [Total_K]
            
            should_merge = (sims > self.merge_threshold) & active_mask
            
            # Indices to merge
            merge_indices = torch.where(should_merge)[0]
            
            if len(merge_indices) > 0:
                # Weighted average
                # sum(n_q * q) / sum(n_q)
                
                qs = P_all[merge_indices]
                ns = C_all[merge_indices].unsqueeze(1)
                
                # Weighted sum
                merged_vec = (qs * ns).sum(dim=0)
                total_n = ns.sum()
                
                merged_vec = merged_vec / total_n
                merged_vec = F.normalize(merged_vec, p=2, dim=0)
                
                merged_protos.append(merged_vec)
                
                # Mark as processed
                active_mask[merge_indices] = False
                
        if len(merged_protos) == 0:
            return torch.zeros(0, self.embedding_dim, device=self.device)
            
        self.global_prototypes = torch.stack(merged_protos)
        return self.global_prototypes

    def get_prototypes(self):
        return self.global_prototypes
