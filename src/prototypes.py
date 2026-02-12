
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

class LocalPrototypeManager:
    def __init__(self, 
                embedding_dim=768, 
                novelty_threshold=0.4, 
                novelty_buffer_size=500,
                ema_alpha=0.1,
                device='cuda'):
        self.embedding_dim = embedding_dim
        self.novelty_threshold = novelty_threshold
        self.novelty_buffer_size = novelty_buffer_size
        self.ema_alpha = ema_alpha
        self.device = device
        
        # Storage
        self.prototypes = torch.zeros(0, embedding_dim, device=device) # [K, D]
        self.counts = torch.zeros(0, device=device) # [K]
        self.novelty_buffer = [] # List of tensors
        
    def update_batch(self, embeddings):
        """
        Online update of local prototypes.
        
        Args:
            embeddings (torch.Tensor): [B, D]
        """
        if embeddings.shape[0] == 0:
            return

        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # If no prototypes yet, add all to buffer
        if self.prototypes.shape[0] == 0:
            self._add_to_buffer(embeddings)
            return

        # 1. Compute similarity to existing prototypes
        # [B, K]
        sims = torch.mm(embeddings, self.prototypes.t())
        max_sim, best_idx = sims.max(dim=1)
        
        # 2. Identify anchored vs novel
        # anchored: max_sim > threshold
        anchored_mask = max_sim > self.novelty_threshold
        
        # 3. Update anchored
        if anchored_mask.any():
            anchored_emb = embeddings[anchored_mask]
            anchored_idx = best_idx[anchored_mask]
            
            # EMA Update
            # For each unique prototype index that got hit, update it?
            # Or per sample?
            # "Update that local prototype by exponential moving average towards the embedding"
            # We can do scatter update or loop for simplicity (K is small ~20)
            
            unique_idxs = anchored_idx.unique()
            for k in unique_idxs:
                # Average of all embeddings assigned to k in this batch
                mask_k = (anchored_idx == k)
                mean_emb_k = anchored_emb[mask_k].mean(dim=0)
                
                # Update: p = (1-alpha)*p + alpha*new
                self.prototypes[k] = (1 - self.ema_alpha) * self.prototypes[k] + self.ema_alpha * mean_emb_k
                self.prototypes[k] = F.normalize(self.prototypes[k], p=2, dim=0)
                self.counts[k] += mask_k.sum()

        # 4. Handle novel
        if (~anchored_mask).any():
            novel_emb = embeddings[~anchored_mask]
            self._add_to_buffer(novel_emb)
            
    def _add_to_buffer(self, embeddings):
        for emb in embeddings:
            self.novelty_buffer.append(emb)
            
        if len(self.novelty_buffer) >= self.novelty_buffer_size:
            self._cluster_buffer()
            
    def _cluster_buffer(self):
        """Run K-Means on buffer and add new prototypes."""
        if len(self.novelty_buffer) < 2:
             return
             
        # Stack
        buffer_tensor = torch.stack(self.novelty_buffer)
        
        # Determine K for buffer? 
        # Strategy: heuristic (e.g., size // 50) or fixed small number?
        # User said: "client runs K-Means... appends these new prototypes"
        # Let's pick K based on data size, e.g., max(1, N/20) constrained.
        N = buffer_tensor.size(0)
        K_new = max(1, min(10, N // 10)) 
        
        # Move to CPU for sklearn
        X = buffer_tensor.cpu().numpy()
        
        kmeans = KMeans(n_clusters=K_new, n_init=10)
        kmeans.fit(X)
        
        new_centers = torch.tensor(kmeans.cluster_centers_, device=self.device)
        new_centers = F.normalize(new_centers, p=2, dim=1)
        
        # Counts
        labels = kmeans.labels_
        new_counts = torch.tensor(np.bincount(labels, minlength=K_new), device=self.device, dtype=torch.float32)
        
        # Append
        self.prototypes = torch.cat([self.prototypes, new_centers], dim=0)
        self.counts = torch.cat([self.counts, new_counts], dim=0)
        
        # Clear buffer
        self.novelty_buffer = []
        
    def get_prototypes(self):
        # Force flush buffer if non-empty at end of round?
        # Usually good idea before sending.
        if len(self.novelty_buffer) > 10: # Minimal threshold
             self._cluster_buffer()
        return self.prototypes, self.counts


class ServerPrototypeManager:
    def __init__(self, merge_threshold=0.85, device='cuda'):
        self.merge_threshold = merge_threshold
        self.device = device
        self.global_prototypes = torch.zeros(0, 768, device=device)
        self.global_counts = torch.zeros(0, device=device)
        self.global_confidences = torch.zeros(0, device=device) # Derived from counts? "increment count which serves as confidence"

    def merge_batch(self, local_protos_list, local_counts_list):
        """
        Aggregates local prototypes into global set.
        
        Args:
            local_protos_list: List of [K_i, D]
            local_counts_list: List of [K_i]
        """
        # Flatten
        all_incoming = []
        all_counts = []
        
        for p, c in zip(local_protos_list, local_counts_list):
            if p.nelement() > 0:
                 all_incoming.append(p)
                 all_counts.append(c)
                 
        if not all_incoming:
            return self.global_prototypes, self.global_confidences
            
        incoming_protos = torch.cat(all_incoming, dim=0)
        incoming_counts = torch.cat(all_counts, dim=0)
        
        # Iterate and Merge or Add
        # "If similar to some global proto -> merge. Else -> add."
        
        for i in range(incoming_protos.size(0)):
            p_new = incoming_protos[i]
            c_new = incoming_counts[i]
            
            if self.global_prototypes.size(0) == 0:
                # Initialize
                self.global_prototypes = p_new.unsqueeze(0)
                self.global_counts = torch.tensor([1.0], device=self.device) # "Initial count of one"
                continue
                
            # Sim to globals
            sims = torch.mm(self.global_prototypes, p_new.unsqueeze(1)).flatten()
            max_sim, best_idx = sims.max(dim=0)
            
            if max_sim > self.merge_threshold:
                # Merge
                k = best_idx
                # Count-weighted average? 
                # User says: "merge it into that global prototype via a count-weighted average"
                # "increment that global prototype's count"
                
                # Formula: new_G = (G * count_G + P * count_P) / (count_G + count_P) ? 
                # OR user says "increment count". 
                # Logic: The global count tracks "how many times it was merged".
                # But for the vector average, do we use the *incoming* count?
                
                # Ideally: global_proto is weighted sum of all contributors.
                # Recursive update: G_new = (G_old * N_old + P_new * N_new) / (N_old + N_new)
                # But user says: "increment that global prototype's count, which serves as its confidence".
                # This implies global_count is just an integer counter of merges.
                
                # Let's stick to simple: Move G towards P.
                old_count = self.global_counts[k]
                # New vector
                # If we assume global_count roughly tracks mass, we can use it.
                # Or just simple average: (G + P)/2
                # Let's use weighted:
                # self.global_prototypes[k] = (self.global_prototypes[k] * old_count + p_new) / (old_count + 1)
                # This treats p_new as weight 1.
                
                # If we use `c_new` (local support) as weight:
                # self.global_prototypes[k] = (self.global_prototypes[k] * old_count + p_new * c_new) / (old_count + c_new)
                # Use simple increment for confidence count.
                
                # Let's use simple logic: weighted by simple 1 vs N merge
                self.global_prototypes[k] = (self.global_prototypes[k] * old_count + p_new) / (old_count + 1)
                self.global_prototypes[k] = F.normalize(self.global_prototypes[k], p=2, dim=0)
                self.global_counts[k] += 1
                
            else:
                # Add
                self.global_prototypes = torch.cat([self.global_prototypes, p_new.unsqueeze(0)], dim=0)
                self.global_counts = torch.cat([self.global_counts, torch.tensor([1.0], device=self.device)], dim=0)
                
        # Update Confidences
        # Normalize counts? 
        # User: "global prototype's count, which serves as its confidence"
        # We can normalize it to [0, 1] range relative to max for the Gated Loss usage.
        # But here we just return raw counts?
        # The Loss function needs "global confidence (reflecting how many local prototypes...)"
        # So raw count is fine, Loss will normalize it or sigmoid it.
        
        self.global_confidences = self.global_counts
        
        return self.global_prototypes, self.global_confidences
