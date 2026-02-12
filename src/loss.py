
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPADLoss(nn.Module):
    def __init__(self, 
                 base_tau=0.5, 
                 temp_gate=0.1, 
                 lambda_entropy=0.1):
        """
        Gated Prototype Anchored Distillation (GPAD) Loss.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy

    def forward(self, embeddings, global_prototypes, global_confidences):
        """
        Args:
            embeddings: [B, D]
            global_prototypes: [M, D]
            global_confidences: [M] (raw counts)
            
        Returns:
            loss: scalar
        """
        if global_prototypes is None or global_prototypes.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        B = embeddings.size(0)
        M = global_prototypes.size(0)
        
        # 1. Normalize
        z = F.normalize(embeddings, p=2, dim=1)
        p = F.normalize(global_prototypes, p=2, dim=1)
        
        # 2. Similarity [B, M]
        sims = torch.mm(z, p.t())
        
        # 3. Soft Assignment & Entropy
        # Softmax over similarities? Usually with temperature.
        # User says: "forms a soft assignment... (via a softmax)"
        softmax_all = F.softmax(sims / 0.1, dim=1) # Fixed temp for assignment?
        entropy = -torch.sum(softmax_all * torch.log(softmax_all + 1e-8), dim=1) # [B]
        
        # 4. Adaptive Threshold
        # "base threshold and adding a term proportional to the normalized entropy"
        # normalizing entropy to [0,1]? Max entropy is log(M).
        max_ent = torch.log(torch.tensor(float(M)))
        ent_norm = entropy / (max_ent + 1e-8)
        
        # User: "adding a term proportional... so detailed uncertain assignments face a stricter threshold"
        # Stricter = Higher threshold.
        tau_adaptive = self.base_tau + self.lambda_entropy * ent_norm # [B]
        
        # 5. Best Matches
        max_sim, best_idx = sims.max(dim=1) # [B]
        
        # 6. Anchored Status
        # "If max sim < adaptive threshold, non-anchored"
        is_anchored = (max_sim > tau_adaptive).float() # [B] 0 or 1
        
        # 7. Gating Weight
        # "sigmoid( (sim - threshold) / maybe_temp ) * anchored"
        # Removed confidence term as requested.
        
        # Sigmoid part
        # (sim - threshold)
        gate_logit = (max_sim - tau_adaptive) 
        # Scale logit 
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)
        
        # Final Gate
        gate = gate_sigmoid * is_anchored # [B]
        
        # 8. Anchoring Loss
        # We multiply by 2 so that the loss value equals the square of the straight-line distance
        # (Euclidean distance) between the two normalized points.
        # Without the 2, it would simply be the cosine distance.
        dist_sq = 2 * (1 - max_sim)
        
        # Weighted mean
        # "averages this gated anchoring loss over the mini-batch"
        loss_per_sample = gate * dist_sq
        loss = loss_per_sample.mean()
        
        return loss
