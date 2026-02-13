import torch
import torch.nn as nn
import torch.nn.functional as F

class GPADLoss(nn.Module):
    """
    Gated Prototype Anchored Distillation (GPAD) Loss.

    This loss function is designed for Federated Continual Learning scenarios where
    local models need to learn from global prototypes without catastrophic forgetting.
    It encourages the local model's embeddings to be close to the best-matching
    global prototype, but only if the match is sufficiently "confident".

    Key Mechanisms:
    1.  **Adaptive Thresholding**: The matching threshold is dynamic. It increases
        when the model is uncertain (high entropy in assignment), preventing
        noisy or ambiguous anchors from distorting the feature space.
    2.  **Gating**: A soft gating mechanism (sigmoid) weighs the loss. If the
        similarity is below the threshold, the gate closes (weight -> 0), minimizing
        the impact of poor matches.
    3.  **Anchoring**: If fused, the loss minimizes the distance between the
        embedding and its matched prototype.
    """

    def __init__(self, 
                base_tau: float = 0.5, 
                temp_gate: float = 0.1, 
                lambda_entropy: float = 0.1):
        """
        Initialize the GPAD Loss.

        Args:
            base_tau (float): The base similarity threshold (0 to 1). Matches with
                            cosine similarity below this value are considered weak.
                            Default: 0.5.
            temp_gate (float): Temperature parameter for the sigmoid gating function.
                            Controls the sharpness of the transition from 0 to 1
                            around the threshold. Lower values make it sharper.
                            Default: 0.1.
            lambda_entropy (float): Coefficient/Weight for the entropy term in the
                                    adaptive threshold calculation. Higher values
                                    make the threshold stricter for uncertain samples.
                                    Default: 0.1.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy

    def forward(self, embeddings: torch.Tensor, global_prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute the GPAD loss for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): The feature embeddings from the local model.
                                        Shape: [Batch_Size (B), Embedding_Dim (D)]
            global_prototypes (torch.Tensor): The global prototypes (centroids) from the server.
                                            Shape: [Num_Prototypes (M), Embedding_Dim (D)]

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss (mean over batch).
                        Returns 0.0 if no prototypes are provided.
        """
        # Handle edge case where no global prototypes exist yet (e.g., first round)
        if global_prototypes is None or global_prototypes.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
            
        B = embeddings.size(0)
        M = global_prototypes.size(0)
        
        # -------------------------------------------------------------------------
        # 1. Normalization
        # Normalize both embeddings and prototypes to the unit hypersphere (L2 norm).
        # This ensures that dot products equate to cosine similarity.
        # -------------------------------------------------------------------------
        z = F.normalize(embeddings, p=2, dim=1)         # Shape: [B, D]
        p = F.normalize(global_prototypes, p=2, dim=1)  # Shape: [M, D]
        
        # -------------------------------------------------------------------------
        # 2. Similarity Computation
        # Compute the cosine similarity matrix between all embeddings and all prototypes.
        # Range: [-1, 1]
        # -------------------------------------------------------------------------
        sims = torch.mm(z, p.t())  # Shape: [B, M]
        
        # -------------------------------------------------------------------------
        # 3. Uncertainty Estimation (Entropy)
        # Calculate the entropy of the assignment probability distribution to measure uncertainty.
        # A high entropy means the embedding is equidistant to multiple prototypes (ambiguous).
        # -------------------------------------------------------------------------
        # Apply softmax with a fixed temperature (0.1) to sharpen the distribution for entropy calc.
        softmax_all = F.softmax(sims / 0.1, dim=1) 
        
        # Standard entropy calculation: -sum(p * log(p))
        # Added eps (1e-8) for numerical stability.
        entropy = -torch.sum(softmax_all * torch.log(softmax_all + 1e-8), dim=1) # Shape: [B]
        
        # Normalize entropy to [0, 1] range to make lambda_entropy scaling consistent.
        # Max entropy for M classes is log(M).
        max_ent = torch.log(torch.tensor(float(M), device=embeddings.device))
        ent_norm = entropy / (max_ent + 1e-8) # Shape: [B]
        
        # -------------------------------------------------------------------------
        # 4. Adaptive Thresholding
        # Calculate a sample-specific threshold.
        # Threshold = Base_Threshold + (Uncertainty_User_Penalty)
        # Highly uncertain samples get a higher (stricter) threshold.
        # -------------------------------------------------------------------------
        tau_adaptive = self.base_tau + self.lambda_entropy * ent_norm # Shape: [B]
        
        # -------------------------------------------------------------------------
        # 5. Best Matches
        # Find the single best matching prototype for each sample.
        # -------------------------------------------------------------------------
        max_sim, best_idx = sims.max(dim=1) # Shape: [B]
        
        # -------------------------------------------------------------------------
        # 6. Anchoring Decision & Gating
        # We only want to pull the embedding towards the prototype if it's a "good" match.
        # 
        # is_anchored: Binary mask (1 if sim > threshold, else 0).
        # gate_sigmoid: Soft continuous gate [0, 1] based on distance to threshold.
        # -------------------------------------------------------------------------
        
        # Hard binary check (used to zero out loss completely if strictly below)
        # Note: In some variations, this might be purely soft. Here we keep the hard check
        # to strictly ignore far-away samples.
        is_anchored = (max_sim > tau_adaptive).float() # Shape: [B]
        
        # Soft gating: 
        #   If max_sim >> tau_adaptive, result -> 1.0
        #   If max_sim << tau_adaptive, result -> 0.0
        #   If max_sim ~= tau_adaptive, result -> 0.5 (controlled by temp_gate)
        gate_logit = (max_sim - tau_adaptive) 
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)
        
        # Combined Gate: Both checks must pass.
        # The 'is_anchored' term ensures we don't apply loss for samples *below* threshold 
        # (though sigmoid would be small, it strictly zeros it).
        gate = gate_sigmoid * is_anchored # Shape: [B]
        
        # -------------------------------------------------------------------------
        # 7. Loss Computation
        # We minimize the Squared Euclidean Distance between the embedding and the prototype.
        # For normalized vectors u, v: ||u - v||^2 = 2(1 - cos(theta)) = 2(1 - u.v)
        # -------------------------------------------------------------------------
        dist_sq = 2 * (1 - max_sim) # Shape: [B]
        
        # Apply the gate to the distance
        loss_per_sample = gate * dist_sq # Shape: [B]
        
        # Average loss over the batch
        loss = loss_per_sample.mean()
        
        return loss