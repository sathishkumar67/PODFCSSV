from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GPADLoss(nn.Module):
    """
    Gated Prototype Anchored Distillation (GPAD) Loss.

    This module implements a regularization loss for Federated Continual Learning that
    prevent catastrophic forgetting by anchoring the local model's latent representations
    to a set of globally learned prototypes.

    Mathematical Formulation:
    -------------------------
    The loss for a single embedding 'z' is defined as:
    
        L_gpad(z) = Gate(z) * Distance(z, v*)
    
    Where:
    - v*: The "best matching" global prototype for z (highest cosine similarity).
    - Distance(z, v*): Euclidean distance between normalized z and v*.
    - Gate(z): A soft gating factor in [0, 1] that determines how strongly to enforce this anchor.

    The Gating Mechanism:
    ---------------------
    The gate is designed to be high (near 1.0) only when the model is "confident" that 
    the embedding z truly belongs to prototype v*. It uses an Adaptive Threshold:

        Gate(z) = Sigmoid( (Sim(z, v*) - Threshold) / Temperature )

    The Threshold is not fixed; it adapts based on the uncertainty of the assignment:
        
        Threshold = Base_Tau + Lambda * Entropy(z)

    If the assignment is ambiguous (high entropy across all prototypes), the threshold 
    increases, making it harder for the gate to open. This prevents the model from 
    learning from noisy or uncertain matches.
    """
    
    # Small constant to prevent division by zero or log(0) errors
    EPSILON: float = 1e-8
    
    # Temperature for the softmax used in entropy calculation (controls sharpness of distribution)
    SOFT_ASSIGNMENT_TEMP: float = 0.1

    def __init__(self, 
                base_tau: float = 0.5, 
                temp_gate: float = 0.1, 
                lambda_entropy: float = 0.1):
        """
        Initialize the GPAD Loss module.

        Args:
            base_tau (float): The minimum similarity required to consider a match "valid" 
                            in the most confident case (zero entropy). 
                            Range: [0, 1]. Default: 0.5.
            temp_gate (float): Temperature scaling for the sigmoid gate. 
                            Lower values (e.g., 0.01) make the gate a sharp Step Function.
                            Higher values (e.g., 1.0) make it a smooth transition.
                            Default: 0.1.
            lambda_entropy (float): Scaling factor for the uncertainty penalty. 
                                    A higher lambda means the threshold rises more steeply 
                                    as the assignment entropy increases.
                                    Default: 0.1.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy

    def forward(self, 
                embeddings: torch.Tensor, 
                global_prototypes: torch.Tensor) -> torch.Tensor:
        """
        Calculates the batch-averaged GPAD loss.

        Process Flow:
        1. Normalize inputs (embeddings & prototypes) to unit length.
        2. Compute Cosine Similarity Matrix (Batch x Num_Prototypes).
        3. Compute Uncertainty (Entropy) for every sample in the batch.
        4. Derive Adaptive Thresholds for each sample based on its uncertainty.
        5. Compute Gating Factors comparing best similarity vs threshold.
        6. Compute Weighted Distance Loss for valid anchors.

        Args:
            embeddings (torch.Tensor): Latent features from the local student model.
                                    Shape: [Batch_Size, Embedding_Dim]
            global_prototypes (torch.Tensor): Global concepts from the server.
                                            Shape: [Num_Prototypes, Embedding_Dim]

        Returns:
            torch.Tensor: Scalar loss value (mean across the batch).        
                        Returns 0.0 if global_prototypes is empty.
        """ 
        # 1. Normalization & Similarity
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)
        
        # 2. Uncertainty & Thresholding
        tau_adaptive = self._compute_adaptive_threshold(sims)
        
        # 3. Gating & Anchoring
        max_sim, gate = self._compute_gating(sims, tau_adaptive)
        
        # 4. Loss Computation
        loss = self._compute_anchored_loss(max_sim, gate)
        
        return loss

    def _compute_similarity_matrix(self, embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity matrix.
        
        Since inputs are L2-normalized:
            Cosine Similarity = Dot Product .
        
        Returns:
            torch.Tensor: Similarity matrix of shape [Batch_Size, Num_Prototypes].
        """
        z = F.normalize(embeddings, p=2, dim=1)
        p = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(z, p.t())

    def _compute_adaptive_threshold(self, sims: torch.Tensor) -> torch.Tensor:
        """
        Calculates the dynamic similarity threshold for each sample.

        Logic:
            1. Convert similarities to a probability distribution (Softmax).
            2. Compute Entropy of this distribution (Measure of ambiguity).
            3. Normalize entropy to [0, 1] range.
            4. Threshold = Base + Lambda * Normalized_Entropy.
        
        Returns:
            torch.Tensor: Threshold values of shape [Batch_Size].
        """
        B, M = sims.shape
        # Soft assignment distribution
        softmax_all = F.softmax(sims / self.SOFT_ASSIGNMENT_TEMP, dim=1)
        
        # Entropy calculation: -sum(p * log(p))
        entropy = -torch.sum(softmax_all * torch.log(softmax_all + self.EPSILON), dim=1)
        
        # Normalize entropy relative to max possible entropy: log(M)
        # This ensures the entropy penalty scale is consistent regardless of bank size.
        max_ent = torch.log(torch.tensor(float(M), device=sims.device))
        ent_norm = entropy / (max_ent + self.EPSILON)
        
        # Compute threshold
        return self.base_tau + self.lambda_entropy * ent_norm

    def _compute_gating(self, sims: torch.Tensor, tau_adaptive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determines the gating weight for each sample based on its best match.

        Logic:
            1. Identify the 'Best Match' (highest similarity prototype).
            2. Check if Similarity > Adaptive Threshold.
            3. Apply Sigmoid to the margin (Sim - Threshold) to get a soft weight [0, 1].
            4. Mask out (set to 0) any samples that strictly fail the threshold check
            to ensure sparse, high-quality anchoring.

        Returns:
            max_sim (torch.Tensor): The highest cosine similarity score for each sample.
            gate (torch.Tensor): The final gating weight in [0, 1].
        """
        max_sim, _ = sims.max(dim=1)
        
        # Hard/Binary Check: Is it strictly above the required threshold?
        # Only valid anchors contribute to the loss.
        is_anchored = (max_sim > tau_adaptive).float()
        
        # Soft Gating: Sigmoid function centered at the threshold.
        # This provides a smooth gradient for samples near the boundary.
        gate_logit = (max_sim - tau_adaptive)
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)
        
        # Combination: Only apply soft weight if it passes the hard check
        gate = gate_sigmoid * is_anchored
        return max_sim, gate

    def _compute_anchored_loss(self, max_sim: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Computes the final weighted distance loss.

        Objective: Minimize distance ||z - v*|| for anchored samples.
        
        Relation between Cosine Sim (S) and Euclidean Dist (D) for unit vectors:
            D^2 = ||z - v||^2 = (z-v).(z-v) = z.z - 2z.v + v.v = 1 - 2S + 1 = 2(1 - S)
            D = sqrt( 2 * (1 - S) )

        Returns:
            torch.Tensor: Scalar mean loss over the batch.
        """
        # Squared Euclidean Distance derived from Cosine Similarity
        dist_sq = 2 * (1 - max_sim)
        
        # Linear Euclidean Distance (with epsilon for sqrt stability at 0)
        dist = torch.sqrt(dist_sq + self.EPSILON)
        
        # Apply Gating:
        # High confidence match -> High Gate -> Full Distance Penalty
        # Low confidence match -> Low Gate -> Little/No Penalty
        loss_per_sample = gate * dist
        
        return loss_per_sample.mean()