"""Loss functions used by the federated continual learning pipeline.

This module contains the implementation of Gated Prototype Anchored
Distillation (GPAD). GPAD is the regularizer that keeps client embeddings close
to the global prototype bank only when the match is confident enough.

The implementation follows the paper's sequence exactly:
1. L2-normalize the client embeddings and the global prototypes.
2. Compute cosine similarities between every embedding and every prototype.
3. Build a soft assignment distribution over the prototype bank.
4. Convert the assignment entropy into an adaptive anchor threshold.
5. Mark an embedding as anchored when its best similarity is greater than or
   equal to that threshold.
6. Apply a smooth sigmoid gate on top of the hard anchor decision.
7. Penalize anchored embeddings with squared Euclidean distance on the unit
   sphere, which simplifies to ``2 * (1 - cosine_similarity)``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPADLoss(nn.Module):
    """Compute the GPAD objective for a mini-batch of embeddings.

    Parameters
    ----------
    base_tau:
        Base cosine-similarity threshold used when the assignment entropy is
        zero.
    temp_gate:
        Temperature of the sigmoid gate. Smaller values make the transition
        around the threshold steeper.
    lambda_entropy:
        Multiplier applied to the normalized entropy when building the adaptive
        threshold.
    soft_assign_temp:
        Temperature used inside the prototype softmax assignment.
    epsilon:
        Small constant used to avoid ``log(0)`` and division-by-zero issues.
    """

    def __init__(
        self,
        base_tau: float = 0.5,
        temp_gate: float = 0.1,
        lambda_entropy: float = 0.1,
        soft_assign_temp: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy
        self.soft_assign_temp = soft_assign_temp
        self.epsilon = epsilon

    def forward(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean GPAD loss for the provided embeddings.

        The function returns a zero-valued tensor that remains connected to the
        graph when the prototype bank is empty. That keeps the training loop
        simple and avoids special-case loss handling in the caller.
        """
        if global_prototypes.size(0) == 0 or embeddings.size(0) == 0:
            return embeddings.sum() * 0.0

        similarities = self._compute_similarity_matrix(
            embeddings=embeddings,
            prototypes=global_prototypes,
        )
        tau_adaptive = self._compute_adaptive_threshold(similarities)
        max_similarity, gate = self._compute_gating(similarities, tau_adaptive)
        return self._compute_anchored_loss(max_similarity, gate)

    def _compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cosine-similarity matrix ``[batch, num_prototypes]``."""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(normalized_embeddings, normalized_prototypes.t())

    def _compute_adaptive_threshold(self, similarities: torch.Tensor) -> torch.Tensor:
        """Convert prototype-assignment entropy into an adaptive threshold.

        When there is only one prototype in the bank, the assignment entropy is
        always zero. In that case the adaptive threshold collapses cleanly to
        ``base_tau``.
        """
        batch_size, num_prototypes = similarities.shape
        if num_prototypes <= 1:
            return similarities.new_full((batch_size,), self.base_tau)

        assignment = F.softmax(similarities / self.soft_assign_temp, dim=1)
        entropy = -torch.sum(
            assignment * torch.log(assignment.clamp(min=self.epsilon)),
            dim=1,
        )
        max_entropy = torch.log(
            similarities.new_tensor(float(num_prototypes)).clamp(min=1.0)
        )
        normalized_entropy = entropy / max_entropy.clamp(min=self.epsilon)
        return self.base_tau + self.lambda_entropy * normalized_entropy

    def _compute_gating(
        self,
        similarities: torch.Tensor,
        tau_adaptive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the best similarity and the final GPAD gate.

        The paper defines the anchor condition with ``>=``. The implementation
        matches that exactly so the code and the equations now share the same
        boundary behavior.
        """
        max_similarity, _ = similarities.max(dim=1)
        hard_anchor = (max_similarity >= tau_adaptive).to(max_similarity.dtype)
        soft_gate = torch.sigmoid((max_similarity - tau_adaptive) / self.temp_gate)
        return max_similarity, hard_anchor * soft_gate

    def _compute_anchored_loss(
        self,
        max_similarity: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gated squared Euclidean distance on the unit sphere."""
        squared_distance = 2.0 * (1.0 - max_similarity)
        return (gate * squared_distance).mean()

    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean mask that marks which embeddings are anchored.

        The caller uses this mask only for routing decisions. No gradients are
        required here, so the method stays in inference mode.
        """
        if global_prototypes.size(0) == 0 or embeddings.size(0) == 0:
            return torch.zeros(
                embeddings.size(0),
                dtype=torch.bool,
                device=embeddings.device,
            )

        similarities = self._compute_similarity_matrix(embeddings, global_prototypes)
        tau_adaptive = self._compute_adaptive_threshold(similarities)
        max_similarity, _ = similarities.max(dim=1)
        return max_similarity >= tau_adaptive
