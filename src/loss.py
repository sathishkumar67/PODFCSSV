"""Define the GPAD loss used by the federated continual-learning path.

GPAD is the prototype-aware regularizer layered on top of MAE reconstruction in
the proposed method. For every batch it:

1. compares embeddings against the current global prototype bank,
2. estimates how ambiguous each prototype assignment is,
3. raises the anchoring threshold for uncertain samples,
4. builds a differentiable gate around that threshold, and
5. penalizes anchored samples when they drift away from their closest
   prototype.

This module therefore contains the logic that decides when the global semantic
memory should influence a local adapter update.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPADLoss(nn.Module):
    """Compute gated prototype anchoring for one mini-batch of embeddings.

    The constructor stores the small set of hyperparameters that govern when a
    sample should trust the global bank:
    1. ``base_tau`` is the similarity floor before uncertainty adjustment,
    2. ``temp_gate`` controls how soft or sharp the sigmoid gate becomes,
    3. ``lambda_entropy`` determines how strongly assignment uncertainty raises
       the threshold,
    4. ``soft_assign_temp`` shapes the prototype-assignment distribution used
       in the entropy calculation, and
    5. ``epsilon`` keeps all logarithms numerically stable.
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
        """Return the batch GPAD loss while preserving gradient flow.

        The forward pass is intentionally simple:
        1. Exit with a graph-connected zero when no prototype bank exists yet.
        2. Measure sample-to-prototype similarity for the whole batch.
        3. Turn the similarity distribution into one adaptive threshold per
           sample.
        4. Compute the final gate that decides how strongly each sample should
           trust the global bank.
        5. Average the gated squared-distance penalty across the batch.
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
        """Build the full cosine-similarity matrix for one batch.

        Both inputs are first normalized onto the unit sphere so a simple matrix
        multiply yields cosine similarity for every sample-prototype pair.
        """
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.mm(normalized_embeddings, normalized_prototypes.t())

    def _compute_adaptive_threshold(self, similarities: torch.Tensor) -> torch.Tensor:
        """Turn prototype-assignment entropy into one threshold per sample.

        The intuition is:
        1. Confident assignments stay close to ``base_tau``.
        2. Ambiguous assignments spread mass over many prototypes.
        3. Higher ambiguity raises the threshold.
        4. Higher thresholds make anchoring more conservative.
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
        """Return the best similarity and the final gate for each sample.

        The gate is built in two parts:
        1. A hard anchor decision that mirrors the routing rule used elsewhere
           in the pipeline.
        2. A smooth sigmoid transition around that threshold so gradients can
           still move samples near the boundary.
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
        """Convert cosine agreement into the final gated distance penalty.

        Higher similarity means lower penalty, and samples with a weak gate
        contribute proportionally less to the final loss.
        """
        squared_distance = 2.0 * (1.0 - max_similarity)
        return (gate * squared_distance).mean()

    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the hard anchored-versus-unanchored routing mask.

        The client uses this helper only for control flow:
        1. Recompute the same similarities used by the differentiable loss.
        2. Recompute the same adaptive threshold.
        3. Mark samples whose best prototype similarity clears that threshold.

        Running under ``torch.no_grad()`` keeps the routing decision lightweight
        while matching the logic of the full forward pass.
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
