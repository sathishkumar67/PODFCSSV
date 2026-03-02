"""
Gated Prototype Anchored Distillation (GPAD) Loss.

This module implements a distillation-style regularization loss for Federated
Continual Self-Supervised Learning.  The loss prevents catastrophic forgetting
by anchoring the client encoder's evolving feature representations to a shared
set of globally aggregated prototype vectors maintained on the server.

Unlike standard knowledge-distillation losses (e.g., KL-divergence on logits),
GPAD operates directly in the L2-normalized feature embedding space and applies
a gated anchoring mechanism that activates ONLY when a prototype match is
confident.  This avoids negative transfer from ambiguous or noisy assignments.

Mathematical Formulation
------------------------
Given one L2-normalized embedding  z ∈ ℝ^D  from the client encoder:

    L_GPAD(z)  =  g(z) · ‖z − v*(z)‖₂

    v*(z) = argmax_{v ∈ V_global}  cos(z, v)
        The nearest global prototype by cosine similarity.

    g(z) = σ((s*(z) − τ(z)) / T_gate) · 𝟙[s*(z) > τ(z)]
        A two-part gate ∈ [0, 1] that fires only when the best-match
        similarity s*(z) exceeds an entropy-adaptive threshold τ(z).

    τ(z) = τ_base + λ_ent · H_norm(z)
        Adaptive threshold.  Rises with the normalized Shannon entropy
        H_norm of the similarity distribution, penalizing equidistant
        (ambiguous) prototype assignments.

    H_norm(z) = H(softmax(sim(z, V) / T_soft)) / log(M)
        Entropy of the temperature-scaled soft assignment distribution,
        divided by the maximum possible entropy log(M) so the penalty
        is invariant to the number of prototypes M.

The batch loss is the mean over all samples:  L = (1/B) Σ_i L_GPAD(z_i).

Design Rationale
----------------
1. **Adaptive Threshold:**  A fixed threshold would either over-anchor
   (forcing alignment to poor matches) or under-anchor (ignoring valid
   matches) depending on the prototype bank's state.  The entropy-adaptive
   threshold automatically becomes stricter when assignments are spread
   across many prototypes and more permissive when one prototype clearly
   dominates.

2. **Hard + Soft Gating:**  The hard binary mask 𝟙[s* > τ] ensures that
   clearly non-matching samples contribute exactly zero loss (no noisy
   gradients), while the sigmoid provides a smooth gradient landscape
   near the decision boundary, avoiding discontinuities during back-
   propagation.

3. **Euclidean Distance on the Unit Sphere:**  Since both z and v* are
   L2-normalized, ‖z − v*‖² = 2(1 − cos(z, v*)).  Using the linear
   (non-squared) distance avoids vanishing gradients as z → v*.

References
----------
[1] Tian et al., "Contrastive Representation Distillation", ICLR 2020.
[2] Li & Hoiem, "Learning without Forgetting", IEEE TPAMI 2018.
[3] McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GPADLoss(nn.Module):
    """Gated Prototype Anchored Distillation (GPAD) Loss.

    Regularizes local feature spaces against a global prototype bank to prevent
    catastrophic forgetting in federated continual learning.  A confidence-gated
    mechanism ensures that only embeddings with an unambiguous prototype match
    receive anchoring forces; ambiguous or truly novel embeddings are left alone.

    The forward pass consists of four sequential stages:

        1. **Similarity**  – Cosine similarity matrix  S ∈ ℝ^{B×M}.
        2. **Threshold**   – Per-sample adaptive threshold τ_i from the
           entropy of each embedding's similarity distribution.
        3. **Gating**      – Hard mask × soft sigmoid → gate g_i ∈ [0, 1].
        4. **Loss**        – Gate-weighted Euclidean distance, averaged.

    Attributes
    ----------
    base_tau : float
        Floor similarity threshold (zero-entropy case).
    temp_gate : float
        Sigmoid temperature controlling gate steepness.
    lambda_entropy : float
        Scaling factor for the entropy-based threshold penalty.
    soft_assign_temp : float
        Temperature for the softmax that produces the entropy distribution.
    epsilon : float
        Numerical stability constant (log, sqrt, normalization).
    """

    def __init__(
        self,
        base_tau: float = 0.5,
        temp_gate: float = 0.1,
        lambda_entropy: float = 0.1,
        soft_assign_temp: float = 0.1,
        epsilon: float = 1e-8,
    ):
        """Initialise the GPAD loss module.

        Parameters
        ----------
        base_tau : float
            Minimum similarity needed for anchoring even when the assignment
            is perfectly unambiguous (entropy = 0).  Higher → stricter.
            Range: 0.3–0.7.  Default: 0.5.
        temp_gate : float
            Sigmoid temperature.  Smaller → sharper (near step-function);
            larger → smoother with more gradient signal at the boundary.
            Range: 0.05–0.5.  Default: 0.1.
        lambda_entropy : float
            Entropy scaling factor.  The effective threshold is
            τ = base_tau + lambda_entropy · H_norm.  Higher → more
            conservative under ambiguous assignments.
            Range: 0.1–0.5.  Default: 0.1.
        soft_assign_temp : float
            Temperature dividing raw similarities before softmax for
            entropy computation:  p = softmax(sim / T).  Lower → peakier
            distribution (low entropy for clear matches, high entropy for
            ambiguous ones).  Range: 0.05–0.5.  Default: 0.1.
        epsilon : float
            Small constant used in three places:
              (a) log-clamping during entropy computation,
              (b) entropy normalisation denominator,
              (c) inside sqrt when converting cosine similarity to distance.
            Default: 1e-8.
        """
        super().__init__()
        self.base_tau = base_tau
        self.temp_gate = temp_gate
        self.lambda_entropy = lambda_entropy
        self.soft_assign_temp = soft_assign_temp
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Forward pass: full four-stage GPAD pipeline
    # ------------------------------------------------------------------
    def forward(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the batch-averaged GPAD distillation loss.

        Parameters
        ----------
        embeddings : torch.Tensor
            Client encoder embeddings, shape ``[B, D]``.  Normalised
            internally — raw vectors are accepted.
        global_prototypes : torch.Tensor
            Global prototype bank from the server, shape ``[M, D]``.

        Returns
        -------
        torch.Tensor
            Scalar loss (0-dim).  Returns a differentiable zero if the
            prototype bank is empty (M = 0).
        """
        # No prototypes → no anchoring possible; return a differentiable
        # zero so that backward() does not break the computation graph.
        if global_prototypes.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Stage 1: full pairwise cosine similarity matrix [B, M].
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)

        # Stage 2: per-sample adaptive thresholds [B].
        tau_adaptive = self._compute_adaptive_threshold(sims)

        # Stage 3: per-sample gate values [B] and best similarities [B].
        max_sim, gate = self._compute_gating(sims, tau_adaptive)

        # Stage 4: gate-weighted Euclidean distance loss → scalar.
        loss = self._compute_anchored_loss(max_sim, gate)

        return loss

    # ------------------------------------------------------------------
    # Stage 1 — Cosine Similarity
    # ------------------------------------------------------------------
    def _compute_similarity_matrix(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """L2-normalise both inputs and compute the cosine similarity matrix.

        After normalisation, cosine similarity reduces to a matrix multiply
        (dot product).  All resulting values lie in [-1, 1], making threshold
        comparisons scale-invariant.

        Parameters
        ----------
        embeddings : torch.Tensor  – shape ``[B, D]``.
        prototypes : torch.Tensor  – shape ``[M, D]``.

        Returns
        -------
        torch.Tensor
            Similarity matrix ``[B, M]``.  Entry (i, j) is cos(z_i, v_j).
        """
        # Project onto the unit hypersphere so dot product = cosine sim.
        z = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        p = F.normalize(prototypes, p=2, dim=1)  # [M, D]

        return torch.mm(z, p.t())  # [B, M]

    # ------------------------------------------------------------------
    # Stage 2 — Adaptive Threshold from Assignment Entropy
    # ------------------------------------------------------------------
    def _compute_adaptive_threshold(self, sims: torch.Tensor) -> torch.Tensor:
        """Derive per-sample thresholds from the entropy of the similarity
        distribution over prototypes.

        High entropy (embedding roughly equidistant to many prototypes)
        → raise the threshold to be more conservative.
        Low entropy (one clear best match) → lower the threshold.

        Computation
        -----------
        1. Temperature-scaled softmax  →  p_j = softmax(s_j / T_soft).
        2. Shannon entropy  H = −Σ p_j · log(p_j).
        3. Normalise to [0, 1]:  H_norm = H / log(M).
        4. Final threshold:  τ_i = base_tau + lambda_entropy · H_norm_i.

        Parameters
        ----------
        sims : torch.Tensor  – shape ``[B, M]``.

        Returns
        -------
        torch.Tensor
            Per-sample thresholds ``[B]`` ∈ [base_tau, base_tau + lambda_entropy].
        """
        B, M = sims.shape

        # Temperature-scaled softmax → probability distribution over protos.
        softmax_all = F.softmax(sims / self.soft_assign_temp, dim=1)  # [B, M]

        # Shannon entropy.  We clamp probabilities from below (instead of
        # adding epsilon) to prevent log(0) without biasing the computation.
        entropy = -torch.sum(
            softmax_all * torch.log(softmax_all.clamp(min=self.epsilon)), dim=1
        )  # [B]

        # Normalise by the maximum possible entropy log(M) (uniform dist).
        max_ent = torch.log(torch.tensor(float(M), device=sims.device))
        ent_norm = entropy / (max_ent + self.epsilon)  # [B], in [0, 1]

        # Adaptive threshold: base floor + entropy-scaled penalty.
        return self.base_tau + self.lambda_entropy * ent_norm

    # ------------------------------------------------------------------
    # Stage 3 — Gating (Hard Mask × Soft Sigmoid)
    # ------------------------------------------------------------------
    def _compute_gating(
        self,
        sims: torch.Tensor,
        tau_adaptive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the two-part confidence gate for each sample.

        The gate combines:
          (a) A **hard binary mask** that zeros out samples whose best-match
              similarity is below the adaptive threshold — prevents any
              gradient leakage from clearly non-matching embeddings.
          (b) A **soft sigmoid weight** that modulates the loss strength
              near the boundary, providing smooth gradients for stable
              training.

        Final gate:  g = sigmoid_weight × hard_mask.

        Parameters
        ----------
        sims         : ``[B, M]``  – cosine similarity matrix.
        tau_adaptive : ``[B]``     – per-sample thresholds.

        Returns
        -------
        max_sim : ``[B]``  – best-match cosine similarity per sample.
        gate    : ``[B]``  – gating weight ∈ [0, 1].
        """
        # Best similarity across all prototypes for each embedding.
        max_sim, _ = sims.max(dim=1)  # [B]

        # Hard mask: strict 0/1 decision.
        is_anchored = (max_sim > tau_adaptive).to(max_sim.dtype)  # [B]

        # Soft sigmoid gate: centred at the threshold, steepness = 1/temp.
        gate_logit = max_sim - tau_adaptive
        gate_sigmoid = torch.sigmoid(gate_logit / self.temp_gate)  # [B]

        # Final gate = sigmoid × hard mask.
        gate = gate_sigmoid * is_anchored  # [B]

        return max_sim, gate

    # ------------------------------------------------------------------
    # Stage 4 — Gate-Weighted Euclidean Distance Loss
    # ------------------------------------------------------------------
    def _compute_anchored_loss(
        self,
        max_sim: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gated Euclidean distance loss for anchored samples.

        For unit-norm vectors the squared Euclidean distance has a closed-form
        relationship with cosine similarity:

            ‖z − v‖² = 2(1 − cos(z, v))
            ‖z − v‖  = √(2(1 − cos(z, v)))

        Using the linear (non-squared) distance ensures that the gradient
        signal does not vanish as z approaches its target prototype.

        Parameters
        ----------
        max_sim : ``[B]``  – best-match cosine similarity.
        gate    : ``[B]``  – per-sample gating weight ∈ [0, 1].

        Returns
        -------
        torch.Tensor
            Scalar mean loss (0-dim tensor).
        """
        # Cosine similarity → squared Euclidean distance on the unit sphere.
        dist_sq = 2.0 * (1.0 - max_sim)  # [B], ∈ [0, 4]

        # Linear distance.  Epsilon inside sqrt guards against dist_sq ≈ 0.
        dist = torch.sqrt(dist_sq + self.epsilon)  # [B], ∈ [0, 2]

        # Gate-weighted distance.  Non-anchored samples (gate ≈ 0) contribute
        # zero loss, avoiding noisy gradient from uncertain assignments.
        loss_per_sample = gate * dist  # [B]

        return loss_per_sample.mean()

    # ==================================================================
    # Anchor Mask — used by FederatedClient for per-embedding routing
    # ==================================================================
    @torch.no_grad()
    def compute_anchor_mask(
        self,
        embeddings: torch.Tensor,
        global_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Classify each embedding as "anchored" (known) or "novel".

        Applies the same similarity + adaptive-threshold logic as the forward
        pass but returns a boolean mask instead of a loss value.  Called by
        ``FederatedClient.train_epoch()`` to route embeddings:

            True  → anchored  → GPAD loss pulls z toward its best global proto.
            False → novel     → routed to local prototype matching / buffer.

        Parameters
        ----------
        embeddings        : ``[B, D]``  – encoder feature embeddings.
        global_prototypes : ``[M, D]``  – current global prototype bank.

        Returns
        -------
        torch.Tensor
            Boolean mask ``[B]``.  All-False when the bank is empty.
        """
        # Empty bank (Round 1): nothing can be anchored.
        if global_prototypes.size(0) == 0:
            return torch.zeros(
                embeddings.size(0), dtype=torch.bool, device=embeddings.device
            )

        # Reuse the same stages as the forward pass for consistency.
        sims = self._compute_similarity_matrix(embeddings, global_prototypes)
        tau_adaptive = self._compute_adaptive_threshold(sims)

        max_sim, _ = sims.max(dim=1)
        return max_sim > tau_adaptive

