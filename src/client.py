"""
Client-Side Components for Federated Continual Self-Supervised Learning.

This module implements the client-side (edge device) logic for the Federated
Learning pipeline. Each client holds a private local dataset, trains an
independent copy of the global model, and communicates only compact prototype
vectors — never raw data — to the server.

Architecture
------------
Two classes compose the client subsystem:

1. ``FederatedClient``: Represents a single participant in the federation.
   Manages local model training, feature extraction, prototype generation,
   online prototype refinement, and the novelty buffer.

2. ``ClientManager``: Factory and dispatch layer that instantiates N clients,
   assigns them to devices, and coordinates training rounds in either
   sequential (CPU) or parallel (multi-GPU) mode.

Training Flow Per Round
-----------------------
Round 1 — Initialization (no global knowledge):
    Loss = MAE reconstruction loss only.
    After training, full K-Means (with ``k_init_prototypes`` clusters) is
    run on all local embeddings to produce the initial local prototypes.

Round > 1 — Continual Learning (global prototypes available):
    Loss = MAE + lambda_proto × GPAD (for anchored embeddings only).
    A per-embedding routing mechanism classifies each embedding as either:
        (a) Anchored  → GPAD loss pulls it toward the best global prototype.
        (b) Non-anchored → Routed to local prototype matching or novelty buffer.

Per-Embedding Routing Decision Flow
------------------------------------
For each embedding z produced by the encoder during training:

    1. Compute cosine similarity to all global prototypes and derive an
       adaptive threshold from the entropy of the similarity distribution.

    2. If max_sim(z, global_bank) > adaptive_threshold:
       → z is **anchored**: Apply GPAD distillation loss.
       → Do NOT update local prototypes (this is a known concept).

    3. If max_sim < adaptive_threshold (non-anchored):
       a. Compare z against all local prototypes.
       b. If max_sim(z, local_protos) > local_update_threshold:
          → **EMA-update** the closest local prototype.
       c. Else:
          → z is **truly novel**: Append to the novelty buffer.

    4. When the novelty buffer reaches ``novelty_buffer_size``:
       → Trigger a fresh K-Means (with ``novelty_k`` clusters) on the buffer.
       → Merge resulting centroids into local prototypes via Merge-or-Add.
       → Clear the buffer.

Prototype Types
---------------
- **Global prototypes**: Maintained by the server; broadcast to clients.
  Used for GPAD anchoring (preventing forgetting of global concepts).
- **Local prototypes**: Maintained per-client; sent to the server each round.
  Summarize the client's private data distribution. Refined online via EMA
  and periodically via buffer K-Means clustering.

References
----------
[1] He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
[2] McMahan et al., "Communication-Efficient Learning of Deep Networks from
    Decentralized Data", AISTATS 2017.
[3] Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from __future__ import annotations
import copy
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================================================
# Logging Configuration
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Simulates a single edge device (client) in the federated network.

    Each client maintains:
    - An independent deep copy of the global model (never shares weights
      directly — only communicates prototypes and final state dicts).
    - A local prototype bank of L2-normalized vectors summarizing its
      private data distribution.
    - A novelty buffer that accumulates truly novel embeddings (those that
      match neither global nor local prototypes) until a fresh K-Means
      clustering is triggered.
    - A local optimizer (default: AdamW) for gradient-based training.

    The client exposes three primary operations per round:
        1. ``train_epoch()``         — One epoch of local SGD with per-embedding routing.
        2. ``generate_prototypes()`` — Round-1 K-Means prototype initialization.
        3. ``get_local_prototypes()``— Retrieve current prototypes (Round > 1).

    Attributes
    ----------
    client_id : int
        Unique numeric identifier for this client (used in logging and
        device assignment).
    device : torch.device
        Hardware device for this client's model and data (e.g., 'cpu',
        'cuda:0').
    dtype : torch.dtype
        Floating-point precision for input tensors (e.g., torch.float32,
        torch.bfloat16).
    model : nn.Module
        Local deep copy of the global model (e.g., ViTMAEForPreTraining
        with IBA adapters).
    optimizer : torch.optim.Optimizer
        Local optimizer instance (AdamW by default).
    local_prototypes : Optional[torch.Tensor]
        Current local prototype matrix [K, D], or None before the first
        call to ``generate_prototypes()``.
    local_update_threshold : float
        Local merge threshold: minimum cosine similarity required to
        trigger an online EMA update of a local prototype.
    local_ema_alpha : float
        EMA interpolation factor for local prototype refinement. Controls
        how quickly local prototypes adapt to new observations.
    lambda_proto : float
        Weight of the GPAD distillation loss in the total training loss:
        total_loss = mae_loss + lambda_proto × gpad_loss.
    novelty_buffer : List[torch.Tensor]
        FIFO buffer of L2-normalized embeddings that failed both global
        anchoring and local prototype matching.
    novelty_buffer_size : int
        Capacity threshold — when the buffer reaches this size, a fresh
        K-Means is triggered.
    novelty_k : int
        Number of clusters for K-Means when processing the novelty buffer.
    kmeans_max_iters : int
        Maximum iteration count for K-Means convergence.
    kmeans_tol : float
        Convergence tolerance — K-Means stops when total centroid shift
        falls below this value.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        optimizer_cls: type = optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        """
        Initialize the federated client.

        Parameters
        ----------
        client_id : int
            Unique identifier for this client. Used for logging, device
            assignment, and tracking across rounds.
        model : nn.Module
            Global model template. A ``copy.deepcopy()`` is created
            internally so that each client trains independently without
            affecting other clients or the global model.
        device : torch.device
            Target hardware device for this client's model and data.
        dtype : torch.dtype
            Floating-point dtype for casting input tensors before feeding
            them to the model.
        optimizer_cls : type
            Optimizer class for local training. Default: ``torch.optim.AdamW``.
        optimizer_kwargs : Dict[str, Any], optional
            Keyword arguments forwarded to the optimizer constructor
            (e.g., ``{"lr": 1e-4, "weight_decay": 0.05}``).
            Default: ``{"lr": 1e-3}`` if not provided.
        local_update_threshold : float
            Local merge threshold. When a non-anchored embedding has cosine
            similarity > this value to its nearest local prototype, that
            prototype receives an EMA update. Embeddings below this
            threshold are sent to the novelty buffer. Range: 0.4–0.8.
            Default: 0.7.
        local_ema_alpha : float
            EMA interpolation factor for local prototype refinement. Used
            in both online non-anchored updates and buffer centroid merges.
            The update rule is: P_new = normalize((1-α)·P_old + α·z).
            Range: 0.05–0.3. Default: 0.1.
        lambda_proto : float
            Weight of the GPAD distillation loss relative to the MAE
            reconstruction loss: total = mae + λ·gpad. Higher values
            enforce stronger alignment to global prototypes at the cost
            of local plasticity. Range: 0.001–0.1. Default: 1.0.
        novelty_buffer_size : int
            Number of truly novel embeddings to accumulate before triggering
            a fresh K-Means clustering to discover new local prototypes.
            Options: 128, 256, 512. Default: 500.
        novelty_k : int
            Number of clusters (K) for K-Means when clustering the novelty
            buffer. Independent of ``k_init_prototypes`` used in Round 1.
            Range: 3–10. Default: 20.
        kmeans_max_iters : int
            Maximum number of iterations for the Spherical K-Means algorithm
            before forced termination. Default: 100.
        kmeans_tol : float
            Convergence tolerance for K-Means. The algorithm terminates
            early when the total Frobenius norm of centroid displacement
            falls below this value. Default: 1e-4.
        """
        self.client_id = client_id
        self.device = device
        self.dtype = dtype
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_tol = kmeans_tol

        # Local prototype bank: shape [K, D], L2-normalized. Populated after
        # the first call to generate_prototypes() and maintained online via
        # EMA updates and buffer clustering thereafter.
        self.local_prototypes: Optional[torch.Tensor] = None

        # Novelty buffer: accumulates L2-normalized embeddings that failed
        # BOTH the global anchor check (non-anchored) AND the local prototype
        # match (below local_update_threshold). When the buffer reaches
        # novelty_buffer_size, a fresh K-Means discovers new concepts.
        self.novelty_buffer: List[torch.Tensor] = []

        # Create an independent deep copy of the global model for this client.
        # Each client trains its own copy — no shared state between clients.
        self.model = copy.deepcopy(model).to(self.device)
        logger.info(f"[Client {self.client_id}] Model copied to {self.device}")

        # Initialize the optimizer over ALL model parameters. In production
        # with IBA adapters, only adapter params have requires_grad=True,
        # so AdamW effectively only updates the adapters.
        opt_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)

        logger.info(
            f"[Client {self.client_id}] Initialized | device={self.device} | "
            f"dtype={self.dtype} | optimizer={optimizer_cls.__name__} | "
            f"ema_alpha={self.local_ema_alpha} | threshold={self.local_update_threshold} | "
            f"lambda_proto={self.lambda_proto} | buffer_size={self.novelty_buffer_size} | "
            f"novelty_k={self.novelty_k}"
        )

    # ======================================================================
    # Feature Extraction
    # ======================================================================
    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature embeddings from the ViT encoder backbone.

        This method accesses ``model.vit`` (the ViTMAEModel encoder inside
        ViTMAEForPreTraining) to obtain the last hidden state, then applies
        global average pooling across the sequence (patch token) dimension to
        produce a single D-dimensional feature vector per input sample.

        The encoder processes images through patch embedding and self-attention,
        returning a sequence of patch token representations. Mean-pooling
        collapses this variable-length sequence into a fixed-size vector
        suitable for prototype comparison and GPAD computation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input image tensor of shape [B, 3, 224, 224], already cast to
            the correct dtype and moved to the correct device.

        Returns
        -------
        torch.Tensor
            Pooled feature embeddings of shape [B, D], where D is the
            encoder's hidden dimension (768 for ViT-Base).
        """
        # Forward through the encoder backbone (not the full MAE model).
        # model.vit returns an object with .last_hidden_state: [B, L, D]
        # where L is the number of patch tokens (+ optional CLS token).
        encoder_output = self.model.vit(inputs)

        # Global average pooling: [B, L, D] → [B, D].
        # This collapses the sequence dimension, producing one feature
        # vector per sample regardless of the number of patch tokens.
        embeddings = encoder_output.last_hidden_state.mean(dim=1)
        return embeddings

    # ======================================================================
    # Local Training
    # ======================================================================
    def train_epoch(
        self,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> float:
        """
        Execute one epoch of local training with per-embedding routing.

        This is the core training method invoked once per communication round.
        It implements the full per-embedding decision flow:

        Step 1 — MAE Forward Pass:
            Feed inputs through the full ViTMAE model to obtain the masked
            autoencoder reconstruction loss (self-supervised objective).

        Step 2 — Feature Extraction:
            Simultaneously extract pooled feature embeddings from the ViT
            encoder for prototype routing and GPAD computation.

        Step 3 — Per-Embedding Routing (Round > 1 only):
            For each embedding in the batch:
            (a) Use the GPAD loss module's ``compute_anchor_mask()`` to
                classify it as anchored (known concept) or non-anchored.
            (b) For anchored embeddings: compute GPAD distillation loss,
                weighted by ``lambda_proto``, and add to the MAE loss.
            (c) For non-anchored embeddings: route to ``_route_non_anchored()``
                for local prototype EMA update or novelty buffer insertion.

        Step 4 — Backward Pass & Optimizer Step:
            Backpropagate the combined loss and update model parameters.

        Loss Composition
        ----------------
        Round 1  (no global prototypes): L = L_MAE
        Round >1 (with global protos):   L = L_MAE + λ · L_GPAD(anchored only)

        Parameters
        ----------
        dataloader : DataLoader
            Iterator over the client's private dataset. Each batch yields
            either a tensor or a tuple/list whose first element is the input.
        global_prototypes : torch.Tensor, optional
            Current global prototype bank from the server, shape [M, D].
            ``None`` during Round 1 (no global knowledge exists yet).
        gpad_loss_fn : nn.Module, optional
            The GPAD distillation loss module. Must implement:
            - ``compute_anchor_mask(embs, protos) → BoolTensor[B]``
            - ``forward(embs, protos) → scalar loss``

        Returns
        -------
        float
            Average total loss over the epoch (scalar, detached).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Determine whether GPAD regularization is active this round.
        has_gpad = (global_prototypes is not None) and (gpad_loss_fn is not None)
        if has_gpad:
            logger.info(
                f"[Client {self.client_id}] Training with MAE + GPAD "
                f"(lambda={self.lambda_proto}, global protos: {global_prototypes.shape[0]})"
            )
        else:
            logger.info(
                f"[Client {self.client_id}] Training with MAE only (no global prototypes)"
            )

        for batch_idx, batch in enumerate(dataloader):
            # --- Data Preparation ---
            # Handle both TensorDataset (returns tuple) and plain tensor datasets.
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            # --- Step 1: MAE Forward Pass ---
            # The full ViTMAE model returns an object with .loss (reconstruction
            # loss) and .hidden_states. For the mock model, .loss is a dummy
            # scalar derived from the encoder output.
            outputs = self.model(inputs)
            mae_loss = getattr(outputs, "loss", None)
            if mae_loss is None:
                # Fallback: if the model doesn't produce a loss attribute,
                # use a differentiable zero to avoid breaking the computation graph.
                mae_loss = torch.tensor(
                    0.0, dtype=self.dtype, device=self.device, requires_grad=True
                )
            final_loss = mae_loss

            # --- Step 2: Feature Extraction ---
            # Extract pooled embeddings from the encoder for prototype routing.
            embeddings = self._extract_features(inputs)

            # --- Step 3: Per-Embedding Routing (active only in Round > 1) ---
            if has_gpad and embeddings is not None:
                # Move global prototypes to this client's device if needed.
                protos_on_device = global_prototypes.to(self.device)

                # 3a. Classify each embedding as anchored (known) or non-anchored (novel)
                # using the GPAD module's adaptive threshold logic.
                anchor_mask = gpad_loss_fn.compute_anchor_mask(
                    embeddings, protos_on_device
                )  # [B] boolean mask

                # 3b. GPAD distillation loss — computed ONLY for anchored embeddings.
                # These are "known" concepts that should stay aligned with the
                # global prototype bank to prevent catastrophic forgetting.
                anchored_embs = embeddings[anchor_mask]
                if anchored_embs.shape[0] > 0:
                    gpad_loss = gpad_loss_fn(anchored_embs, protos_on_device)
                    final_loss = final_loss + self.lambda_proto * gpad_loss

                # 3c. Non-anchored embeddings — these did NOT match any global
                # prototype confidently. Route them through local prototype
                # matching and the novelty buffer for concept discovery.
                non_anchored_embs = embeddings[~anchor_mask]
                if non_anchored_embs.shape[0] > 0:
                    self._route_non_anchored(non_anchored_embs)

            # --- Step 4: Backward Pass ---
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(
            f"[Client {self.client_id}] Epoch complete | avg_loss={avg_loss:.6f} | "
            f"batches={num_batches} | buffer_size={len(self.novelty_buffer)}"
        )
        return avg_loss

    # ======================================================================
    # Per-Embedding Routing: Local Prototype Update & Novelty Buffer
    # ======================================================================
    @torch.no_grad()
    def _route_non_anchored(self, embeddings: torch.Tensor) -> None:
        """
        Route non-anchored embeddings through local prototype matching.

        This method implements the second stage of the per-embedding routing
        decision flow. For each non-anchored embedding (those that did NOT
        match any global prototype), we check whether it matches a LOCAL
        prototype:

        Case A — No local prototypes exist yet:
            All embeddings are appended to the novelty buffer. This occurs
            at the start of training before ``generate_prototypes()`` has
            been called.

        Case B — Local prototypes exist:
            For each embedding z:
            - If cos(z, nearest_local_proto) > ``local_update_threshold``:
                → EMA-update the nearest local prototype:
                  P_new = normalize((1 - α)·P_old + α·z)
                This refines existing local concepts with new evidence.
            - Else:
                → z does not match any known concept (global or local).
                  Append it to the novelty buffer for future clustering.

        After processing all embeddings, check whether the novelty buffer
        has reached its capacity. If so, trigger K-Means clustering.

        Parameters
        ----------
        embeddings : torch.Tensor
            Batch of non-anchored embeddings from the current training batch,
            shape [B', D] where B' ≤ B.
        """
        # L2-normalize and detach from the computation graph (this runs
        # under @torch.no_grad() so no gradients are tracked).
        z_norm = F.normalize(embeddings.detach(), p=2, dim=1)  # [B', D]

        # --- Case A: No local prototypes yet → buffer everything ---
        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            for i in range(z_norm.shape[0]):
                self.novelty_buffer.append(z_norm[i].clone())
            self._maybe_cluster_buffer()
            return

        # --- Case B: Compare against existing local prototypes ---
        # Ensure local prototypes are on the same device as the embeddings.
        if self.local_prototypes.device != self.device:
            self.local_prototypes = self.local_prototypes.to(self.device)

        # L2-normalize local prototypes for valid cosine similarity.
        p_norm = F.normalize(self.local_prototypes, p=2, dim=1)  # [K_local, D]

        # Full cosine similarity matrix between embeddings and local prototypes.
        sims = torch.mm(z_norm, p_norm.t())  # [B', K_local]

        # Find the nearest local prototype for each embedding.
        max_sim, best_idx = sims.max(dim=1)  # both [B']

        for i in range(z_norm.shape[0]):
            if max_sim[i] > self.local_update_threshold:
                # LOCAL MATCH: The embedding is close enough to an existing
                # local prototype → refine that prototype via EMA blending.
                proto_idx = best_idx[i]
                old_proto = self.local_prototypes[proto_idx]
                blended = (
                    (1 - self.local_ema_alpha) * old_proto
                    + self.local_ema_alpha * z_norm[i]
                )
                # Re-normalize: the EMA blend of two unit vectors is NOT a
                # unit vector (its norm < 1), so project back to the sphere.
                self.local_prototypes[proto_idx] = F.normalize(
                    blended, p=2, dim=0
                )
            else:
                # TRULY NOVEL: This embedding matches neither global nor local
                # prototypes — it may represent a new visual concept. Append
                # to the novelty buffer for later K-Means clustering.
                self.novelty_buffer.append(z_norm[i].clone())

        # Check if the buffer has reached capacity and trigger clustering.
        self._maybe_cluster_buffer()

    def _maybe_cluster_buffer(self) -> None:
        """
        Check if the novelty buffer has reached its capacity threshold.

        If the buffer contains at least ``novelty_buffer_size`` embeddings,
        trigger a fresh K-Means clustering to extract new local prototype
        candidates and merge or add them to the local prototype bank.
        """
        if len(self.novelty_buffer) >= self.novelty_buffer_size:
            logger.info(
                f"[Client {self.client_id}] Novelty buffer full "
                f"({len(self.novelty_buffer)} >= {self.novelty_buffer_size}). "
                f"Triggering fresh K-Means clustering..."
            )
            self._cluster_novelty_buffer()

    @torch.no_grad()
    def _cluster_novelty_buffer(self) -> None:
        """
        Cluster the novelty buffer via K-Means and integrate results into
        local prototypes using a Merge-or-Add strategy.

        This method is automatically invoked when the novelty buffer reaches
        ``novelty_buffer_size``. It performs the following steps:

        1. Stack buffered embeddings into a tensor [N_buffer, D] and L2-normalize.
        2. Run Spherical K-Means with K = min(novelty_k, N_buffer) clusters.
        3. For each resulting centroid:
           - If it is similar to an existing local prototype (cosine sim >
             ``local_update_threshold``): **Merge** via EMA update.
           - If it is dissimilar to all existing prototypes: **Add** as a
             new local prototype.
        4. Clear the novelty buffer.

        This Merge-or-Add strategy prevents duplicate prototypes while allowing
        the local bank to grow organically as new visual concepts are discovered.
        """
        if len(self.novelty_buffer) == 0:
            return

        # Stack the buffer list into a single tensor and L2-normalize.
        buffer_tensor = torch.stack(self.novelty_buffer, dim=0).to(self.device)
        buffer_tensor = F.normalize(buffer_tensor, p=2, dim=1)  # [N_buffer, D]

        # Clamp K to the number of available samples (buffer may be smaller
        # than novelty_k on the very first trigger).
        K = min(self.novelty_k, buffer_tensor.shape[0])

        logger.info(
            f"[Client {self.client_id}] Clustering novelty buffer: "
            f"{buffer_tensor.shape[0]} samples → K={K}"
        )

        # Run Spherical K-Means on the buffered novel embeddings.
        new_centroids = self._kmeans(buffer_tensor, K=K)  # [K, D]

        # --- Merge-or-Add new centroids into local prototypes ---
        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            # First time: all centroids become the initial local prototypes.
            self.local_prototypes = new_centroids
            merged_count = 0
            added_count = K
        else:
            # Ensure local prototypes are on the correct device.
            if self.local_prototypes.device != self.device:
                self.local_prototypes = self.local_prototypes.to(self.device)

            # L2-normalize both sets for valid cosine similarity.
            p_norm = F.normalize(self.local_prototypes, p=2, dim=1)
            c_norm = F.normalize(new_centroids, p=2, dim=1)

            # Full similarity matrix: [K_new, K_existing]
            sims = torch.mm(c_norm, p_norm.t())
            max_sim, best_idx = sims.max(dim=1)  # both [K_new]

            # Clone existing prototypes for safe in-place EMA updates.
            updated_protos = self.local_prototypes.clone()
            protos_to_add = []
            merged_count = 0
            added_count = 0

            for i in range(new_centroids.shape[0]):
                if max_sim[i] > self.local_update_threshold:
                    # MERGE: This centroid is similar to an existing local
                    # prototype → update the existing one via EMA.
                    idx = best_idx[i]
                    old_proto = updated_protos[idx]
                    blended = (
                        (1 - self.local_ema_alpha) * old_proto
                        + self.local_ema_alpha * new_centroids[i]
                    )
                    # Re-normalize: EMA blend is not a unit vector.
                    updated_protos[idx] = F.normalize(blended, p=2, dim=0)
                    merged_count += 1
                else:
                    # ADD: This centroid represents a genuinely new concept
                    # not captured by any existing local prototype.
                    protos_to_add.append(new_centroids[i])
                    added_count += 1

            # Combine the EMA-updated existing prototypes with newly added ones.
            if len(protos_to_add) > 0:
                new_stack = torch.stack(protos_to_add, dim=0)
                self.local_prototypes = torch.cat(
                    [updated_protos, new_stack], dim=0
                )
            else:
                self.local_prototypes = updated_protos

        # Clear the novelty buffer — all information has been captured
        # in the updated local prototypes.
        self.novelty_buffer.clear()

        logger.info(
            f"[Client {self.client_id}] Buffer clustered → "
            f"Merged {merged_count}, Added {added_count} -> "
            f"Total {self.local_prototypes.shape[0]} local protos"
        )

    def get_local_prototypes(self) -> Optional[torch.Tensor]:
        """
        Return the current local prototypes without re-clustering.

        Used by the orchestrator in Round > 1 to collect the prototypes that
        have been maintained via online EMA updates and buffer clustering
        throughout training, rather than running a full post-training K-Means.

        Returns
        -------
        Optional[torch.Tensor]
            Local prototype matrix of shape [K, D], L2-normalized. Returns
            ``None`` if prototypes have not been initialized yet.
        """
        return self.local_prototypes

    # ======================================================================
    # Round-1 Prototype Generation (Post-Training K-Means)
    # ======================================================================
    @torch.no_grad()
    def generate_prototypes(
        self, dataloader: DataLoader, K_init: int = 10
    ) -> torch.Tensor:
        """
        Generate initial local prototypes by clustering the client's encoder
        feature space with Spherical K-Means.

        This method is called ONLY in Round 1, after the initial MAE-only
        training epoch. It creates the first set of local prototypes that
        summarize the client's private data distribution. In subsequent rounds,
        prototypes are maintained online via EMA updates and buffer clustering.

        Pipeline:
            1. Feature Extraction: Run the ViT encoder on the entire local
               dataset, collecting pooled embeddings [N, D].
            2. L2 Normalization: Project all embeddings onto the unit
               hypersphere so that cosine similarity = dot product.
            3. Spherical K-Means: Partition the N embeddings into K_init
               clusters. The resulting K centroids are the local prototypes.
            4. Store Prototypes: Save a detached clone for online EMA updates
               in subsequent training rounds.

        Parameters
        ----------
        dataloader : DataLoader
            Iterator over the client's private dataset.
        K_init : int
            Number of prototype clusters to generate. This is the
            ``k_init_prototypes`` hyperparameter from CONFIG.
            Range: 5–50. Default: 10.

        Returns
        -------
        torch.Tensor
            Local prototype matrix of shape [K_init, D], L2-normalized.
        """
        self.model.eval()
        all_features = []

        logger.info(
            f"[Client {self.client_id}] Extracting features for prototype generation..."
        )

        # Step 1: Feature Extraction — run encoder on all local data.
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            with torch.inference_mode():
                features = self._extract_features(inputs)

            all_features.append(features)

        # Concatenate all batch features: [N_total, D]
        embeddings = torch.cat(all_features, dim=0)
        logger.info(
            f"[Client {self.client_id}] Extracted {embeddings.shape[0]} "
            f"embeddings of dim {embeddings.shape[1]}"
        )

        # Step 2: L2 Normalization — project onto unit hypersphere.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 3: Spherical K-Means clustering.
        centroids = self._kmeans(embeddings, K=K_init)
        logger.info(
            f"[Client {self.client_id}] K-Means complete | K={K_init} | "
            f"centroid shape={centroids.shape}"
        )

        # Step 4: Store for online EMA updates in subsequent rounds.
        self.local_prototypes = centroids.detach().clone()

        return centroids

    # ======================================================================
    # Spherical K-Means Clustering (Pure PyTorch Implementation)
    # ======================================================================
    def _kmeans(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """
        Spherical K-Means clustering on L2-normalized embeddings.

        Since all input embeddings lie on the unit hypersphere, cosine
        similarity (equivalently, dot product for unit-norm vectors) serves
        as the natural similarity metric. This variant is often called
        "Spherical K-Means" and is the standard choice for clustering
        normalized feature representations.

        Algorithm:
            1. **Initialization**: Randomly select K data points as initial
               centroids (Forgy initialization).
            2. **Assignment**: Compute cosine similarity between every data
               point and all K centroids via matrix multiplication. Assign
               each point to its most similar centroid.
            3. **Update**: Recompute each centroid as the arithmetic mean of
               all points assigned to it, then L2-normalize back to the unit
               sphere.
            4. **Convergence**: Stop when the total centroid displacement
               (Frobenius norm of the centroid shift matrix) falls below
               ``self.kmeans_tol``, or when ``self.kmeans_max_iters`` is
               reached.
            5. **Empty Cluster Recovery**: If any cluster loses all its
               members (degenerate solution), re-seed its centroid with a
               randomly selected data point.

        Parameters
        ----------
        X : torch.Tensor
            Input data matrix, shape [N, D]. Assumed to be L2-normalized
            (all rows have unit norm).
        K : int
            Number of clusters (prototypes) to form. Clamped to N if the
            dataset has fewer than K samples.

        Returns
        -------
        torch.Tensor
            Final centroid matrix of shape [K, D], L2-normalized.
        """
        N, D = X.shape

        # Clamp K to the number of data points to avoid out-of-bounds
        # errors when the dataset is very small (e.g., novelty buffer
        # triggers with fewer samples than novelty_k).
        K = min(K, N)

        # Step 1: Forgy initialization — randomly select K data points.
        indices = torch.randperm(N, device=X.device)[:K]
        centroids = X[indices].clone()  # [K, D]

        for iteration in range(self.kmeans_max_iters):
            # Ensure centroids are on the unit sphere before computing
            # similarities. This is redundant in iteration 0 (they come
            # from normalized data) but necessary after the mean update step.
            centroids = F.normalize(centroids, p=2, dim=1)

            # Step 2: Assignment — cosine similarity via dot product.
            # sims[i, k] = cos(X[i], centroids[k]) for all i, k.
            sims = torch.mm(X, centroids.t())  # [N, K]

            # Assign each point to the centroid with the highest similarity.
            _, labels = sims.max(dim=1)  # [N]

            # Step 3: Update — recompute centroids as cluster means.
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                cluster_mask = labels == k
                if cluster_mask.sum() > 0:
                    # Arithmetic mean of all points in cluster k.
                    new_centroids[k] = X[cluster_mask].mean(dim=0)
                else:
                    # Empty cluster recovery: re-seed with a random data point
                    # to avoid degenerate solutions with fewer than K clusters.
                    new_idx = torch.randint(0, N, (1,), device=X.device).item()
                    new_centroids[k] = X[new_idx]

            # Step 4: Convergence check — total centroid displacement.
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if center_shift < self.kmeans_tol:
                logger.info(
                    f"[Client {self.client_id}] K-Means converged at "
                    f"iteration {iteration + 1}"
                )
                break

        # Final L2-normalization to ensure all centroids lie exactly on
        # the unit sphere (the mean step can produce sub-unit-norm vectors).
        return F.normalize(centroids, p=2, dim=1)


class ClientManager:
    """
    Factory and dispatch manager for federated client instances.

    In a real-world deployment, each client would reside on a separate
    physical device. This class simulates that by:
    1. Instantiating N ``FederatedClient`` objects, each with a deep copy
       of the global model and its own device assignment.
    2. Dispatching training commands to all clients per round, either
       sequentially (CPU) or in parallel (multi-GPU).

    Execution Modes
    ---------------
    **Parallel (Multi-GPU)**: When ``gpu_count > 0``, each client is
    assigned to a dedicated GPU (strict 1:1 mapping: Client i → cuda:i).
    Training is executed concurrently via ``ThreadPoolExecutor``. PyTorch
    releases the GIL during CUDA kernel execution, enabling true parallelism
    across GPUs.

    **Sequential (CPU)**: When ``gpu_count == 0``, all clients share the
    CPU and are trained one after another. Threading is not used because
    Python's GIL prevents true CPU parallelism and the threading overhead
    would degrade performance.

    Attributes
    ----------
    clients : List[FederatedClient]
        List of instantiated client objects, one per simulated edge device.
    num_clients : int
        Total number of clients in the federation.
    gpu_count : int
        Number of available GPUs (0 = CPU-only sequential mode).
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_clients: int,
        gpu_count: int = 0,
        dtype: torch.dtype = torch.float32,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        """
        Initialize the client manager and spawn all federated clients.

        Parameters
        ----------
        base_model : nn.Module
            Global model template. Each client receives an independent
            deep copy of this model.
        num_clients : int
            Number of federated clients to simulate.
        gpu_count : int
            Number of available GPUs. If > 0, enforces strict 1:1
            Client-GPU mapping (raises ValueError if num_clients != gpu_count).
            Default: 0 (CPU mode).
        dtype : torch.dtype
            Floating-point dtype for input tensors. Default: torch.float32.
        optimizer_kwargs : Dict[str, Any], optional
            Keyword arguments forwarded to each client's optimizer constructor
            (e.g., ``{"lr": 1e-4, "weight_decay": 0.05}``).
        local_update_threshold : float
            Local merge threshold for online EMA prototype updates.
            Range: 0.4–0.8. Default: 0.7.
        local_ema_alpha : float
            EMA interpolation factor for local prototype refinement.
            Range: 0.05–0.3. Default: 0.1.
        lambda_proto : float
            GPAD loss weight. Range: 0.001–0.1. Default: 1.0.
        novelty_buffer_size : int
            Buffer capacity before triggering K-Means.
            Options: 128, 256, 512. Default: 500.
        novelty_k : int
            K for buffer K-Means clustering. Range: 3–10. Default: 20.
        kmeans_max_iters : int
            Maximum K-Means iterations. Default: 100.
        kmeans_tol : float
            K-Means convergence tolerance. Default: 1e-4.
        """
        self.clients: List[FederatedClient] = []
        self.num_clients = num_clients
        self.gpu_count = gpu_count
        self.dtype = dtype
        self.optimizer_kwargs = optimizer_kwargs
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k
        self.kmeans_max_iters = kmeans_max_iters
        self.kmeans_tol = kmeans_tol

        self._initialize_clients(base_model)

    def _initialize_clients(self, base_model: nn.Module) -> None:
        """
        Instantiate and configure all federated clients.

        Device Assignment:
            - GPU mode: Client i → ``cuda:i`` (strict 1:1 mapping).
            - CPU mode: All clients → ``cpu``.

        Parameters
        ----------
        base_model : nn.Module
            Global model template to deep-copy for each client.
        """
        logger.info(
            f"[ClientManager] Initializing {self.num_clients} clients..."
        )

        # Enforce strict 1:1 Client-GPU mapping in multi-GPU mode.
        if self.gpu_count > 0:
            if self.num_clients != self.gpu_count:
                raise ValueError(
                    f"Strict 1:1 Client-GPU mapping required. "
                    f"Requested {self.num_clients} clients but found "
                    f"{self.gpu_count} GPUs."
                )
            logger.info(
                f"[ClientManager] Parallel Mode: {self.num_clients} clients "
                f"-> {self.gpu_count} GPUs"
            )
        else:
            logger.info(
                f"[ClientManager] Sequential Mode: {self.num_clients} "
                f"clients on CPU"
            )

        for i in range(self.num_clients):
            # Assign device: cuda:i for GPU mode, cpu for CPU mode.
            device = (
                torch.device(f"cuda:{i}")
                if self.gpu_count > 0
                else torch.device("cpu")
            )

            # Create a FederatedClient with all hyperparameters forwarded
            # from the manager (which received them from CONFIG).
            client = FederatedClient(
                client_id=i,
                model=base_model,
                device=device,
                dtype=self.dtype,
                optimizer_kwargs=self.optimizer_kwargs,
                local_update_threshold=self.local_update_threshold,
                local_ema_alpha=self.local_ema_alpha,
                lambda_proto=self.lambda_proto,
                novelty_buffer_size=self.novelty_buffer_size,
                novelty_k=self.novelty_k,
                kmeans_max_iters=self.kmeans_max_iters,
                kmeans_tol=self.kmeans_tol,
            )
            self.clients.append(client)

        logger.info(
            f"[ClientManager] All {self.num_clients} clients initialized "
            f"successfully."
        )

    def train_round(
        self,
        dataloaders: List[DataLoader],
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> List[float]:
        """
        Dispatch one round of local training to ALL clients.

        In GPU mode, training is parallelized across GPUs using
        ``ThreadPoolExecutor`` (PyTorch releases the GIL during CUDA kernel
        execution, enabling true parallelism). In CPU mode, clients are
        trained sequentially to avoid GIL contention overhead.

        Parameters
        ----------
        dataloaders : List[DataLoader]
            One DataLoader per client. The list length MUST equal
            ``num_clients`` (raises ValueError otherwise).
        global_prototypes : torch.Tensor, optional
            Current global prototype bank for GPAD loss, shape [M, D].
            ``None`` during Round 1.
        gpad_loss_fn : nn.Module, optional
            The GPAD distillation loss module. ``None`` during Round 1.

        Returns
        -------
        List[float]
            Average training loss for each client, ordered by client_id.
            Failed clients have ``float('nan')`` as their loss.

        Raises
        ------
        ValueError
            If ``len(dataloaders) != num_clients``.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Dataloader count ({len(dataloaders)}) does not match "
                f"client count ({self.num_clients})"
            )

        round_losses = [0.0] * self.num_clients

        if self.gpu_count > 0:
            # --- Parallel execution: one thread per GPU ---
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                logger.info(
                    f"[ClientManager] Spawning {self.num_clients} training "
                    f"threads (1 per GPU)..."
                )
                futures = {}
                for i, client in enumerate(self.clients):
                    futures[
                        executor.submit(
                            client.train_epoch,
                            dataloaders[i],
                            global_prototypes=global_prototypes,
                            gpad_loss_fn=gpad_loss_fn,
                        )
                    ] = i

                for future in futures:
                    client_idx = futures[future]
                    try:
                        loss = future.result()
                        round_losses[client_idx] = loss
                        logger.info(
                            f"[ClientManager] Client {client_idx} "
                            f"(GPU {client_idx}) finished | loss={loss:.4f}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[ClientManager] Client {client_idx} FAILED: {e}"
                        )
                        round_losses[client_idx] = float("nan")
        else:
            # --- Sequential execution on CPU ---
            logger.info(
                f"[ClientManager] Running sequential training on CPU for "
                f"{self.num_clients} clients..."
            )
            for i, client in enumerate(self.clients):
                try:
                    loss = client.train_epoch(
                        dataloaders[i],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn,
                    )
                    round_losses[i] = loss
                    logger.info(
                        f"[ClientManager] Client {i} (CPU) finished | "
                        f"loss={loss:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[ClientManager] Client {i} FAILED: {e}"
                    )
                    round_losses[i] = float("nan")

        return round_losses