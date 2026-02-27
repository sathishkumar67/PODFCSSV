"""
Server-Side Components for Federated Continual Self-Supervised Learning.

This module implements the two core server-side responsibilities in the
Federated Learning (FL) pipeline:

1. **Global Prototype Bank** (``GlobalPrototypeBank``):
   Maintains a dynamically growing set of L2-normalized prototype vectors
   that represent the visual concepts discovered across all clients. New
   local prototypes from each round are integrated via a "Merge-or-Add"
   strategy with Exponential Moving Average (EMA) updates.

2. **Federated Model Aggregation** (``FederatedModelServer``):
   Implements the standard Federated Averaging (FedAvg) algorithm to
   combine client model weights into a single global consensus model.

3. **Round Orchestration** (``run_server_round``):
   A convenience function that executes both prototype merging and weight
   aggregation for a single communication round.

4. **Global Model Wrapper** (``GlobalModel``):
   Encapsulates the server-side model instance, handling initialization
   from pretrained checkpoints and weight updates from FedAvg.

Prototype Bank Design
---------------------
The Global Prototype Bank is the shared knowledge base that enables
continual learning across the federation. Unlike approaches that fix the
number of prototypes a priori, our bank grows organically:

- **Merge**: When an incoming local prototype has cosine similarity ≥
  ``merge_threshold`` to an existing global prototype, the global prototype
  is updated in-place via EMA:  G_new = normalize((1-α)·G_old + α·P_local).

- **Add**: When similarity is below the threshold, the incoming prototype
  represents a genuinely novel visual concept and is appended to the bank
  (subject to the ``max_prototypes`` capacity limit).

This design allows the system to autonomously discover new classes or
features while refining existing ones, without requiring any class labels
or a predefined number of categories.

References
----------
[1] McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.
[2] Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional, Any, Tuple

import torch
import torch.nn.functional as F

# ==========================================================================
# Logging Configuration
# ==========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class GlobalPrototypeBank:
    """
    Server-side global prototype bank for federated prototype aggregation.

    This class manages the lifecycle of the global prototype bank — a
    dynamically sized collection of L2-normalized vectors on the unit
    hypersphere. Each vector represents a distinct visual concept (cluster
    centroid) discovered across the federation.

    The bank implements an online "Merge-or-Add" strategy:
        1. For each incoming local prototype, compute its cosine similarity
           against ALL existing global prototypes.
        2. If the best match exceeds ``merge_threshold`` → **Merge** via EMA.
        3. Otherwise → **Add** as a new global prototype (if below capacity).

    All prototype vectors are always kept L2-normalized to ensure that cosine
    similarity equals the dot product, enabling efficient similarity computation.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the feature embedding space (D).
    merge_threshold : float
        Server-side cosine similarity threshold for merging. If the best
        match between an incoming prototype and the bank exceeds this value,
        the existing prototype is updated via EMA. Otherwise, the incoming
        prototype is added as a new entry.
    ema_alpha : float
        Server-side EMA interpolation factor for prototype updates. Controls
        how quickly global prototypes adapt to new observations.
    device : torch.device
        Computation device for prototype tensors.
    max_prototypes : int
        Maximum number of prototypes the bank can hold. Once at capacity,
        novel prototypes are rejected (but merges still occur).
    prototypes : torch.Tensor
        The prototype matrix of shape [M, D], where M is the current number
        of prototypes. All rows are L2-normalized.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        merge_threshold: float = 0.8,
        ema_alpha: float = 0.1,
        device: str = "cpu",
        max_prototypes: int = 50,
    ) -> None:
        """
        Initialize the global prototype bank.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the feature vectors. Must match the encoder's
            output dimension (e.g., 768 for ViT-Base, 32 for mock model).
            Default: 768.
        merge_threshold : float
            Server-side global merge threshold. When the cosine similarity
            between an incoming local prototype and its best-match global
            prototype equals or exceeds this value, the global prototype is
            updated via EMA instead of adding a new entry. Higher values
            make the bank grow faster (more new prototypes), while lower
            values consolidate more aggressively. Range: 0.5–0.85.
            Default: 0.8.
        ema_alpha : float
            Exponential Moving Average interpolation factor for updating
            global prototypes. The update rule is:
                G_new = normalize( (1 - α) · G_old + α · P_incoming )
            Small α (e.g., 0.01): Global prototypes change slowly, providing
            high stability across rounds but slower adaptation.
            Large α (e.g., 0.2): Faster adaptation to recent client data
            but more susceptible to noise from individual clients.
            Range: 0.01–0.2. Default: 0.1.
        device : str
            Target computation device ('cpu', 'cuda', or 'cuda:N').
            Default: 'cpu'.
        max_prototypes : int
            Maximum capacity of the global prototype bank. Once the bank
            reaches this size, no new prototypes can be added — only EMA
            merges into existing prototypes are allowed. This prevents
            unbounded bank growth in long-running or highly heterogeneous
            federations. Range: 20–200. Default: 50.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        self.max_prototypes = max_prototypes

        # Initialize an empty prototype matrix. Shape: [0, D].
        # This will grow as prototypes are added during aggregation.
        self.prototypes = torch.zeros(0, embedding_dim, device=self.device)

    @torch.no_grad()
    def merge_local_prototypes(
        self,
        local_protos_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Integrate local prototypes from all clients into the global bank.

        This is the core aggregation step executed once per server round.
        All local prototype tensors from participating clients are concatenated,
        L2-normalized, and then processed sequentially against the current
        global bank using the Merge-or-Add strategy.

        Sequential processing (rather than batch parallel) is required because
        each incoming prototype can modify the bank state (via merge or add),
        and subsequent prototypes must see the updated bank. For example, two
        clients might independently discover the same novel concept — the first
        one adds it, and the second should merge into it rather than creating a
        duplicate.

        Parameters
        ----------
        local_protos_list : List[torch.Tensor]
            A list of prototype tensors, one per client. Each tensor has shape
            [K_i, D] where K_i is the number of prototypes from client i and
            D is the embedding dimension. Tensors do not need to be
            pre-normalized — normalization is applied internally.

        Returns
        -------
        torch.Tensor
            The updated global prototype bank of shape [M_new, D], where
            M_new >= M_old (bank can only grow or stay the same size, never
            shrink). All rows are L2-normalized.
        """
        # Early exit if no client submitted prototypes this round.
        if not local_protos_list:
            return self.prototypes

        # Concatenate all local prototypes into a single batch [N_total, D]
        # and move to the bank's device for computation.
        incoming = torch.cat(local_protos_list, dim=0).to(self.device)

        # L2-normalize incoming prototypes so that dot product == cosine
        # similarity. This is essential for valid threshold comparisons.
        incoming = F.normalize(incoming, p=2, dim=1)

        # Process each incoming prototype sequentially against the bank.
        for i in range(incoming.size(0)):
            p_new = incoming[i]  # Single prototype vector, shape [D]

            # --- Special case: bank is empty (first round initialization) ---
            # Simply add the first prototype to bootstrap the bank.
            if self.prototypes.size(0) == 0:
                self.prototypes = p_new.unsqueeze(0)  # [1, D]
                continue

            # Defensive re-normalization of the bank before similarity
            # computation. EMA updates can cause slight norm drift due to
            # floating-point precision, so we project back to the unit sphere.
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            # Compute cosine similarity between p_new and ALL global prototypes.
            # Since both are unit-norm, this is a simple matrix-vector product.
            # Result shape: [M] — one similarity score per global prototype.
            sims = torch.mv(self.prototypes, p_new)

            # Identify the best-matching global prototype.
            max_sim, best_idx = sims.max(dim=0)

            # --- Merge-or-Add Decision ---
            if max_sim >= self.merge_threshold:
                # MERGE: The incoming prototype matches an existing concept.
                # Update the best-match global prototype via EMA blending.
                old_vec = self.prototypes[best_idx]
                blended = (1 - self.ema_alpha) * old_vec + self.ema_alpha * p_new

                # Re-normalize immediately: the EMA blend of two unit vectors
                # is NOT a unit vector (its norm < 1), so we must project back
                # onto the unit sphere to maintain the L2-norm invariant.
                self.prototypes[best_idx] = F.normalize(blended, p=2, dim=0)
            else:
                # ADD: The incoming prototype represents a novel concept not
                # yet captured by any existing global prototype.
                # Append it to the bank only if the capacity limit allows.
                if self.max_prototypes is None or self.prototypes.size(0) < self.max_prototypes:
                    self.prototypes = torch.cat(
                        [self.prototypes, p_new.unsqueeze(0)], dim=0
                    )
                else:
                    logger.info(
                        f"Global bank at capacity ({self.max_prototypes}). "
                        f"Skipping novel prototype."
                    )

        return self.prototypes

    def get_prototypes(self) -> torch.Tensor:
        """
        Retrieve the current state of the global prototype bank.

        Returns
        -------
        torch.Tensor
            The prototype matrix of shape [M, D], L2-normalized. Returns an
            empty tensor [0, D] if no prototypes have been added yet.
        """
        return self.prototypes


class FederatedModelServer:
    """
    Federated Averaging (FedAvg) weight aggregation server.

    This class implements the server-side model aggregation step of the FedAvg
    algorithm. After each communication round, every participating client sends
    its locally-updated model weights (state dict) to the server. This class
    computes the element-wise arithmetic mean of all client weights to produce
    a single global consensus model.

    The uniform averaging assumes equal data contribution from all clients.
    In future extensions, this can be replaced with weighted averaging based
    on local dataset sizes (as in the original FedAvg paper).

    Design Notes
    ------------
    - All weight tensors are cast to float32 before averaging to prevent
      overflow/underflow errors when using reduced-precision dtypes (e.g.,
      bfloat16) across many clients.
    - If a parameter key is missing from any client's state dict, that
      parameter is skipped entirely (with a warning) rather than averaging
      over a partial subset, which would introduce bias.
    """

    @torch.no_grad()
    def aggregate_weights(
        self,
        client_weights_map: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model weights from all participating clients via FedAvg.

        For each parameter key (layer name) present in ALL client state dicts,
        the corresponding tensors are stacked along a new batch dimension and
        averaged. The result is a global state dict that can be loaded directly
        into the model via ``model.load_state_dict()``.

        Parameters
        ----------
        client_weights_map : Dict[str, Dict[str, torch.Tensor]]
            Mapping from client ID (str) to that client's model state dict.
            Each state dict maps parameter names (e.g., 'encoder.weight') to
            weight tensors. All clients must share the same model architecture.

        Returns
        -------
        Dict[str, torch.Tensor]
            The aggregated global state dict. Each tensor is the element-wise
            mean of the corresponding tensors from all clients. Returns an
            empty dict if no client weights were received.
        """
        if not client_weights_map:
            logger.warning("No client weights received for aggregation.")
            return {}

        client_ids = list(client_weights_map.keys())
        num_clients = len(client_ids)

        # Optimization: with only one client, no averaging is needed — just
        # return that client's weights directly.
        if num_clients == 1:
            return client_weights_map[client_ids[0]]

        # Use the first client's state dict as a structural template to
        # define the set of parameter keys we expect from all clients.
        reference_state = client_weights_map[client_ids[0]]
        reference_keys = reference_state.keys()

        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            # Collect this parameter's tensor from every client. We cast to
            # float32 for high-precision averaging, which is critical when
            # the model uses bfloat16 or float16 (reduced precision can cause
            # significant rounding errors when summing many tensors).
            tensors = []
            for cid in client_ids:
                if key in client_weights_map[cid]:
                    tensors.append(client_weights_map[cid][key].float())

            if len(tensors) == num_clients:
                # All clients contributed this parameter → compute the mean.
                # Stack into [N_clients, *param_shape] then average along dim=0.
                stacked = torch.stack(tensors, dim=0)
                global_state[key] = stacked.mean(dim=0)
            else:
                # Some clients are missing this parameter (architecture mismatch
                # or partial state dict). Skip to avoid biased averaging.
                logger.warning(
                    f"Layer '{key}' missing in {num_clients - len(tensors)}/{num_clients} "
                    f"clients. Skipping aggregation for this parameter."
                )

        return global_state


def run_server_round(
    proto_manager: GlobalPrototypeBank,
    model_server: FederatedModelServer,
    client_payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute the complete server-side logic for a single communication round.

    This function orchestrates both server-side aggregation tasks:
        1. **Prototype Aggregation**: Extract local prototype tensors from all
           client payloads and merge them into the global prototype bank using
           the Merge-or-Add strategy with EMA.
        2. **Weight Aggregation**: Extract local model state dicts and compute
           the FedAvg global weights (element-wise mean).

    The returned dict contains the updated global state that the orchestrator
    broadcasts to all clients at the start of the next round.

    Parameters
    ----------
    proto_manager : GlobalPrototypeBank
        The server's prototype bank instance. Modified in-place by the
        merge operation.
    model_server : FederatedModelServer
        The FedAvg aggregation server instance.
    client_payloads : List[Dict[str, Any]]
        List of payloads received from participating clients. Each payload
        is a dict with the following expected keys:
            - 'client_id' (str): Unique identifier for the client.
            - 'protos' (torch.Tensor, optional): Local prototypes [K_i, D].
            - 'weights' (Dict[str, torch.Tensor]): Local model state dict.

    Returns
    -------
    Dict[str, Any]
        Updated global state with two keys:
            - 'global_prototypes' (torch.Tensor): The merged global prototype
              bank of shape [M, D].
            - 'global_weights' (Dict[str, torch.Tensor]): The FedAvg-aggregated
              global model state dict.
        Returns an empty dict if no client payloads were received.
    """
    if not client_payloads:
        return {}

    # Step 1: Prototype Aggregation
    # Collect all local prototype tensors into a flat list. Client identity
    # is irrelevant for prototype merging — each prototype is treated as an
    # independent observation of a visual concept on the unit hypersphere.
    protos = [p["protos"] for p in client_payloads if "protos" in p]
    global_protos = proto_manager.merge_local_prototypes(protos)

    # Step 2: Weight Aggregation
    # Build a mapping from client ID to state dict for FedAvg.
    client_weights_map = {}
    for i, p in enumerate(client_payloads):
        if "weights" in p:
            cid = p.get("client_id", f"unknown_client_{i}")
            client_weights_map[cid] = p["weights"]

    global_weights = model_server.aggregate_weights(client_weights_map)

    return {
        "global_prototypes": global_protos,
        "global_weights": global_weights,
    }


class GlobalModel:
    """
    Server-side global model wrapper with pretrained backbone loading.

    This class encapsulates the server's copy of the global model (e.g.,
    ViT-MAE with IBA adapters). It handles two responsibilities:

    1. **Initialization**: Load a pretrained ViTMAEForPreTraining checkpoint
       from Hugging Face, inject IBA adapters, and freeze the backbone.
       Falls back to a lightweight mock if dependencies are unavailable.

    2. **Weight Updates**: After each round's FedAvg aggregation, load the
       averaged weights into the model. Uses ``strict=False`` to handle
       partial state dicts (e.g., when only adapter weights are communicated).

    Attributes
    ----------
    device : torch.device
        Computation device for the global model.
    model : nn.Module
        The actual PyTorch model (ViTMAE + adapters, or mock fallback).
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the global model from a pretrained checkpoint.

        Attempts to load ``ViTMAEForPreTraining`` from the ``facebook/vit-mae-base``
        checkpoint and inject IBA adapters. If the ``transformers`` library is
        not installed or the download fails, a lightweight mock model is used
        instead for pipeline testing.

        Parameters
        ----------
        device : str
            Target device for the global model ('cpu', 'cuda', or 'cuda:N').
            Default: 'cpu'.
        """
        self.device = torch.device(device)
        logger.info(f"Initializing Global Model on {self.device}...")

        try:
            from transformers import ViTMAEForPreTraining
            from src.mae_with_adapter import inject_adapters

            # Load pretrained ViT-MAE backbone from Hugging Face Hub.
            self.model = ViTMAEForPreTraining.from_pretrained(
                "facebook/vit-mae-base"
            )

            # Inject IBA adapters into every encoder layer. This freezes the
            # backbone and makes only the adapter parameters trainable.
            self.model = inject_adapters(self.model)

            self.model.to(self.device)

            # Server model stays in eval mode — it is used for aggregation
            # and broadcasting, not for direct training.
            self.model.eval()

            logger.info("Global Model successfully initialized with Adapters.")

        except ImportError as e:
            logger.warning(
                f"Failed to import required modules for real model: {e}. "
                f"Using Mock Model."
            )
            # Fallback mock for testing without HuggingFace dependencies.
            self.model = torch.nn.Linear(10, 10).to(self.device)

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def update_model_weights(
        self,
        aggregated_weights: Dict[str, torch.Tensor],
    ) -> None:
        """
        Load FedAvg-aggregated weights into the global model.

        Uses ``strict=False`` because clients may only send back the trainable
        adapter parameters (not the frozen backbone weights). Missing keys
        (frozen backbone layers) and unexpected keys (if any) are logged for
        debugging.

        Parameters
        ----------
        aggregated_weights : Dict[str, torch.Tensor]
            The global state dict produced by ``FederatedModelServer.aggregate_weights()``.
            May be a full state dict or a partial dict containing only adapter
            parameters.
        """
        if not aggregated_weights:
            logger.warning("Received empty weight update. Skipping.")
            return

        try:
            # strict=False allows partial state dict loading — essential when
            # only adapter weights are communicated (backbone remains frozen).
            keys = self.model.load_state_dict(aggregated_weights, strict=False)
            logger.info(
                f"Global Model Updated. "
                f"Missing keys: {len(keys.missing_keys)}, "
                f"Unexpected keys: {len(keys.unexpected_keys)}"
            )
        except Exception as e:
            logger.error(f"Failed to update global model weights: {e}")
            raise