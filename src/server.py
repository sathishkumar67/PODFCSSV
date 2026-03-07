r"""
Server-Side Central Orchestration for Federated Continual Representation Learning.

This module implements the stateless coordination hub for the federated learning 
graph. The server aggregates high-dimensional knowledge from isolated edge nodes 
without ever directly inspecting private data, fulfilling privacy-preserving 
machine learning (PPML) constraints.

Core Mathematical Mechanisms
----------------------------
1. **Merge-or-Add Basis Expansion (`GlobalPrototypeBank`)**:
   Maintains the global representational basis $P_{global} \in \mathbb{R}^{M \times D}$. 
   Density peaks from local client buffers are checked for collinearity against 
   the global basis using cosine similarity. If $\cos(x_{new}, p_m) \geq \tau_{merge}$, 
   the centroid is updated via momentum: 
   $p_m^{(t+1)} = \text{Norm}((1-\alpha)p_m^{(t)} + \alpha x_{new})$.
   Otherwise, the cluster is registered as a novel concept, expanding $M$.

2. **Distributed Function Approximation (`FederatedModelServer`)**:
   Implements the celebrated FedAvg algorithm for the parameter bottleneck tensors:
   $\theta^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} \theta_k^{(t)}$.
   Because data truncation ensures uniform contribution $n_k$, the aggregation 
   simplifies to the element-wise arithmetic mean of adapter $\Delta W$ projections.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class GlobalPrototypeBank:
    r"""
    Central repository for tracking and discovering federated visual concepts.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        merge_threshold: float = 0.8,
        ema_alpha: float = 0.1,
        device: str = "cpu",
        max_prototypes: int = 50,
    ) -> None:
        r"""
        Instantiates the global memory bank and heuristic boundaries.

        Args:
            embedding_dim: Projection space dimensionality $\mathbb{R}^D$.
            merge_threshold: Cosine similarity $\tau_{merge}$ for concept equivalence.
            ema_alpha: Momentum velocity $\alpha$ for tracking centroid drift.
            device: Accelerator target.
            max_prototypes: Hard upper bound $M_{max}$ on representational cardinality.
        """
        self.embedding_dim = embedding_dim
        self.merge_threshold = merge_threshold
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)
        self.max_prototypes = max_prototypes

        self.prototypes = torch.zeros(0, embedding_dim, device=self.device)

    @torch.no_grad()
    def merge_local_prototypes(
        self,
        local_protos_list: List[torch.Tensor],
    ) -> torch.Tensor:
        r"""
        Synthesizes the federated representations via global Merge-or-Add logic.
        
        Strictly processes topological assignments sequentially. If concurrent 
        clients discover the same local concept space, sequential stochastic 
        merging prevents exponential inflation of dense region centroids.
        """
        if not local_protos_list:
            return self.prototypes

        incoming = torch.cat(local_protos_list, dim=0).to(self.device)
        incoming = F.normalize(incoming, p=2, dim=1)

        if self.prototypes.size(0) == 0:
            if (
                self.max_prototypes is not None
                and incoming.size(0) > self.max_prototypes
            ):
                logger.warning(
                    f"Initialization bottleneck: incoming volume ({incoming.size(0)}) "
                    f"exceeds boundary constraint $M_{{max}}$ ({self.max_prototypes}). Truncating basis."
                )
                self.prototypes = incoming[: self.max_prototypes]
            else:
                self.prototypes = incoming
            return self.prototypes

        for i in range(incoming.size(0)):
            p_new = incoming[i]  

            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            sims = torch.mv(self.prototypes, p_new)  
            max_sim, best_idx = sims.max(dim=0)

            if max_sim >= self.merge_threshold:
                old_vec = self.prototypes[best_idx]
                blended = (1 - self.ema_alpha) * old_vec + self.ema_alpha * p_new
                self.prototypes[best_idx] = F.normalize(blended, p=2, dim=0)
            else:
                if (
                    self.max_prototypes is None
                    or self.prototypes.size(0) < self.max_prototypes
                ):
                    self.prototypes = torch.cat(
                        [self.prototypes, p_new.unsqueeze(0)], dim=0
                    )
                else:
                    logger.info(
                        f"Topological capacity saturation ($M={self.max_prototypes}$). "
                        f"Discarding independent novel discovery."
                    )

        return self.prototypes

    def get_prototypes(self) -> torch.Tensor:
        r"""
        Exposes the fully synthesized $P_{global} \in \mathbb{R}^{M \times D}$ space.
        """
        return self.prototypes


class FederatedModelServer:
    r"""
    Standard FedAvg linear synchronization interface.
    
    Synthesizes the global weight state $\theta^{(t)}$ by evaluating the expected 
    value over the disparate client sub-topologies. Assumes strict homogeneous 
    architectural conformity across all submitted parameter dicts.
    """

    @torch.no_grad()
    def aggregate_weights(
        self,
        client_weights_map: Dict[str, Dict[str, torch.Tensor]],
        current_global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        r"""
        Executes parameter mean aggregation using the previous global trace 
        as an implicit fallback matrix for dropped keys.
        """
        if not client_weights_map:
            logger.warning("Network submission vector empty; aborting synchronization round.")
            return {}

        client_ids = list(client_weights_map.keys())
        num_clients = len(client_ids)

        if num_clients == 1:
            return client_weights_map[client_ids[0]]

        reference_keys = client_weights_map[client_ids[0]].keys()
        global_state: Dict[str, torch.Tensor] = {}

        for key in reference_keys:
            if key in client_weights_map[client_ids[0]]:
                accumulated_weights = client_weights_map[client_ids[0]][key].clone()
            elif current_global_weights and key in current_global_weights:
                logger.warning(
                    f"Parametric absence at key '{key}' from agent '{client_ids[0]}'. "
                    "Imputing via zero-order global hold."
                )
                accumulated_weights = current_global_weights[key].clone()
            else:
                logger.warning(
                    f"Irrecoverable structural omission on key '{key}'. "
                    "Graph alignment compromised for this parameter."
                )
                continue 

            for i in range(1, num_clients):
                cid = client_ids[i]
                client_sd = client_weights_map[cid]

                if key in client_sd:
                    accumulated_weights += client_sd[key]
                elif current_global_weights and key in current_global_weights:
                    logger.warning(
                        f"Parametric absence at key '{key}' for agent {cid}. "
                        "Substituting global invariant."
                    )
                    accumulated_weights += current_global_weights[key]
                else:
                    logger.error(
                        f"Fatal node misalignment: Key '{key}' untracked by '{cid}' "
                        "and unavailable in global dictionary. Aggregation biased."
                    )
                    pass 

            global_state[key] = accumulated_weights / num_clients

        return global_state


# ══════════════════════════════════════════════════════════════════════════
# 3. ROUND ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════

from typing import Tuple

def run_server_round(
    proto_manager: GlobalPrototypeBank,
    model_server: FederatedModelServer,
    client_payloads: List[Dict[str, Any]],
    current_global_weights: Dict[str, torch.Tensor],
    round_idx: int = 1,
    server_model_ema_alpha: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
    Executes the bipartite synchronization protocol for a single communication interval.
    
    Orchestrates the asynchronous assimilation of both the representation manifold 
    (prototypes) and the deep learning parameters (FedAvg weights), acting as the 
    definitive state transition barrier between epochs $(t)$ and $(t+1)$.
    """
    if not client_payloads:
        return torch.empty(0, 0), {}

    protos = [p["protos"] for p in client_payloads if "protos" in p]
    global_protos = proto_manager.merge_local_prototypes(protos)

    client_weights_map = {}
    for i, p in enumerate(client_payloads):
        if "weights" in p:
            cid = p.get("client_id", f"unknown_client_{i}")
            client_weights_map[cid] = p["weights"]

    aggregated_state_dict = model_server.aggregate_weights(
        client_weights_map, 
        current_global_weights
    )

    if round_idx > 1 and current_global_weights is not None:
        for key in aggregated_state_dict:
            if key in current_global_weights:
                old_w = current_global_weights[key]
                new_w = aggregated_state_dict[key]
                aggregated_state_dict[key] = (1.0 - server_model_ema_alpha) * old_w + server_model_ema_alpha * new_w

    return global_protos, aggregated_state_dict


# ══════════════════════════════════════════════════════════════════════════
# 4. GLOBAL MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════

class GlobalModel:
    r"""
    Maintains the persistent parameter representation for the central server.
    
    Serves exclusively as an inference and initialization hub. Synthesizes the 
    foundation `facebook/vit-mae-base` autoencoder and explicitly injects the 
    trainable Information Bottleneck Adapters.
    """

    def __init__(self, device: str = "cpu") -> None:
        r"""
        Dynamically allocates the pre-trained graph into the server memory layout.
        
        Gracefully defaults to a topological dummy layer if the huggingface 
        safetensors are unavailable in the host environment.
        """
        self.device = torch.device(device)
        logger.info(f"Instantiating Global Parameter Graph on {self.device}...")

        try:
            from transformers import ViTMAEForPreTraining
            from src.mae_with_adapter import inject_adapters

            self.model = ViTMAEForPreTraining.from_pretrained(
                "facebook/vit-mae-base"
            )
            self.model = inject_adapters(self.model)
            self.model.to(self.device)
            self.model.eval()  

            logger.info("Global Topology successfully initialized (Adapters Active).")

        except ImportError as e:
            logger.warning(f"Import failed ({e}) — falling back to stub model.")
            self.model = torch.nn.Linear(10, 10).to(self.device)

        except Exception as e:
            logger.error(f"Model init error: {e}")
            raise

    def update_model_weights(
        self,
        aggregated_weights: Dict[str, torch.Tensor],
    ) -> None:
        r"""
        Ingests the FedAvg approximation into the active PyTorch compute graph.
        
        Permits `strict=False` mapping because edge devices communicate strictly 
        via sparse Parameter-Efficient Fine-Tuning ($\Delta W_{adapter}$) tensors.
        """
        if not aggregated_weights:
            logger.warning("Empty structural dictionary submitted. Bypassing update.")
            return

        try:
            keys = self.model.load_state_dict(aggregated_weights, strict=False)
            logger.info(
                f"Global Topology Transmuted. "
                f"Missing keys (Frozen): {len(keys.missing_keys)}, "
                f"Unexpected keys: {len(keys.unexpected_keys)}"
            )
        except Exception as e:
            logger.error(f"Critical fault asserting local parameters to global graph: {e}")
            raise