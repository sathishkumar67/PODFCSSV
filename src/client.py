r"""
Client-Side Topology for Federated Continual Self-Supervised Learning.

This module encapsulates the local execution physics of simulated edge devices 
(clients). In the context of Federated Continual Learning, each client operates 
as an isolated data silo, independently optimizing a local objective while 
coordinating with a central parameter bank to mitigate catastrophic forgetting.

The `FederatedClient` manages the local empirical risk minimization (ERM), 
feature extraction, and prototype discovery. The `ClientManager` handles the 
dispatch orchestration across available hardware accelerators.

Federated Execution Lifecycle
-----------------------------
- **Round 1 (Initialization)**: Local optimization on $\mathcal{L}_{MAE}$. 
  Post-training, global average pooled embeddings are clustered via Spherical 
  K-Means to initialize the local prototype representation matrix $P_{local}$.
- **Round T > 1 (Continual Phase)**: Local optimization on 
  $\mathcal{L} = \mathcal{L}_{MAE} + \lambda_{H} \mathcal{L}_{GPAD}$. 
  Embeddings are routed probabilistically:
  1. **Anchored**: Alignment with $P_{global}$ via GPAD gradient projection.
  2. **Non-anchored**: Projected against $P_{local}$. If similarity exceeds 
     $\tau_{local}$, $P_{local}$ is updated via normalized Exponential Moving 
     Average (EMA). Otherwise, the embedding is cached in a stochastic Novelty Buffer.

Novelty Integration (Merge-or-Add)
----------------------------------
When the Novelty Buffer hits capacity, it is dynamically clustered. 
Centroids highly colinear with existing $P_{local}$ are merged via EMA to track 
concept drift, while orthogonal centroids are appended to expand the client's 
representational basis.
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FederatedClient:
    r"""
    Isolated computational node representing a federated participant.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        local_update_threshold: float = 0.7,
        local_ema_alpha: float = 0.1,
        lambda_proto: float = 1.0,
        novelty_buffer_size: int = 500,
        novelty_k: int = 20,
        kmeans_max_iters: int = 100,
        kmeans_tol: float = 1e-4,
    ) -> None:
        r"""
        Initializes the client's memory boundaries and hyperparameter states.
        
        Args:
            client_id: Unique topological identifier.
            model: Foundation structural template (deep-copied strictly for isolation).
            device: Accelerator assignment target.
            dtype: Tensor precision target.
            optimizer_kwargs: Stochastic gradient descent configuration parameters.
            local_update_threshold: $\tau_{local}$ similarity scalar for EMA merging.
            local_ema_alpha: Local momentum parameter $\alpha_{EMA}$.
            lambda_proto: Optimization weighting constraint $\lambda_H$ for GPAD.
            novelty_buffer_size: Max capacity $C_{buf}$ for un-anchored vectors.
            novelty_k: Hyperparameter $K_{buf}$ partitioning the buffer topology.
            kmeans_max_iters: Algorithmic convergence limit.
            kmeans_tol: Float tolerance for cluster displacement stability.
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

        self.local_prototypes: Optional[torch.Tensor] = None
        self.novelty_buffer: List[torch.Tensor] = []

        self.model = copy.deepcopy(model).to(self.device)
        logger.info(f"[Client {self.client_id}] Graph Instantiated on {self.device}")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.optimizer = optim.AdamW(trainable_params, **opt_kwargs)

        logger.info(
            f"[Client {self.client_id}] Setup Complete | "
            f"device={self.device} | dtype={self.dtype} | "
            f"ema={self.local_ema_alpha} | tau_l={self.local_update_threshold} | "
            f"lambda_p={self.lambda_proto} | buf_cap={self.novelty_buffer_size} | "
            f"k={self.novelty_k}"
        )

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        Extracts un-masked topology spatial features via the frozen backbone graph.
        Reserved strict for static inference/Generation phases ($t=0$).
        """
        encoder_output = self.model.vit(inputs)
        return encoder_output.last_hidden_state.mean(dim=1)

    def train_epoch(
        self,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> float:
        r"""
        Executes one full epoch of stochastic local optimization over $\mathcal{D}_k$.
        The graph propagates both the Generative Masked Autoencoder mapping 
        and the Contrastive GPAD distillation loss simultaneously.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        has_gpad = (global_prototypes is not None) and (gpad_loss_fn is not None)
        if has_gpad:
            logger.info(
                f"[Client {self.client_id}] Objective state: MAE + GPAD "
                f"(\\lambda={self.lambda_proto}, |V|={global_prototypes.shape[0]})"
            )
        else:
            logger.info(f"[Client {self.client_id}] Objective state: MAE base metric")

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            # Output constraint ensures simultaneous mask evaluation across endpoints
            outputs = self.model(inputs, output_hidden_states=True)
            mae_loss = getattr(outputs, "loss", None)
            
            if mae_loss is None:
                mae_loss = torch.tensor(
                    0.0, dtype=self.dtype, device=self.device, requires_grad=True
                )
            final_loss = mae_loss

            # Isolate latent mapping from spatial domain representation (exclude CLS token)
            last_hidden = outputs.hidden_states[-1]              
            embeddings = last_hidden[:, 1:, :].mean(dim=1).detach()  

            if has_gpad and embeddings is not None:
                protos_on_device = global_prototypes.detach().to(self.device)

                # Heuristic routing evaluation step using GPAD entropy threshold bounds
                anchor_mask = gpad_loss_fn.compute_anchor_mask(
                    embeddings, protos_on_device
                ) 

                anchored_embs = embeddings[anchor_mask]
                if anchored_embs.shape[0] > 0:
                    gpad_loss = gpad_loss_fn(anchored_embs, protos_on_device)
                    final_loss = final_loss + self.lambda_proto * gpad_loss

                non_anchored_embs = embeddings[~anchor_mask]
                if non_anchored_embs.shape[0] > 0:
                    self._route_non_anchored(non_anchored_embs)

            # Backpropagation of stochastic network gradients
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(
            f"[Client {self.client_id}] Epoch concluded | Loss={avg_loss:.6f} | "
            f"Batches={num_batches} | buf_load={len(self.novelty_buffer)}"
        )
        return avg_loss

    # ==================================================================
    # Per-Embedding Routing: Local Prototype Update & Novelty Buffer
    # ==================================================================
    @torch.no_grad()
    def _route_non_anchored(self, embeddings: torch.Tensor) -> None:
        r"""
        Processes embeddings orthogonal to the global prototype basis.
        
        If a local centroid exists within the $\tau_{local}$ neighborhood, 
        it is updated online via EMA. Otherwise, the embedding is cached 
        in the stochastic Novelty Buffer for batch clustering.
        """
        z_norm = F.normalize(embeddings.detach(), p=2, dim=1)  

        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            for i in range(z_norm.shape[0]):
                self.novelty_buffer.append(z_norm[i].clone().cpu())
            self._maybe_cluster_buffer()
            return

        if self.local_prototypes.device != self.device:
            self.local_prototypes = self.local_prototypes.to(self.device)

        p_norm = F.normalize(self.local_prototypes, p=2, dim=1)  

        sims = torch.mm(z_norm, p_norm.t())
        max_sim, best_idx = sims.max(dim=1)  

        for i in range(z_norm.shape[0]):
            if max_sim[i] > self.local_update_threshold:
                proto_idx = best_idx[i]
                old_proto = p_norm[proto_idx]                  
                blended = (
                    (1 - self.local_ema_alpha) * old_proto
                    + self.local_ema_alpha * z_norm[i]         
                )
                self.local_prototypes[proto_idx] = F.normalize(
                    blended, p=2, dim=0
                )
            else:
                self.novelty_buffer.append(z_norm[i].clone().cpu())

        self._maybe_cluster_buffer()

    def _maybe_cluster_buffer(self) -> None:
        r"""
        Invokes deterministic clustering upon reaching $C_{buf}$ capacity.
        """
        if len(self.novelty_buffer) >= self.novelty_buffer_size:
            logger.info(
                f"[Client {self.client_id}] Buffer Threshold Met "
                f"({len(self.novelty_buffer)} \\geq {self.novelty_buffer_size}) "
                f"-> Initiating Basis Expansion"
            )
            self._cluster_novelty_buffer()

    @torch.no_grad()
    def _cluster_novelty_buffer(self) -> None:
        r"""
        Implements the Merge-or-Add representation dynamics.
        
        Discovers local density peaks within the cache queue using $K_{buf}$-Means 
        on the hypersphere. Collisions with existing prototypical concepts trigger 
        EMA merging to prevent capacity bloat, while structurally independent 
        concepts expand the network's cardinality.
        """
        if len(self.novelty_buffer) == 0:
            return

        buffer_tensor = torch.stack(self.novelty_buffer, dim=0).to(self.device)
        buffer_tensor = F.normalize(buffer_tensor, p=2, dim=1)  

        K = min(self.novelty_k, buffer_tensor.shape[0])
        logger.info(
            f"[Client {self.client_id}] Extracting structural density peaks: "
            f"N={buffer_tensor.shape[0]} -> K={K}"
        )

        new_centroids = self._kmeans(buffer_tensor, K=K)  

        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            self.local_prototypes = new_centroids
            merged_count = 0
            added_count = K
        else:
            if self.local_prototypes.device != self.device:
                self.local_prototypes = self.local_prototypes.to(self.device)

            p_norm = F.normalize(self.local_prototypes, p=2, dim=1)
            c_norm = F.normalize(new_centroids, p=2, dim=1)

            sims = torch.mm(c_norm, p_norm.t())
            max_sim, best_idx = sims.max(dim=1)

            updated_protos = p_norm.clone()
            protos_to_add = []
            merged_count = 0
            added_count = 0

            for i in range(new_centroids.shape[0]):
                if max_sim[i] > self.local_update_threshold:
                    idx = best_idx[i]
                    old_proto = updated_protos[idx]            
                    blended = (
                        (1 - self.local_ema_alpha) * old_proto
                        + self.local_ema_alpha * c_norm[i]     
                    )
                    updated_protos[idx] = F.normalize(blended, p=2, dim=0)
                    merged_count += 1
                else:
                    protos_to_add.append(new_centroids[i])
                    added_count += 1

            if len(protos_to_add) > 0:
                new_stack = torch.stack(protos_to_add, dim=0)
                self.local_prototypes = torch.cat(
                    [updated_protos, new_stack], dim=0
                )
            else:
                self.local_prototypes = updated_protos

        self.novelty_buffer.clear()

        logger.info(
            f"[Client {self.client_id}] Integration Cycle Complete | "
            f"EMA Merges: {merged_count}, Basis Additions: {added_count} | "
            f"Total Basis Dim: {self.local_prototypes.shape[0]}"
        )

    def get_local_prototypes(self) -> Optional[torch.Tensor]:
        r"""Return current local prototypes ``[K, D]`` (or None if unset)."""
        return self.local_prototypes

    # ==================================================================
    # Round-1 Prototype Generation (Post-Training K-Means)
    # ==================================================================
    @torch.no_grad()
    def generate_prototypes(
        self, dataloader: DataLoader, K_init: int = 10
    ) -> torch.Tensor:
        r"""
        Populates the $P_{local}$ matrix at $t=1$.
        
        Requires a complete empirical pass over the local dataset distribution $\mathcal{D}_1$ 
        to synthesize robust initial centroids via unconstrained mapping.
        """
        self.model.eval()
        all_features = []

        logger.info(f"[Client {self.client_id}] Initiating Phase-1 Topology Scan...")

        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            with torch.inference_mode():
                features = self._extract_features(inputs)
            all_features.append(features)

        embeddings = torch.cat(all_features, dim=0)  
        logger.info(
            f"[Client {self.client_id}] Embedding Yield: {embeddings.shape[0]} "
            f"vectors (dim={embeddings.shape[1]})"
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        centroids = self._kmeans(embeddings, K=K_init)
        logger.info(f"[Client {self.client_id}] Centroid discovery complete (K={K_init})")

        self.local_prototypes = centroids.detach().clone()
        return centroids

    # ==================================================================
    # Spherical K-Means Clustering (Pure PyTorch)
    # ==================================================================
    def _kmeans(self, X: torch.Tensor, K: int) -> torch.Tensor:
        r"""
        Implements Spherical K-Means optimization algorithm.
        Minimizes cosine distance $\min \sum (1 - \cos(x_i, c_j))$ via Lloyd's heuristic, 
        projecting centroids back to strictly unit norm after every update iteration.
        """
        N, D = X.shape
        K = min(K, N)  

        indices = torch.randperm(N, device=X.device)[:K]
        centroids = X[indices].clone()  

        for iteration in range(self.kmeans_max_iters):
            centroids = F.normalize(centroids, p=2, dim=1)

            sims = torch.mm(X, centroids.t())  
            _, labels = sims.max(dim=1)         

            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    new_centroids[k] = X[torch.randint(0, N, (1,), device=X.device).item()]

            new_centroids = F.normalize(new_centroids, p=2, dim=1)
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if center_shift < self.kmeans_tol:
                logger.info(
                    f"[Client {self.client_id}] Convergence established at "
                    f"step {iteration + 1} | $\\Delta=${center_shift:.6f}"
                )
                break

        return F.normalize(centroids, p=2, dim=1)


# ══════════════════════════════════════════════════════════════════════════
# CLIENT MANAGER (Factory + Dispatch)
# ══════════════════════════════════════════════════════════════════════════

class ClientManager:
    r"""
    Topology orchestrator for federated participant simulation.

    Manages a deterministic ensemble of $\mathcal{N}$ `FederatedClient` objects, 
    dispatching training objectives iteratively or in asynchronous true-parallel 
    (disabling the GIL) mapping threads to dedicated CUDA endpoints.
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
        r"""
        Constructs the client ensemble and orchestrates device allocation.

        Args:
            base_model: Foundation structural template.
            num_clients: Total cardinality of the federated graph $|\mathcal{N}|$.
            gpu_count: Available CUDA accelerators.
            dtype: Tensor precision.
            optimizer_kwargs: Dictionary configuration for local AdamW.
            local_update_threshold: $\tau_{local}$ similarity margin.
            local_ema_alpha: Local momentum parameter $\alpha_{EMA}$.
            lambda_proto: Optimization weighting constraint $\lambda_H$.
            novelty_buffer_size: Max capacity $C_{buf}$.
            novelty_k: Cluster budget $K_{buf}$.
            kmeans_max_iters: Algorithmic limits.
            kmeans_tol: Displacement tolerance limit.
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

    # ------------------------------------------------------------------
    def _initialize_clients(self, base_model: nn.Module) -> None:
        r"""
        Maps federated nodes strictly 1:1 to available physical CUDA cores,
        or multiplexes them across the CPU host matrix logically.
        """
        logger.info(
            f"[ClientManager] Synchronizing {self.num_clients} client spaces..."
        )

        if self.gpu_count > 0:
            if self.num_clients != self.gpu_count:
                raise ValueError(
                    f"Bijective client-hardware constraint violation: {self.num_clients} "
                    f"clients versus {self.gpu_count} CUDA cores."
                )
            logger.info(
                f"[ClientManager] Hardware mapped: true {self.num_clients}-core parallelism"
            )
        else:
            logger.info(
                f"[ClientManager] CPU multiplexing: sequentially evaluating {self.num_clients} clients"
            )

        for i in range(self.num_clients):
            device = (
                torch.device(f"cuda:{i}")
                if self.gpu_count > 0
                else torch.device("cpu")
            )

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
            f"[ClientManager] {self.num_clients} clients ready."
        )

    # ------------------------------------------------------------------
    def train_round(
        self,
        dataloaders: List[DataLoader],
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> List[float]:
        r"""
        Initiates communication round optimization cycles.
        
        Exploits ThreadPoolExecutor for OS-level threading when bypassing the GIL 
        in CUDA acceleration contexts. Defaults to blocked sequential execution 
        for host CPU compute paths.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                fr"Data boundary mismatch: $|\mathcal{{D}}| = {len(dataloaders)}$, "
                fr"$|\mathcal{{N}}| = {self.num_clients}$"
            )

        round_losses = [0.0] * self.num_clients

        if self.gpu_count > 0:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                logger.info(
                    f"[ClientManager] Instantiating parallel thread pool "
                    f"(|Workers|={self.num_clients})..."
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
                    idx = futures[future]
                    try:
                        loss = future.result()
                        round_losses[idx] = loss
                        logger.info(
                            f"[ClientManager] Endpoint {idx} (cuda:{idx}) "
                            f"Loss Convergence = {loss:.4f}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[ClientManager] Endpoint {idx} Critical Fault: {e}"
                        )
                        round_losses[idx] = float("nan")
        else:
            logger.info(
                f"[ClientManager] Initiating serial topology execution "
                f"(|N|={self.num_clients})..."
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
                        f"[ClientManager] Endpoint {i} (cpu) "
                        f"Loss Convergence = {loss:.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[ClientManager] Endpoint {i} Critical Fault: {e}"
                    )
                    round_losses[i] = float("nan")

        return round_losses