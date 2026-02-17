"""
Federated Client Module for Continual Self-Supervised Learning.

This module defines the client-side components of the Federated Learning pipeline.
Each client operates on a private local dataset, trains a copy of the global model,
and communicates only compact prototype vectors (not raw data) to the server.

Components
----------
1. FederatedClient : Represents a single edge device in the federation.
2. ClientManager   : Orchestrates multiple FederatedClient instances.

Training Phases
---------------
- Round 1 (Initialization): Pure Masked Autoencoder (MAE) loss. No global knowledge exists yet.
- Round > 1 (Continual):     MAE loss + GPAD distillation loss using global prototypes
                            received from the server, preventing catastrophic forgetting.

Prototype Lifecycle
-------------------
1. After each training epoch, the client extracts feature embeddings from its local data
using the ViT encoder (model.vit).
2. These embeddings are clustered via K-Means to produce K local prototype vectors.
3. Between training batches, prototypes are refined online via EMA updates for any
sample whose cosine similarity to its nearest prototype exceeds a threshold.
4. The final prototypes are sent to the server for global aggregation.
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

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Simulates a single edge device (client) in the Federated Learning network.

    Each client holds:
    - A **deep copy** of the global model so that local training is independent.
    - A set of **local prototypes** that summarize its private data distribution.
    - An **optimizer** that updates only the model's trainable parameters.

    The client exposes two main operations per round:
    1. `train_epoch`         — One epoch of local gradient descent.
    2. `generate_prototypes` — Cluster local features into K prototype vectors.

    Attributes
    ----------
    client_id : int
        Unique identifier for this client.
    device : torch.device
        Hardware device this client's model lives on (CPU or a specific GPU).
    dtype : torch.dtype
        Floating-point precision used for inputs and computations (e.g. float32, bfloat16).
    model : nn.Module
        Local copy of the global model (e.g. ViTMAEForPreTraining with adapters).
    optimizer : torch.optim.Optimizer
        Local optimizer instance (default: AdamW).
    local_prototypes : Optional[torch.Tensor]
        Current set of local prototype vectors, shape [K, D]. None until first
        call to `generate_prototypes`.
    local_update_threshold : float
        Cosine-similarity threshold for online EMA updates of local prototypes.
    local_ema_alpha : float
        Interpolation factor for EMA prototype refinement (0 = no update, 1 = full replace).
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
    ) -> None:
        """
        Initialize the Federated Client.

        Parameters
        ----------
        client_id : int
            Unique numeric identifier for logging and tracking.
        model : nn.Module
            The global model template. A deep copy is created internally so that
            each client trains independently without affecting others.
        device : torch.device
            Target device for this client's model and data (e.g. 'cpu', 'cuda:0').
        dtype : torch.dtype
            Data type for input tensors (e.g. torch.float32, torch.bfloat16).
        optimizer_cls : type, optional
            Optimizer class to use for local training. Defaults to AdamW.
        optimizer_kwargs : Dict[str, Any], optional
            Keyword arguments passed to the optimizer constructor.
            Defaults to {"lr": 1e-3}.
        local_update_threshold : float, optional
            Minimum cosine similarity required to trigger an online EMA update
            of a local prototype during training. Defaults to 0.7.
        local_ema_alpha : float, optional
            EMA interpolation factor for online prototype refinement.
            Defaults to 0.1 (slow, stable updates).
        lambda_proto : float, optional
            Scaling factor for the GPAD distillation loss.
            total_loss = mae_loss + lambda_proto * gpad_loss.
            Defaults to 1.0.
        novelty_buffer_size : int, optional
            Number of non-anchored embeddings to accumulate before triggering
            a fresh K-Means clustering to update local prototypes.
            Defaults to 500.
        novelty_k : int, optional
            Number of clusters (K) for K-Means when clustering the novelty
            buffer. This is independent of `k_init_prototypes` used in Round 1.
            Defaults to 20.
        """
        self.client_id = client_id
        self.device = device
        self.dtype = dtype
        self.local_update_threshold = local_update_threshold
        self.local_ema_alpha = local_ema_alpha
        self.lambda_proto = lambda_proto
        self.novelty_buffer_size = novelty_buffer_size
        self.novelty_k = novelty_k

        # Local prototype bank — populated after the first call to generate_prototypes()
        self.local_prototypes: Optional[torch.Tensor] = None

        # Novelty buffer — accumulates truly novel embeddings (those that fail
        # both the global anchor threshold AND the local prototype threshold).
        # When len(novelty_buffer) >= novelty_buffer_size, a fresh K-Means is
        # run on the buffer to produce new local prototypes.
        self.novelty_buffer: List[torch.Tensor] = []

        # 1. Create an independent model copy for this client
        self.model = copy.deepcopy(model).to(self.device)
        logger.info(f"[Client {self.client_id}] Model copied to {self.device}")

        # 2. Initialize the optimizer over ALL model parameters
        opt_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **opt_kwargs)

        logger.info(
            f"[Client {self.client_id}] Initialized | device={self.device} | "
            f"dtype={self.dtype} | optimizer={optimizer_cls.__name__} | "
            f"ema_alpha={self.local_ema_alpha} | threshold={self.local_update_threshold} | "
            f"lambda_proto={self.lambda_proto} | buffer_size={self.novelty_buffer_size} | "
            f"novelty_k={self.novelty_k}"
        )

    # ------------------------------------------------------------------
    # Feature Extraction Helper
    # ------------------------------------------------------------------
    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature embeddings from the ViT encoder.

        This method accesses `model.vit` directly (the ViT encoder inside
        ViTMAEForPreTraining) to obtain the last hidden state, then applies
        mean pooling across the sequence dimension.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor (e.g. pixel values) already cast to the correct
            dtype and moved to the correct device.

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape [Batch, Hidden_Dim].
        """
        # model.vit is the ViTMAEModel encoder inside ViTMAEForPreTraining.
        # It returns a BaseModelOutput with .last_hidden_state of shape [B, L, D].
        encoder_output = self.model.vit(inputs)

        # Mean-pool over the sequence length dimension: [B, L, D] -> [B, D]
        embeddings = encoder_output.last_hidden_state.mean(dim=1)
        return embeddings

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        dataloader: DataLoader,
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> float:
        """
        Execute one epoch of local training with per-embedding routing.

        Per-Embedding Decision Flow (Round > 1)
        ----------------------------------------
        For each embedding z in the batch:

        1. Compute similarity to global prototypes and adaptive threshold.
        2. **Anchored** (sim > threshold): Apply GPAD loss toward matched
            global prototype.  Do NOT update local prototypes or buffer.
        3. **Non-anchored** (sim <= threshold): Check against local prototypes:
            a. Local sim > local_update_threshold → EMA-update closest local proto.
            b. Local sim <= local_update_threshold → Add z to **novelty buffer**.
        4. If novelty buffer reaches ``novelty_buffer_size``, trigger a fresh
            K-Means clustering on the buffer to produce new local prototypes.

        Loss Composition
        ----------------
        - **Round 1** (no global prototypes): Loss = MAE only.
        - **Round > 1**: Loss = MAE + lambda_proto * GPAD (anchored only).

        Parameters
        ----------
        dataloader : DataLoader
            Iterator over the client's private dataset.
        global_prototypes : torch.Tensor, optional
            Global prototype bank from the server, shape [M, D].
            None during Round 1.
        gpad_loss_fn : nn.Module, optional
            The GPAD distillation loss module.  Must expose
            ``compute_anchor_mask(embeddings, protos) -> BoolTensor``.

        Returns
        -------
        float
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        has_gpad = (global_prototypes is not None) and (gpad_loss_fn is not None)
        if has_gpad:
            logger.info(
                f"[Client {self.client_id}] Training with MAE + GPAD "
                f"(lambda={self.lambda_proto}, global protos: {global_prototypes.shape[0]})"
            )
        else:
            logger.info(f"[Client {self.client_id}] Training with MAE only (no global prototypes)")

        for batch_idx, batch in enumerate(dataloader):
            # ---- Data Preparation ----
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            # ---- Step 1: Forward Pass (MAE) ----
            outputs = self.model(inputs)
            mae_loss = getattr(outputs, "loss", None)
            if mae_loss is None:
                mae_loss = torch.tensor(0.0, dtype=self.dtype,
                                        device=self.device, requires_grad=True)
            final_loss = mae_loss

            # ---- Step 2: Feature Extraction ----
            embeddings = self._extract_features(inputs)

            # ---- Step 3: Per-Embedding Routing (Round > 1) ----
            if has_gpad and embeddings is not None:
                protos_on_device = global_prototypes.to(self.device)

                # 3a. Determine which embeddings are anchored vs novel
                anchor_mask = gpad_loss_fn.compute_anchor_mask(
                    embeddings, protos_on_device
                )  # [B] bool: True = anchored

                # 3b. GPAD loss — only for anchored embeddings
                anchored_embs = embeddings[anchor_mask]
                if anchored_embs.shape[0] > 0:
                    gpad_loss = gpad_loss_fn(anchored_embs, protos_on_device)
                    final_loss = final_loss + self.lambda_proto * gpad_loss

                # 3c. Non-anchored embeddings → check local protos / buffer
                non_anchored_embs = embeddings[~anchor_mask]
                if non_anchored_embs.shape[0] > 0:
                    self._route_non_anchored(non_anchored_embs)

            # ---- Step 4: Backward Pass ----
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

    # ------------------------------------------------------------------
    # Online Local Prototype Update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _route_non_anchored(self, embeddings: torch.Tensor) -> None:
        """
        Route non-anchored (novel) embeddings through local prototype check.

        For each non-anchored embedding:
        - If it matches a local prototype (cosine sim > local_update_threshold),
          update that prototype via EMA.
        - If it matches NO local prototype, add it to the novelty buffer.

        When the novelty buffer reaches ``novelty_buffer_size``, a fresh
        K-Means is triggered to produce new local prototypes.

        Parameters
        ----------
        embeddings : torch.Tensor
            Non-anchored embeddings from the current batch, shape [B', D].
        """
        z_norm = F.normalize(embeddings.detach(), p=2, dim=1)

        # ---- Case A: No local prototypes yet → all go into buffer ----
        if self.local_prototypes is None or self.local_prototypes.shape[0] == 0:
            for i in range(z_norm.shape[0]):
                self.novelty_buffer.append(z_norm[i].clone())
            self._maybe_cluster_buffer()
            return

        # ---- Case B: Compare against local prototypes ----
        if self.local_prototypes.device != self.device:
            self.local_prototypes = self.local_prototypes.to(self.device)

        p_norm = F.normalize(self.local_prototypes, p=2, dim=1)
        sims = torch.mm(z_norm, p_norm.t())       # [B', K_local]
        max_sim, best_idx = sims.max(dim=1)        # [B']

        for i in range(z_norm.shape[0]):
            if max_sim[i] > self.local_update_threshold:
                # EMA-update the closest local prototype
                proto_idx = best_idx[i]
                old_proto = self.local_prototypes[proto_idx]
                updated = ((1 - self.local_ema_alpha) * old_proto
                           + self.local_ema_alpha * z_norm[i])
                self.local_prototypes[proto_idx] = updated
            else:
                # Truly novel — add to buffer
                self.novelty_buffer.append(z_norm[i].clone())

        self._maybe_cluster_buffer()

    def _maybe_cluster_buffer(self) -> None:
        """
        Check if the novelty buffer has reached its threshold and trigger
        clustering if so.
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
        Run a fresh K-Means on the novelty buffer to produce new local prototypes.

        This method is called when the novelty buffer reaches its size threshold.
        It performs a completely re-initialized clustering (not reusing old
        centroids) to discover new visual concepts from truly novel samples.

        After clustering:
        - ``self.local_prototypes`` is **replaced** with the new centroids.
        - The novelty buffer is cleared.
        """
        if len(self.novelty_buffer) == 0:
            return

        # Stack buffer list into a single tensor [N_buffer, D]
        buffer_tensor = torch.stack(self.novelty_buffer, dim=0).to(self.device)
        buffer_tensor = F.normalize(buffer_tensor, p=2, dim=1)

        # Use novelty_k (clamped to buffer size)
        K = min(self.novelty_k, buffer_tensor.shape[0])

        logger.info(
            f"[Client {self.client_id}] Clustering novelty buffer: "
            f"{buffer_tensor.shape[0]} samples → K={K}"
        )

        # Fresh K-Means — completely re-initialized centroids
        new_protos = self._kmeans(buffer_tensor, K=K)

        # Replace local prototypes with the newly discovered concepts
        self.local_prototypes = new_protos.detach().clone()

        # Clear the buffer
        self.novelty_buffer.clear()

        logger.info(
            f"[Client {self.client_id}] Buffer clustered → "
            f"{self.local_prototypes.shape[0]} new local prototypes"
        )

    def get_local_prototypes(self) -> Optional[torch.Tensor]:
        """
        Return the current local prototypes without re-clustering.

        Used by the orchestrator in Round > 1 to collect prototypes
        (which are maintained via online EMA + buffer clustering rather
        than full post-training K-Means).

        Returns
        -------
        Optional[torch.Tensor]
            Local prototypes of shape [K, D], or None if not yet initialized.
        """
        return self.local_prototypes

    # ------------------------------------------------------------------
    # Prototype Generation (Post-Training)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_prototypes(self, dataloader: DataLoader, K_init: int = 10) -> torch.Tensor:
        """
        Generate local prototypes by clustering the client's feature space.

        This method is called **after** training to create a compact summary
        of the client's learned representations. The prototypes are then
        sent to the server for global aggregation.

        Procedure
        ---------
        1. **Feature Extraction**: Run the encoder on all local data to collect
            embeddings. Uses `model.vit` to get `last_hidden_state`, then
            mean-pools over the sequence dimension.
        2. **L2 Normalization**: Project embeddings onto the unit hypersphere
            so that cosine similarity equals dot product.
        3. **K-Means Clustering**: Partition the N embeddings into K clusters.
            The K centroids become the local prototypes.
        4. **Store Prototypes**: Save a detached copy for online EMA updates
        during the next training round.

        Parameters
        ----------
        dataloader : DataLoader
            Iterator over the client's private dataset.
        K_init : int, optional
            Number of prototypes (clusters) to generate. Defaults to 10.

        Returns
        -------
        torch.Tensor
            Local prototype matrix of shape [K_init, Hidden_Dim], L2-normalized.
        """
        self.model.eval()
        all_features = []

        logger.info(f"[Client {self.client_id}] Extracting features for prototype generation...")

        # 1. Feature Extraction
        for batch in dataloader:
            # TensorDataset yields a tuple/list of tensors; plain Datasets may yield a tensor
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(self.dtype).to(self.device)

            with torch.inference_mode():
                features = self._extract_features(inputs)

            all_features.append(features)

        # Concatenate all batch features into a single matrix: [N_samples, D]
        embeddings = torch.cat(all_features, dim=0)
        logger.info(f"[Client {self.client_id}] Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

        # 2. L2 Normalization onto the unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 3. K-Means Clustering
        centroids = self._kmeans(embeddings, K=K_init)
        logger.info(f"[Client {self.client_id}] K-Means complete | K={K_init} | centroid shape={centroids.shape}")

        # 4. Store for online EMA updates during the next training round
        self.local_prototypes = centroids.detach().clone()

        return centroids

    # ------------------------------------------------------------------
    # K-Means Clustering (PyTorch Implementation)
    # ------------------------------------------------------------------
    def _kmeans(self, X: torch.Tensor, K: int, max_iters: int = 100) -> torch.Tensor:
        """
        Spherical K-Means clustering on L2-normalized embeddings.

        Because the input embeddings are already on the unit hypersphere,
        cosine similarity (equivalently, dot product) is used as the
        distance metric. This is often called "Spherical K-Means".

        Algorithm
        ---------
        1. **Initialization**: Randomly select K data points as initial centroids.
        2. **Assignment**: Compute cosine similarity between each point and all
            centroids; assign each point to its most similar centroid.
        3. **Update**: Recompute each centroid as the mean of its assigned points,
            then re-normalize to the unit sphere.
        4. **Convergence**: Repeat until centroid shift falls below 1e-4 or
            max_iters is reached.
        5. **Empty Cluster Handling**: If a centroid loses all members, it is
            re-seeded with a random data point to avoid degenerate solutions.

        Parameters
        ----------
        X : torch.Tensor
            Input data matrix, shape [N, D], assumed L2-normalized.
        K : int
            Number of clusters (prototypes) to form.
        max_iters : int, optional
            Maximum number of iterations. Defaults to 100.

        Returns
        -------
        torch.Tensor
            Final centroids, shape [K, D], L2-normalized.
        """
        N, D = X.shape

        # Clamp K to N to avoid index-out-of-bounds when samples < clusters
        K = min(K, N)

        # 1. Random Initialization from data points
        indices = torch.randperm(N, device=X.device)[:K]
        centroids = X[indices].clone()

        for iteration in range(max_iters):
            # Ensure centroids stay on the unit sphere
            centroids = F.normalize(centroids, p=2, dim=1)

            # 2. Assignment Step: cosine similarity = dot product (since both are normalized)
            # sims shape: [N, K]
            sims = torch.mm(X, centroids.t())

            # Each point is assigned to the centroid with highest similarity
            _, labels = sims.max(dim=1)

            # 3. Update Step: recompute centroids as cluster means
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                cluster_mask = (labels == k)
                if cluster_mask.sum() > 0:
                    # Mean of all points assigned to cluster k
                    new_centroids[k] = X[cluster_mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed with a random data point
                    new_idx = torch.randint(0, N, (1,), device=X.device).item()
                    new_centroids[k] = X[new_idx]

            # 4. Convergence Check
            center_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids

            if center_shift < 1e-4:
                logger.info(f"[Client {self.client_id}] K-Means converged at iteration {iteration + 1}")
                break

        # Final normalization to ensure output is on the unit sphere
        return F.normalize(centroids, p=2, dim=1)


class ClientManager:
    """
    Orchestrator for Multiple Federated Clients.

    In a real-world deployment, each client would reside on a separate physical
    device. This class simulates that by managing a list of `FederatedClient`
    instances and dispatching training commands to them.

    Execution Modes
    ---------------
    - **Parallel (Multi-GPU)**: When `gpu_count > 0`, each client is assigned
        to a dedicated GPU (strict 1:1 mapping) and training is executed
        concurrently via `ThreadPoolExecutor`. PyTorch releases the GIL for
        CUDA operations, enabling true parallelism.
    - **Sequential (CPU)**: When `gpu_count == 0`, clients run one after another
        on CPU to avoid GIL contention overhead from threading.

    Attributes
    ----------
    clients : List[FederatedClient]
        List of instantiated client objects.
    num_clients : int
        Total number of clients in the federation.
    gpu_count : int
        Number of available GPUs (0 = CPU-only mode).
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
    ) -> None:
        """
        Initialize the Client Manager and spawn all clients.

        Parameters
        ----------
        base_model : nn.Module
            The global model template. Each client receives a deep copy.
        num_clients : int
            Number of federated clients to simulate.
        gpu_count : int, optional
            Number of available GPUs. If > 0, enforces strict 1:1 Client-GPU
            mapping. Defaults to 0 (CPU mode).
        dtype : torch.dtype, optional
            Floating-point type used for inputs. Defaults to torch.float32.
        optimizer_kwargs : Dict[str, Any], optional
            Keyword arguments passed to each client's optimizer constructor
            (e.g. {"lr": 1e-4, "weight_decay": 0.05}).
        local_update_threshold : float, optional
            Cosine-similarity threshold for online EMA updates of local
            prototypes. Defaults to 0.7.
        local_ema_alpha : float, optional
            EMA interpolation factor for online prototype refinement.
            Defaults to 0.1.
        lambda_proto : float, optional
            GPAD loss weight: total = mae + lambda_proto * gpad. Defaults to 1.0.
        novelty_buffer_size : int, optional
            Buffer size before triggering K-Means on novel samples. Defaults to 500.
        novelty_k : int, optional
            K for buffer K-Means clustering. Defaults to 20.
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

        self._initialize_clients(base_model)

    def _initialize_clients(self, base_model: nn.Module) -> None:
        """
        Internal helper to instantiate and configure all clients.

        Device Assignment Logic
        -----------------------
        - GPU mode: Client i -> cuda:i (strict 1:1 mapping).
        - CPU mode: All clients share the CPU.
        """
        logger.info(f"[ClientManager] Initializing {self.num_clients} clients...")

        # Enforce strict 1:1 Client-GPU mapping
        if self.gpu_count > 0:
            if self.num_clients != self.gpu_count:
                raise ValueError(
                    f"Strict 1:1 Client-GPU mapping required. "
                    f"Requested {self.num_clients} clients but found {self.gpu_count} GPUs."
                )
            logger.info(f"[ClientManager] Parallel Mode: {self.num_clients} clients -> {self.gpu_count} GPUs")
        else:
            logger.info(f"[ClientManager] Sequential Mode: {self.num_clients} clients on CPU")

        for i in range(self.num_clients):
            # Assign device based on mode
            if self.gpu_count > 0:
                device = torch.device(f"cuda:{i}")
            else:
                device = torch.device("cpu")

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
            )
            self.clients.append(client)

        logger.info(f"[ClientManager] All {self.num_clients} clients initialized successfully.")

    def train_round(
        self,
        dataloaders: List[DataLoader],
        global_prototypes: Optional[torch.Tensor] = None,
        gpad_loss_fn: Optional[nn.Module] = None,
    ) -> List[float]:
        """
        Trigger one round of local training for ALL clients.

        Dispatch Logic
        --------------
        - **GPU mode**: Launches concurrent threads via `ThreadPoolExecutor`.
            PyTorch releases the GIL during CUDA kernel execution, so true
            parallelism is achieved.
        - **CPU mode**: Iterates sequentially to avoid GIL contention.

        Parameters
        ----------
        dataloaders : List[DataLoader]
            One DataLoader per client. Length must equal `num_clients`.
        global_prototypes : torch.Tensor, optional
            The current global prototype bank (for GPAD loss).
        gpad_loss_fn : nn.Module, optional
            The GPAD distillation loss module.

        Returns
        -------
        List[float]
            Average training loss for each client.

        Raises
        ------
        ValueError
            If the number of dataloaders does not match the number of clients.
        """
        if len(dataloaders) != self.num_clients:
            raise ValueError(
                f"Dataloader count ({len(dataloaders)}) does not match "
                f"client count ({self.num_clients})"
            )

        round_losses = [0.0] * self.num_clients

        if self.gpu_count > 0:
            # ---- Parallel Execution (Multi-GPU) ----
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
                logger.info(f"[ClientManager] Spawning {self.num_clients} training threads (1 per GPU)...")
                futures = {}
                for i, client in enumerate(self.clients):
                    futures[executor.submit(
                        client.train_epoch,
                        dataloaders[i],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn,
                    )] = i

                for future in futures:
                    client_idx = futures[future]
                    try:
                        loss = future.result()
                        round_losses[client_idx] = loss
                        logger.info(f"[ClientManager] Client {client_idx} (GPU {client_idx}) finished | loss={loss:.4f}")
                    except Exception as e:
                        logger.error(f"[ClientManager] Client {client_idx} FAILED: {e}")
                        round_losses[client_idx] = float("nan")
        else:
            # ---- Sequential Execution (CPU) ----
            logger.info(f"[ClientManager] Running sequential training on CPU for {self.num_clients} clients...")
            for i, client in enumerate(self.clients):
                try:
                    loss = client.train_epoch(
                        dataloaders[i],
                        global_prototypes=global_prototypes,
                        gpad_loss_fn=gpad_loss_fn,
                    )
                    round_losses[i] = loss
                    logger.info(f"[ClientManager] Client {i} (CPU) finished | loss={loss:.4f}")
                except Exception as e:
                    logger.error(f"[ClientManager] Client {i} FAILED: {e}")
                    round_losses[i] = float("nan")

        return round_losses