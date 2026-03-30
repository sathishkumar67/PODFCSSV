"""Expose the reusable building blocks behind the current pipeline.

The top-level training file imports from ``src`` so the experiment can read
like a high-level checklist:
1. ``src.mae_with_adapter`` builds the frozen MAE backbone plus trainable adapters.
2. ``src.loss`` defines GPAD, the federated prototype-anchoring loss.
3. ``src.client`` handles client-side optimization and persistent local memory.
4. ``src.server`` handles prototype merging and adapter aggregation on the server.
"""

from src.client import ClientManager, FederatedClient
from src.loss import GPADLoss
from src.mae_with_adapter import IBA_Adapter, ViTBlockWithAdapter, inject_adapters
from src.server import (
    FederatedModelServer,
    GlobalModel,
    GlobalPrototypeBank,
    run_server_round,
)

__all__ = [
    "ClientManager",
    "FederatedClient",
    "FederatedModelServer",
    "GlobalModel",
    "GlobalPrototypeBank",
    "GPADLoss",
    "IBA_Adapter",
    "ViTBlockWithAdapter",
    "inject_adapters",
    "run_server_round",
]
