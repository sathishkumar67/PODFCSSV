"""Expose the core building blocks used by every entrypoint in the repo.

The package is intentionally small so the high-level scripts can read almost
like a pipeline description:
1. Adapter wrappers define which MAE parameters remain trainable.
2. GPAD defines the prototype-anchored loss used in the federated run.
3. Client helpers run local training and maintain local prototype memory.
4. Server helpers merge client updates back into one global state.
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
