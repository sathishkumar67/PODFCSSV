"""Public exports for the PODFCSSV package.

The package is intentionally small, so the top-level namespace re-exports the
main building blocks used by the training scripts:
1. Adapter injection and adapter modules.
2. GPAD loss.
3. Client-side training utilities.
4. Server-side aggregation utilities.
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
