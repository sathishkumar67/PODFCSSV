"""
Parameter-Efficient Fine-Tuning via Information-Bottlenecked Adapters (IBA).

This module lets you fine-tune a large pre-trained ViT-MAE model while
keeping the backbone weights completely frozen.  Lightweight bottleneck
MLP modules ("adapters") are injected after every frozen encoder layer.
Only the adapter parameters (~1 % of total) are trained, which makes
communication in a Federated Learning setting highly efficient because
clients send/receive far fewer parameters each round.

The approach rests on three ideas:

1. **Bottleneck Adapters** (Houlsby et al., ICML 2019):
   Small down-project → activation → up-project modules inserted into
   each transformer block that learn a residual correction to the hidden
   states.

2. **Information Bottleneck** (Tishby et al., 2000):
   The low-rank down-projection forces the adapter to keep only
   task-relevant information and discard noise.

3. **Masked Autoencoders** (He et al., CVPR 2022):
   The frozen backbone is a ViT-MAE model pre-trained with masked image
   modelling, providing strong visual features as the starting point.

Module Contents
---------------
- ``IBA_Adapter``         – The bottleneck adapter module.
- ``ViTBlockWithAdapter`` – Wraps a frozen encoder layer + trainable adapter.
- ``inject_adapters``     – Entry-point: injects adapters into a pre-trained model.
- ``_log_param_stats``    – Diagnostic utility: prints frozen-vs-trainable split.

References
----------
[1] Houlsby, N. et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019.
[2] Tishby, N. et al., "The Information Bottleneck Method", 2000.
[3] He, K. et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Tuple, Union
from transformers import PreTrainedModel, ViTMAEForPreTraining


# ═══════════════════════════════════════════════════════════════════════════
# 1. ADAPTER MODULE
# ═══════════════════════════════════════════════════════════════════════════

class IBA_Adapter(nn.Module):
    """Information-Bottlenecked Adapter (IBA).

    A two-layer bottleneck MLP that learns a small residual correction
    ΔH to frozen hidden states:

        H_out = H_frozen + ΔH

    Forward computation (for input H of shape ``[B, L, D]``):

        1. **Down-project** – D → d  (d ≪ D, the bottleneck).
        2. **Activation**   – GELU non-linearity.
        3. **Up-project**   – d → D  (back to full dimension).
        4. **Dropout**      – regularisation against overfitting on small
           local datasets (important in FL).
        5. **Residual add** – H + ΔH  (skip connection).

    The up-project layer is initialised to **zero**, so at step 0 the
    adapter produces ΔH = 0 and the wrapped block behaves identically to
    the original frozen block, ensuring a smooth start to training.

    Parameters
    ----------
    input_dim      : int        – Hidden dimension D of the backbone (e.g. 768).
    bottleneck_dim : int        – Compressed dimension d.  Default: 64.
    dropout        : float      – Dropout probability.  Default: 0.0.
    activation     : nn.Module  – Non-linearity between projections.  Default: GELU.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation if activation is not None else nn.GELU()

        # Down-project: D → d  (information bottleneck).
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)

        # Up-project: d → D  (reconstruction back to full hidden dim).
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=True)

        # Dropout: prevents adapter overfitting on local data.
        self.dropout = nn.Dropout(dropout)

        # Apply identity-init strategy (up-project = zero at start).
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights for stable training.

        - **Down-project**: Kaiming Normal (fan_out, relu) preserves the
          variance of activations through the bottleneck.
        - **Up-project**: Zero-initialised weights *and* bias so that ΔH = 0
          at step 0.  This is the most critical initialisation choice: it
          guarantees the adapter starts as an identity function and gradually
          learns corrections without disrupting the pre-trained features.
        """
        nn.init.kaiming_normal_(
            self.down_project.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)

        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the adapted hidden states.

        ``H' = H + Dropout(W_up · σ(W_down · H + b_down) + b_up)``

        Parameters
        ----------
        x : torch.Tensor – ``[B, L, D]`` hidden states from the frozen block.

        Returns
        -------
        torch.Tensor – ``[B, L, D]`` adapted hidden states (same shape).
        """
        residual = x                      # save for skip connection
        x = self.down_project(x)          # D → d
        x = self.activation(x)            # non-linearity
        x = self.up_project(x)            # d → D
        x = self.dropout(x)               # regularisation
        return residual + x               # residual add

    def __repr__(self) -> str:
        return f"IBA_Adapter(in={self.input_dim}, bottleneck={self.bottleneck_dim})"


# ═══════════════════════════════════════════════════════════════════════════
# 2. ADAPTER-INJECTED TRANSFORMER BLOCK WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class ViTBlockWithAdapter(nn.Module):
    """Wraps one frozen ViT encoder layer with a trainable IBA adapter.

    Transparently replaces the original layer in the encoder's ModuleList.
    When called by the encoder loop, this wrapper:

        1. Runs the **original frozen block** (self-attention → FFN → LN).
        2. Extracts the hidden states from the output (handles tuple,
           ModelOutput, or raw-tensor returns from Hugging Face).
        3. Applies the IBA adapter to add ΔH.
        4. Repackages the result in the original return format so that
           downstream layers are unaware of the modification.

    Parameters
    ----------
    original_block : nn.Module   – The frozen ViTMAELayer being wrapped.
    adapter        : IBA_Adapter – The trainable adapter applied after it.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]:
        """Run the frozen block, then apply the adapter.

        Parameters
        ----------
        hidden_states : ``[B, L, D]`` input tensor.
        *args, **kwargs
            Captured from the parent encoder loop (e.g. head_mask,
            output_attentions).  Not forwarded to the inner block because
            ViTMAELayer does not accept them.

        Returns
        -------
        Same structure as the original block, with hidden states replaced
        by the adapter-modified version.
        """
        # 1. Run the original frozen encoder block.
        outputs = self.original_block(hidden_states)

        # 2. Extract hidden states from the heterogeneous return type.
        if isinstance(outputs, tuple):
            x = outputs[0]
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states
        else:
            x = outputs

        # 3. Apply the IBA adapter (adds the learned residual ΔH).
        x = self.adapter(x)

        # 4. Repackage to match the original return structure.
        if isinstance(outputs, tuple):
            return (x,) + outputs[1:]
        elif hasattr(outputs, "hidden_states"):
            try:
                outputs.hidden_states = x
                return outputs
            except (AttributeError, TypeError):
                return (x,)
        else:
            return x


# ═══════════════════════════════════════════════════════════════════════════
# 3. ADAPTER INJECTION ENTRY-POINT
# ═══════════════════════════════════════════════════════════════════════════

def inject_adapters(
    model: PreTrainedModel,
    bottleneck_dim: int = 64,
) -> PreTrainedModel:
    """Inject IBA adapters into every encoder layer of a pre-trained ViT.

    After this function returns:
      - The entire backbone is frozen (requires_grad = False).
      - Every encoder layer is wrapped in ``ViTBlockWithAdapter``.
      - Only the adapter parameters (~1 % of total) are trainable.

    Procedure
    ---------
    1. **Freeze** all backbone parameters.
    2. **Locate** the encoder module list
       (``model.vit.encoder.layer`` or ``model.encoder.layer``).
    3. For each layer: create an IBA adapter, match its device/dtype to
       the layer, and replace the layer with a wrapped version.
    4. Print a parameter audit (total / frozen / trainable).

    Parameters
    ----------
    model          : PreTrainedModel – e.g. ``ViTMAEForPreTraining``.
    bottleneck_dim : int             – Adapter bottleneck dim.  Default: 64.

    Returns
    -------
    PreTrainedModel
        The same model object, mutated in-place with adapters inserted.

    Raises
    ------
    AttributeError
        If the encoder module list cannot be found at the expected paths.
    """
    print(f"\n{'='*60}")
    print("[Adapter Injection] Starting")
    print(f"{'='*60}")

    # Step 1 — Freeze every backbone parameter.
    print("[Adapter Injection] Freezing backbone...")
    for param in model.parameters():
        param.requires_grad = False

    # Step 2 — Locate the encoder's ModuleList of transformer layers.
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        encoder = model.vit.encoder          # ViTMAEForPreTraining layout
        config = model.config
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        encoder = model.encoder              # generic BERT / ViT layout
        config = model.config
    else:
        raise AttributeError(
            "Unrecognised model structure — expected `model.vit.encoder` "
            "or `model.encoder` with a `.layer` ModuleList."
        )

    input_dim = config.hidden_size
    num_layers = len(encoder.layer)
    print(f"[Adapter Injection] hidden_dim={input_dim}, layers={num_layers}")
    print(f"[Adapter Injection] bottleneck_dim={bottleneck_dim}")

    # Step 3 — Wrap each encoder layer with an adapter.
    for i, layer in enumerate(encoder.layer):
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)

        # Match the adapter's device and dtype to the layer it wraps.
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)

        encoder.layer[i] = ViTBlockWithAdapter(
            original_block=layer, adapter=adapter
        )

        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Injected layer {i + 1}/{num_layers}")

    print("[Adapter Injection] Done (decoder layers untouched).")

    # Step 4 — Print parameter audit.
    _log_param_stats(model)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARAMETER AUDIT UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _log_param_stats(model: nn.Module) -> None:
    """Print a frozen / trainable parameter summary.

    A healthy adapter-tuned ViT-Base should show ≈1 % trainable.

    Parameters
    ----------
    model : nn.Module – The model to inspect.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    ratio = (trainable / total) * 100 if total > 0 else 0.0

    print(f"\n[Param Audit]")
    print(f"  Total:     {total:>12,}")
    print(f"  Frozen:    {frozen:>12,}")
    print(f"  Trainable: {trainable:>12,}  ({ratio:.2f}%)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 5. STANDALONE INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Quick smoke test: load ViT-MAE, inject adapters, forward pass."""
    print("[Test] Loading facebook/vit-mae-base...")
    try:
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        model = inject_adapters(model, bottleneck_dim=64)

        print("[Test] Running forward pass...")
        dummy = torch.randn(1, 3, 224, 224)
        dummy = dummy.to(next(model.parameters()).device)

        output = model(dummy)
        loss_val = output.loss.item() if hasattr(output, "loss") else "N/A"
        print(f"[Test] OK — loss={loss_val}")

    except Exception as e:
        print(f"[Test] Failed: {e}")