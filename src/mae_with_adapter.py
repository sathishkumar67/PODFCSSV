"""
Parameter-Efficient Fine-Tuning via Information-Bottlenecked Adapters (IBA).

This module implements an adapter-based approach to fine-tune large
Vision Transformer (ViT) models—specifically Hugging Face's ViTMAE—without
modifying the pre-trained backbone weights. Instead, lightweight bottleneck
modules called "adapters" are injected after each frozen encoder layer.
Only these adapter parameters (~1 % of total) are trained, making the
approach highly communication-efficient for Federated Learning.

The design draws on three pillars:

1. **Bottleneck Adapters** (Houlsby et al., ICML 2019):
   Small MLP modules inserted inside transformer blocks that learn a
   task-specific residual correction to the frozen hidden states.

2. **Information Bottleneck** (Tishby et al., 2000):
   The down-projection forces the adapter to retain only information
   that is relevant to the downstream objective, discarding noise.

3. **Masked Autoencoders** (He et al., CVPR 2022):
   The frozen backbone is a ViTMAE model pre-trained with masked image
   modelling, providing strong visual features as the starting point.

Module Contents
---------------
- ``IBA_Adapter``           : The bottleneck adapter module itself.
- ``ViTBlockWithAdapter``   : A wrapper that pairs a frozen encoder layer
                              with a trainable adapter.
- ``inject_adapters``       : The main entry-point that surgically inserts
                              adapters into an existing pre-trained model.
- ``_log_param_stats``      : A small diagnostic utility that prints the
                              frozen-vs-trainable parameter split.

References
----------
[1] Houlsby, N. et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019.
[2] Tishby, N. et al., "The Information Bottleneck Method", 2000.
[3] He, K. et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any
from transformers import PreTrainedModel, ViTMAEForPreTraining


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ADAPTER MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class IBA_Adapter(nn.Module):
    """
    Information-Bottlenecked Adapter (IBA) for Parameter-Efficient Fine-Tuning.

    This module implements a two-layer bottleneck MLP that is inserted after
    each frozen transformer block. It learns a small residual correction
    ΔH to the hidden states, so that the output of a wrapped block becomes:

        H_out = H_frozen + ΔH

    Architecture (step by step)
    ---------------------------
    Given an input tensor H of shape (Batch, SeqLen, D):

        Step 1 — Down-Project  : H is linearly projected from D dimensions
                                 down to a much smaller d dimensions (d << D).
                                 This is the "information bottleneck" that
                                 forces the adapter to retain only the most
                                 task-relevant information.

        Step 2 — Activation    : A non-linear activation (GELU by default)
                                 is applied to the compressed representation,
                                 allowing the adapter to model complex
                                 non-linear relationships.

        Step 3 — Up-Project    : The d-dimensional representation is projected
                                 back up to the original D dimensions,
                                 reconstructing a full-size correction vector.

        Step 4 — Dropout       : Dropout regularization is applied to prevent
                                 the adapter from overfitting on small local
                                 datasets (critical in Federated Learning).

        Step 5 — Residual Add  : The adapter output (ΔH) is added back to
                                 the original input H, forming a skip connection.

    Identity Initialization
    -----------------------
    A crucial detail for stable training: the Up-Projection layer's weights
    and biases are initialized to **zero**. This means that at the very first
    training step, the adapter produces ΔH = 0, so the wrapped block behaves
    exactly like the original frozen block. The adapters then learn to
    gradually modify the features as training progresses, avoiding any
    "semantic shock" that would disrupt the pre-trained representations.

    Parameters
    ----------
    input_dim : int
        The hidden dimension of the backbone model (e.g. 768 for ViT-Base/16).
    bottleneck_dim : int
        The dimension of the compressed bottleneck space. Smaller values
        give fewer parameters but may limit the adapter's capacity.
        Default: 64.
    dropout : float
        Dropout probability applied after the up-projection. Default: 0.0.
    activation : nn.Module
        Non-linear activation between the two projections. Default: GELU.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU()
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation

        # Step 1 component: compress D → d
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)

        # Step 3 component: reconstruct d → D
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=True)

        # Step 4 component: regularization
        self.dropout = nn.Dropout(dropout)

        # Apply the identity-init strategy immediately after construction
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply carefully chosen weight initialization for training stability.

        Strategy
        --------
        - **Down-Projection (W_down)**:
            Kaiming Normal initialization with ``fan_out`` mode and ``relu``
            nonlinearity. This preserves the variance of activations flowing
            through the bottleneck, preventing vanishing or exploding
            gradients in the early training steps.

        - **Down-Projection Bias (b_down)**:
            Initialized to zero so that the bias does not introduce any
            constant offset before the activation function.

        - **Up-Projection (W_up) — ZERO INIT**:
            Both weights and bias are set to zero. This is the single most
            important initialization choice in the adapter: it guarantees
            that the adapter output is identically zero at step 0, making
            the wrapped block behave like the original frozen block.
            Without this, random initialization would immediately corrupt
            the pre-trained features.

        - **Up-Projection Bias (b_up) — ZERO INIT**:
            Also set to zero to complete the identity initialization.
        """
        # Down-projection: Kaiming Normal maintains activation variance
        nn.init.kaiming_normal_(self.down_project.weight, mode='fan_out', nonlinearity='relu')
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)

        # Up-projection: Zero initialization for identity behaviour at step 0
        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the adapted hidden states via the bottleneck residual path.

        Mathematically:
            H' = H + Dropout( W_up · σ( W_down · H + b_down ) + b_up )

        Parameters
        ----------
        x : torch.Tensor
            Input hidden states from the transformer block.
            Shape: ``(Batch, SeqLen, D)``

        Returns
        -------
        torch.Tensor
            Adapted hidden states with the same shape as the input.
        """
        # Save the original input for the residual (skip) connection
        residual = x

        # Step 1: Compress from D dimensions down to d dimensions
        x = self.down_project(x)

        # Step 2: Apply non-linear activation
        x = self.activation(x)

        # Step 3: Reconstruct back to D dimensions
        x = self.up_project(x)

        # Step 4: Apply dropout for regularization
        x = self.dropout(x)

        # Step 5: Add the residual — this is the skip connection
        return residual + x

    def __repr__(self) -> str:
        return f"IBA_Adapter(in_features={self.input_dim}, bottleneck={self.bottleneck_dim})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ADAPTER-INJECTED TRANSFORMER BLOCK WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class ViTBlockWithAdapter(nn.Module):
    """
    Wraps a single frozen ViT encoder layer with a trainable IBA Adapter.

    This wrapper transparently replaces the original layer inside the
    encoder's ``nn.ModuleList``. When the encoder calls ``layer(hidden_states)``,
    this wrapper:

        1. Runs the **original frozen block** to produce its normal output
           (self-attention → feed-forward → layer norm).

        2. **Extracts the hidden states** from the output, handling
           Hugging Face's heterogeneous return types (tuple, ModelOutput,
           or raw tensor).

        3. **Applies the IBA Adapter** to the hidden states, adding the
           learned residual correction ΔH.

        4. **Repackages the result** into the same structure that the
           original block returned, so that downstream layers and the
           model pipeline remain completely unaware of the modification.

    Parameters
    ----------
    original_block : nn.Module
        The frozen transformer layer being wrapped (e.g. ``ViTMAELayer``).
    adapter : IBA_Adapter
        The trainable adapter instance to apply after the block.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        super().__init__()
        # Store the frozen block and trainable adapter as sub-modules
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]:
        """
        Execute the frozen block followed by the trainable adapter.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape ``(Batch, SeqLen, D)``.
        *args, **kwargs
            Additional arguments forwarded from the parent ViTEncoder loop
            (e.g. ``head_mask``, ``output_attentions``). These are captured
            here to prevent a TypeError but are intentionally NOT passed to
            the inner block, because ViTMAELayer does not accept them.

        Returns
        -------
        Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]
            The output in the same format as the original block, but with
            the hidden states replaced by the adapter-modified version.
        """
        # Step 1: Run the original frozen encoder block
        #         Only hidden_states is passed — extra args like head_mask
        #         are dropped because ViTMAELayer's forward() rejects them.
        outputs = self.original_block(hidden_states)

        # Step 2: Extract hidden states from the return value
        #         Hugging Face blocks may return:
        #           - A tuple:       (hidden_states, attention_weights, ...)
        #           - A ModelOutput:  object with .hidden_states attribute
        #           - A raw tensor:   just the hidden states directly
        if isinstance(outputs, tuple):
            x = outputs[0]
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states
        else:
            x = outputs

        # Step 3: Apply the IBA adapter to add the learned correction ΔH
        x = self.adapter(x)

        # Step 4: Repackage the result to match the original return type
        #         so that downstream layers receive the expected structure
        if isinstance(outputs, tuple):
            # Replace the first element (hidden states) while keeping the rest
            return (x,) + outputs[1:]
        elif hasattr(outputs, "hidden_states"):
            # Try to update the ModelOutput object in place
            try:
                outputs.hidden_states = x
                return outputs
            except (AttributeError, TypeError):
                # Some ModelOutput objects are immutable — fall back to tuple
                return (x,)
        else:
            # Raw tensor output — return the adapted tensor directly
            return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ADAPTER INJECTION UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def inject_adapters(model: PreTrainedModel, bottleneck_dim: int = 64) -> PreTrainedModel:
    """
    Inject IBA adapters into every encoder layer of a pre-trained ViT model.

    This function performs the surgical modification needed to convert a
    standard pre-trained model into an adapter-tuned model suitable for
    Federated Learning. After this function returns, training will only
    update ~1 % of the total parameters (the adapters), while the backbone
    remains frozen.

    Procedure (step by step)
    ------------------------
    Step 1 — **Freeze the backbone**:
        Set ``requires_grad = False`` for every parameter in the model.
        This ensures the pre-trained weights are never modified during
        training, preserving the learned representations.

    Step 2 — **Locate the encoder**:
        Inspect the model structure to find the list of transformer layers.
        Supports two common layouts:
        - ``model.vit.encoder.layer`` (ViTMAEForPreTraining)
        - ``model.encoder.layer``     (generic BERT/ViT)

    Step 3 — **Wrap each layer**:
        For every transformer layer in the encoder:
        a) Create a fresh ``IBA_Adapter`` with the correct dimensions.
        b) Move the adapter to the same device and dtype as the layer
           (handles GPU placement and mixed precision automatically).
        c) Wrap the original layer in ``ViTBlockWithAdapter``.
        d) Replace the layer in the encoder's ``nn.ModuleList``.

    Step 4 — **Audit parameters**:
        Print a summary of total, frozen, and trainable parameters to
        verify that only the adapters are trainable.

    Parameters
    ----------
    model : PreTrainedModel
        A Hugging Face pre-trained model (e.g. ``ViTMAEForPreTraining``).
    bottleneck_dim : int
        The bottleneck dimension for every adapter. Default: 64.

    Returns
    -------
    PreTrainedModel
        The same model object, mutated in-place with adapters injected
        and the backbone frozen.

    Raises
    ------
    AttributeError
        If the model structure is not recognized (i.e. cannot find the
        encoder's layer list at the expected locations).
    """
    print(f"\n{'='*60}")
    print(f"[Adapter Injection] Starting procedure")
    print(f"{'='*60}")

    # Step 1 — Freeze every parameter in the backbone
    print("[Adapter Injection] Freezing backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Step 2 — Locate the encoder's ModuleList of transformer layers
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        # ViTMAEForPreTraining layout: model.vit.encoder.layer
        encoder = model.vit.encoder
        config = model.config
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # Generic BERT/ViT layout: model.encoder.layer
        encoder = model.encoder
        config = model.config
    else:
        raise AttributeError(
            "Unrecognized model structure — expected `model.vit.encoder` or "
            "`model.encoder` with a `.layer` ModuleList."
        )

    input_dim = config.hidden_size
    num_layers = len(encoder.layer)
    print(f"[Adapter Injection] Backbone: hidden_dim={input_dim}, num_layers={num_layers}")
    print(f"[Adapter Injection] Adapter:  bottleneck_dim={bottleneck_dim}")

    # Step 3 — Iterate through every encoder layer and wrap it
    for i, layer in enumerate(encoder.layer):
        # 3a. Create a fresh adapter with matching dimensions
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)

        # 3b. Move adapter to the same device/dtype as the layer it wraps
        #     (this handles GPU placement and FP16/BF16 mixed precision)
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)

        # 3c. Replace the original layer with the adapter-wrapped version
        encoder.layer[i] = ViTBlockWithAdapter(original_block=layer, adapter=adapter)

        # Progress logging every 4 layers and at the end
        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Injected layer {i + 1}/{num_layers}")

    print("[Adapter Injection] Complete (decoder layers untouched).")

    # Step 4 — Print parameter audit to verify the freeze/inject split
    _log_param_stats(model)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PARAMETER AUDIT UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def _log_param_stats(model: nn.Module) -> None:
    """
    Print a summary of frozen vs. trainable parameters.

    This diagnostic utility is called after adapter injection to verify
    that the backbone is fully frozen and only the adapter weights are
    trainable. A healthy adapter-tuned ViT-Base should show approximately
    ~1 % trainable parameters.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STANDALONE INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick integration test to verify the adapter injection pipeline.

    Steps
    -----
    1. Load the pre-trained ViTMAE model from Hugging Face.
    2. Run ``inject_adapters`` to freeze the backbone and add adapters.
    3. Create a dummy image tensor and run a forward pass.
    4. Verify that the model produces a valid loss without errors.

    Expected output:
        - Parameter audit showing ~1% trainable (adapter) parameters.
        - A finite loss value from the forward pass.
    """
    print("[Test] Loading facebook/vit-mae-base...")
    try:
        # Load the pre-trained ViTMAE model
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        # Inject adapters into all 12 encoder layers
        model = inject_adapters(model, bottleneck_dim=64)

        # Create a dummy input image: 1 image, 3 channels, 224x224 pixels
        print("[Test] Running forward pass sanity check...")
        dummy = torch.randn(1, 3, 224, 224)
        device = next(model.parameters()).device
        dummy = dummy.to(device)

        # Forward pass — the model should compute MAE loss internally
        output = model(dummy)
        loss_val = output.loss.item() if hasattr(output, "loss") else "N/A"
        print(f"[Test] Forward pass OK — loss={loss_val}")

    except Exception as e:
        print(f"[Test] Failed: {e}")