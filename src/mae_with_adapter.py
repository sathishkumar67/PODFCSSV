"""
Parameter-Efficient Fine-Tuning via Information-Bottleneck Adapters (IBA).

This module implements the adapter-based fine-tuning strategy for scaling
Federated Continual Learning. By injecting lightweight, low-rank Multi-Layer 
Perceptrons (MLPs) into a frozen pre-trained Vision Transformer (ViT-MAE), 
we constrain the trainable parameter footprint to ~1% of the original model. 
This drastic reduction is mathematically essential for Federated Learning,
as it directly minimizes end-to-end communication bandwidth and enables
resource-constrained edge clients to engage in locally isolated training.

Theoretical Foundations
-----------------------
1. **Information Bottleneck Principals (Tishby et al., 2000):** 
   By compressing high-dimensional hidden states $H \in \mathbb{R}^{D}$ 
   down to a low-dimensional manifold $h_{down} \in \mathbb{R}^{d}$ where $d \ll D$, 
   the network functions as a regularized noise filter, intrinsically 
   discarding task-irrelevant semantics and retaining essential concepts.

2. **Residual Adapter Tuning (Houlsby et al., 2019):**
   The architecture strictly adheres to a sequential insertion heuristic, 
   acting as a residual correction to the frozen representation. 
   The calculation is formulated as: $H' = H + W_{up}\sigma(W_{down}H)$.

3. **Identity Initialization Strategy:**
   At $t=0$, the adapter must act as a perfect identity function $H' = H$ 
   to prevent immediate catastrophic disruption of the highly converged 
   pre-trained weights. Thus, $W_{up}$ must be initialized to exact zero.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any, Tuple, Union
from transformers import PreTrainedModel, ViTMAEForPreTraining

class IBA_Adapter(nn.Module):
    """
    Information-Bottleneck Adapter Block.

    This module learns a low-rank residual projection ΔH decoupled from 
    the massively parameterized Transformer backbone. It compresses, 
    activates, and decompresses the hidden state sequence.

    Mathematical Formulation
    ------------------------
    Given a hidden state tensor $H \in \mathbb{R}^{B \\times L \\times D}$:
    1. $h_{down} = H W_{down}^T + b_{down}$ : $\mathbb{R}^D \rightarrow \mathbb{R}^d$
    2. $h_{act} = \text{GELU}(h_{down})$
    3. $h_{up} = h_{act} W_{up}^T + b_{up}$ : $\mathbb{R}^d \rightarrow \mathbb{R}^D$
    4. $H' = H + \text{Dropout}(h_{up})$

    Initialization Strategy
    -----------------------
    $W_{down}$ follows Kaiming Normal (He) initialization to maintain variance.
    $W_{up}$ is perfectly zero-initialized. This guarantees $\frac{\partial \mathcal{L}}{\partial W_{up}} = 0$ 
    at the very first forward pass, acting as an identity skip-connection.
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
        
        # Non-linear gating
        self.activation = activation if activation is not None else nn.GELU()

        # D -> d: High-dimensional spatial compression 
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)

        # d -> D: Information decompression back to Transformer hidden dimension
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=True)

        # Stochastic regularization
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Enforce identity equivalence at convergence step 0.
        """
        # Maintain variance of standard activations across the bottleneck
        nn.init.kaiming_normal_(
            self.down_project.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.down_project.bias is not None:
            # Positive constant bias prevents dead GELU neurons early on
            nn.init.constant_(self.down_project.bias, 0.1)

        # Enforce Identity: H' = H + 0
        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the residual adaptation step $H_{n+1} = H_n + \Delta H$.
        """
        residual = x                      
        x = self.down_project(x)          
        x = self.activation(x)            
        x = self.up_project(x)            
        x = self.dropout(x)               
        return residual + x               


class ViTBlockWithAdapter(nn.Module):
    """
    Polymorphic wrapper class around a frozen ViT Encoder layer.
    
    This abstracts away the HuggingFace transformer implementation, 
    dynamically executing the frozen self-attention/FFN block, extracting 
    the resulting sequence hidden states, passing them through the IBA, 
    and repackaging the output to perfectly match the original Tuple format.
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
        # 1. Forward propagation through the frozen multi-head attention graph
        outputs = self.original_block(hidden_states)

        # 2. Heuristic extraction of the `hidden_states` tensor from varied return types
        if isinstance(outputs, tuple):
            x = outputs[0]
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states
        else:
            x = outputs

        # 3. Residual Bottleneck correction
        x = self.adapter(x)

        # 4. State reconstruction compliant with standard Transformers API
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


def inject_adapters(
    model: PreTrainedModel,
    bottleneck_dim: int = 64,
) -> PreTrainedModel:
    """
    Mutates the PreTrainedModel computation graph in-place.

    Execution Flow
    --------------
    1. Traverses the entire parameter tree and forcefully disables `requires_grad`.
    2. Identifies the encoder `ModuleList` structure (handling generic ViT implementations).
    3. Wraps the upper half of the encoder layers with `ViTBlockWithAdapter` components.
       (Injecting into the upper half is an empirical standard: lower layers extract 
        generic edges/textures, while upper layers represent domain-specific abstractions
        which are the target of our continual learning tasks).
    """
    print(f"\\n{'='*60}")
    print("[Adapter Injection] Initializing PEFT Graph Mutator")
    print(f"{'='*60}")

    # Aggressive Autograd detachment: fully freeze the foundation model
    print("[Adapter Injection] Detaching foundation backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False

    # Introspect standard transformer layout
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        encoder = model.vit.encoder          
        config = model.config
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        encoder = model.encoder              
        config = model.config
    else:
        raise AttributeError(
            "Graph introspection failed: Encoders `module.vit.encoder` "
            "or `module.encoder` missing from definition hierarchy."
        )

    input_dim = config.hidden_size
    num_layers = len(encoder.layer)
    
    # PEFT Heuristic: Tune the upper representations
    inject_start_layer = num_layers // 2

    print(f"[Adapter Injection] Representation Dim: {input_dim}, Encoder Depth: {num_layers}")
    print(f"[Adapter Injection] Target Insertion Range: Layers {inject_start_layer} -> {num_layers-1}")
    print(f"[Adapter Injection] Rank Constraint: r={bottleneck_dim}")

    # Mutate the sequential ModuleList
    for i in range(inject_start_layer, num_layers):
        layer = encoder.layer[i]
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)

        # Enforce tensor properties to match the attached layer precisely
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)

        encoder.layer[i] = ViTBlockWithAdapter(
            original_block=layer, adapter=adapter
        )

        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Successfully Injected Layer Module {i + 1}/{num_layers}")

    print("[Adapter Injection] Mutator Pass Complete.")
    _log_param_stats(model)
    return model


def _log_param_stats(model: nn.Module) -> None:
    """
    Calculates the topological parameter density ratio to verify the
    low-rank parameter footprint constraints required for federated networking.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    ratio = (trainable / total) * 100 if total > 0 else 0.0

    print(f"\\n[Topological Density Audit]")
    print(f"  Total Params:        {total:>12,}")
    print(f"  Frozen Backbone:     {frozen:>12,}")
    print(f"  Trainable Adapters:  {trainable:>12,} ({ratio:.2f}% of Total Mass)")
    print(f"{'='*60}\\n")


if __name__ == "__main__":
    """Integration integrity test."""
    print("[Verification] Loading Pre-trained Distribution from HuggingFace...")
    try:
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        model = inject_adapters(model, bottleneck_dim=64)

        print("[Verification] Synthesizing forward tensor propagation...")
        dummy = torch.randn(1, 3, 224, 224)
        dummy = dummy.to(next(model.parameters()).device)

        output = model(dummy)
        loss_val = output.loss.item() if hasattr(output, "loss") else "N/A"
        print(f"[Verification] OK — Pre-training Loss Response: {loss_val}")

    except Exception as e:
        print(f"[Verification] Graph Integration Failed: {e}")