from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any
from transformers import PreTrainedModel, ViTMAEForPreTraining, ViTMAEModel



class IBA_Adapter(nn.Module):
    """
    Information-Bottlenecked Adapter (IBA) module.

    This module implements a bottleneck architecture (Down-project -> Activation -> Up-project)
    inserted into frozen networks to introduce trainable parameters for efficient adaptation.
    
    Architecture:
        Input [B, L, D] -> Linear(D, d) -> Activation -> Linear(d, D) -> Dropout -> + Residual
    
    Key Design Principles:
        1. **Bottleneck**: Compresses information to force the model to learn efficient features.
        2. **Identity Initialization**: The up-projection is initialized to zero, ensuring 
        the adapter starts as an identity function (Adapter(x) = 0). This prevents 
        "semantic shock" to the pre-trained backbone at the start of training.

    Attributes:
        input_dim (int): Original hidden dimension.
        bottleneck_dim (int): Compressed dimension.
        down_project (nn.Linear): Dimensionality reduction layer.
        activation (nn.Module): Non-linear activation function.
        up_project (nn.Linear): Dimensionality restoration layer.
        dropout (nn.Dropout): Regularization layer.
    """

    def __init__(
        self, 
        input_dim: int, 
        bottleneck_dim: int = 64, 
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU()
    ) -> None:
        """
        Initializes the IBA Adapter.

        Args:
            input_dim (int): The hidden dimension of the backbone model (e.g., 768 for ViT-Base).
            bottleneck_dim (int): The reduced dimension for the bottleneck. Lower values 
                compress information more (Information Bottleneck principle). Defaults to 64.
            dropout (float): Dropout probability applied after the up-projection. Defaults to 0.0.
            activation (nn.Module): Activation function to use between projections. Defaults to GELU.
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation

        # Down-projection: Compress semantic information
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Up-projection: Reconstruct features for the next layer
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies specific initialization strategies to ensure stable training start.
        """
        # 1. Kaiming Normal for down_project to maintain variance through the non-linearity.
        nn.init.kaiming_normal_(self.down_project.weight, nonlinearity='linear')
        
        # 2. Zeros for up_project. This ensures the adapter output is initially 0.
        #    result = Input + 0. This preserves the pre-trained behavior exactly.
        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch_Size, Seq_Len, Hidden_Dim].

        Returns:
            torch.Tensor: Adapted features of the same shape as input.
        """
        residual = x
        
        # Bottleneck compression
        x = self.down_project(x)
        x = self.activation(x)
        
        # Note: Variational noise injection (e.g., for Zeus/V4 methods) 
        # would typically be applied here if probabilistic modeling is desired.
        
        # Reconstruction & Regularization
        x = self.up_project(x)
        x = self.dropout(x)
        
        # Residual connection preserves original features while adding adaptation
        return residual + x

    def __repr__(self) -> str:
        """Custom string representation for easier debugging."""
        return f"IBA_Adapter(in={self.input_dim}, btl={self.bottleneck_dim})"


class ViTBlockWithAdapter(nn.Module):
    """
    Wrapper class to inject an Adapter into a Hugging Face ViTLayer.

    It intercepts the output of the original frozen block, passes the hidden states
    through the adapter, and repackages the output to match Hugging Face's 
    return signature exactly.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        """
        Args:
            original_block (nn.Module): The original, frozen Transformer block.
            adapter (IBA_Adapter): The trainable adapter instance.
        """
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False,
        **kwargs: Any
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]:
        """
        Forward pass matching standard Hugging Face ViTLayer signature.
        
        Args:
            hidden_states (torch.Tensor): Input tensor.
            head_mask (Optional[torch.Tensor]): Mask for attention heads.
            output_attentions (bool): Whether to return attention weights.
            **kwargs: Additional arguments required by specific HF implementations.

        Returns:
            Tuple containing the modified hidden state and optional attention weights.
        """
        # 1. Run the original frozen ViT Block
        # HF blocks typically return a tuple: (hidden_states, attention_weights (optional), ...)
        # We explicitly EXCLUDE output_attentions from the call as ViTMAE doesn't support it by default
        outputs = self.original_block(
            hidden_states, 
            head_mask=head_mask, 
            **kwargs
        )
        
        # 2. Extract Hidden States and Logic for Return Packaging
        if isinstance(outputs, tuple):
            x = outputs[0]
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states
        else:
            x = outputs
        
        # 3. Apply the IBA Adapter
        x = self.adapter(x)
        
        # 4. Repackage output to maintain compatibility with HF pipeline
        if isinstance(outputs, tuple):
            # Reconstruct the tuple with the adapted hidden state
            return (x,) + outputs[1:]
        elif hasattr(outputs, "hidden_states"):
            # If it's a ModelOutput, we try to create a new one or modify in place?
            # Creating a new one is safer but requires knowing the class.
            # Mutating in place works if it's mutable.
            # A simpler hack that often works for HF is returning a tuple if it came as ModelOutput,
            # but some downstream layers check isinstance(ModelOutput).
            # However, standard ViTEncoder loop handles tuple or ModelOutput.
            # But if it wasn't a tuple originally, let's try to return what it expects.
            # Most robust: Just update the hidden_states attribute if mutable.
            try:
                outputs.hidden_states = x
                return outputs
            except:
                # If immutable, we fallback to tuple which HF usually accepts
                return (x,) 
        else:
            # It was a Tensor, return a Tensor
            return x


def inject_adapters(model: PreTrainedModel, bottleneck_dim: int = 64) -> PreTrainedModel:
    """
    Injects IBA Adapters into the Encoder of a ViTMAE (or similar) model.

    This function performs the following operations:
    1. Freezes all existing parameters in the model.
    2. Identifies the Encoder layers.
    3. Wraps each layer with `ViTBlockWithAdapter`.
    4. Unfreezes ONLY the new Adapter parameters.

    Args:
        model (PreTrainedModel): The Hugging Face ViTMAE model instance.
        bottleneck_dim (int): Dimension of the adapter bottleneck.

    Returns:
        PreTrainedModel: The modified model with adapters injected.
    
    Raises:
        AttributeError: If the model structure does not match standard ViT hierarchies.
    """
    print(f"\n{'='*60}")
    print(f"[System] Starting Adapter Injection Procedure")
    print(f"{'='*60}")

    # 1. Freeze the entire model backbone
    print("[Config] Freezing original backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Locate the Encoder
    # We verify structure to prevent runtime errors later
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        # Standard ViTMAE structure
        encoder = model.vit.encoder
        config = model.config
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        # Generic BERT/ViT structure fallback
        encoder = model.encoder
        config = model.config
    else:
        raise AttributeError(
            "Could not locate 'encoder.layer'. "
            "Model structure unknown (expected 'vit.encoder' or 'encoder')."
        )

    input_dim = config.hidden_size
    num_layers = len(encoder.layer)

    print(f"[Config] Model Config: Hidden Dim={input_dim}, Layers={num_layers}")
    print(f"[Config] Adapter Config: Bottleneck Dim={bottleneck_dim}")

    # 3. Iterate and Replace
    print("[Action] Injecting adapters into encoder layers...")
    
    for i, layer in enumerate(encoder.layer):
        # Instantiate the adapter
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        
        # CRITICAL: Ensure adapter is on the same device and dtype as the layer it wraps.
        # This handles cases where the model is already on GPU or in FP16/BF16.
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        
        # Wrap the original layer
        wrapped_layer = ViTBlockWithAdapter(original_block=layer, adapter=adapter)
        
        # Mutate the ModuleList in-place
        encoder.layer[i] = wrapped_layer
        
        # Simple progress indicator for large models
        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Processed layer {i + 1}/{num_layers}")

    print(f"[System] Injection Complete. Decoder layers ignored (if present).")
    
    # 4. Verification of Trainable Parameters
    count_trainable_params(model)
    
    return model


def count_trainable_params(model: nn.Module) -> None:
    """
    Utility to calculate and print the count of frozen vs trainable parameters.
    
    Args:
        model (nn.Module): The model to audit.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0
    
    print(f"\n[Stats] Parameter Audit:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Frozen Backbone:      {frozen_params:,}")
    print(f"  - Trainable (Adapters): {trainable_params:,}")
    print(f"  - Trainable Ratio:      {ratio:.2f}%")
    print(f"{'='*60}\n")


# =============================================================================
# Main Execution Block (For Testing)
# =============================================================================
if __name__ == "__main__":
    # Simulate loading a model (mocking correct behavior if transformers is installed)
    print("[Main] Loading pre-trained ViTMAE...")
    try:
        # NOTE: Requires `pip install transformers`
        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        
        # Inject Adapters
        model = inject_adapters(model, bottleneck_dim=64)
        
        # Sanity Check: Forward pass
        print("[Main] Running dummy forward pass to verify graph integrity...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # Forward pass (ensure gradients flow through adapters)
        output = model(dummy_input)
        
        loss_val = output.loss.item() if hasattr(output, "loss") else "N/A"
        print(f"[Success] Forward pass complete. Loss: {loss_val}")
        
    except ImportError:
        print("[Error] 'transformers' library not found. Please install it to run this test.")
    except Exception as e:
        print(f"[Error] An error occurred during execution: {e}")