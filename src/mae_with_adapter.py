from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Any
from transformers import PreTrainedModel, ViTMAEForPreTraining

class IBA_Adapter(nn.Module):
    """
    Information-Bottlenecked Adapter (IBA) Module for Efficient parameter-efficient Fine-Tuning.

    Overview
    --------
    The IBA Adapter is a lightweight neural network module designed to be inserted 
    into pre-trained frozen backbones (like ViT or BERT). It introduces a small 
    number of trainable parameters to adapt the model to new tasks (or domains) 
    without retraining the entire massive network.

    Architectural Design
    --------------------
    The adapter follows a "Bottleneck" structure to minimize parameter count while 
    maximizing adaptation capability. Ideally, it compresses high-dimensional 
    semantic features into a compact representation and then reconstructs them.

    Structure:
        Input (D) -> Down-Projection (d) -> Non-Linearity -> Up-Projection (D) -> Dropout -> + Residual

    Key Design Principles
    ---------------------
    1.  **Information Bottleneck**: By projecting high-dimensional features (D) 
        down to a smaller dimension (d), the model is forced to learn only the 
        most salient features relevant to the specific task, ignoring noise.
    
    2.  **Identity Initialization**: A critical stability feature for Federated Learning.
        -   The Up-Projection layer is initialized with **zeros**.
        -   This ensures that at initialization (step 0), the adapter output is exactly 0.
        -   Result: `Layer(x) + Adapter(x) = Layer(x) + 0 = Layer(x)`.
        -   This prevents "catastrophic forgetting" or "semantic shock" where random 
            initialization would distort the carefully learned features of the 
            pre-trained backbone.

    Attributes
    ----------
    input_dim : int
        The dimensionality of the input features (Hidden Size of the backbone).
    bottleneck_dim : int
        The dimensionality of the compressed bottleneck space.
    down_project : nn.Linear
        Linear layer reducing dimension from `input_dim` to `bottleneck_dim`.
    activation : nn.Module
        Non-linear activation function (e.g., GELU, ReLU) to enable learning complex patterns.
    up_project : nn.Linear
        Linear layer restoring dimension from `bottleneck_dim` back to `input_dim`.
    dropout : nn.Dropout
        Dropout layer for regularization during training.
    """

    def __init__(
        self, 
        input_dim: int, 
        bottleneck_dim: int = 64, 
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU()
    ) -> None:
        """
        Initializes the IBA Adapter with the specified configuration.

        Parameters
        ----------
        input_dim : int
            The hidden size of the pre-trained model (e.g., 768 for ViT-Base).
        bottleneck_dim : int, optional
            The size of the bottleneck. Smaller values result in fewer parameters 
            but may limit capacity. Defaults to 64.
        dropout : float, optional
            Dropout probability applied to the output of the adapter. Defaults to 0.0.
        activation : nn.Module, optional
            The activation function to use within the bottleneck. Defaults to nn.GELU().
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation

        # 1. Down-Projection Layer
        # Compresses the input semantic vector into the bottleneck space (D -> d).
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)
        
        # 2. Up-Projection Layer
        # Reconstructs the semantic vector from the bottleneck space (d -> D).
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=True)
        
        # 3. Regularization
        self.dropout = nn.Dropout(dropout)
        
        # 4. Weight Initialization
        # Apply strict initialization rules to ensure stable convergence.
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies robust initialization strategies for the adapter layers.

        Initialization Strategy
        -----------------------
        1.  **Down-Projection**: 
            -   **Weights**: Kaiming Normal (He Initialization) with 'relu' nonlinearity mode. 
                This maintains the variance of activations through the layer, preventing 
                vanishing/exploding gradients in the bottleneck.
            -   **Bias**: Initialized to Zero.

        2.  **Up-Projection**:
            -   **Weights & Bias**: Zero Initialization. 
            -   **Reasoning**: This ensures the adapter contributes nothing (0) at the 
                very start of training. The model initially behaves exactly like the 
                original frozen backbone, allowing the adapter to gradually learn 
                modifications rather than starting with random noise.
        """
        # A. Down-Projection Initialization
        # We use 'mode=fan_out' and 'nonlinearity=relu' as a robust default for linear layers followed by activations.
        nn.init.kaiming_normal_(self.down_project.weight, mode='fan_out', nonlinearity='relu')
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)
        
        # B. Up-Projection Initialization (Identity Init)
        # This is the most critical step for stability in fine-tuning.
        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the adapter module.

        The flow is:
        Input -> [Down Project] -> [Activation] -> [Up Project] -> [Dropout] -> + Input (Residual)

        Parameters
        ----------
        x : torch.Tensor
            The input hidden states from the transformer block.
            Shape: [Batch_Size, Sequence_Length, Hidden_Dimension]

        Returns
        -------
        torch.Tensor
            The adapted hidden states, with the exact same shape as the input.
        """
        # Save the original input for the residual connection
        residual = x
        
        # 1. Compression: Project down to bottleneck dimension
        x = self.down_project(x)
        
        # 2. Non-Linearity: Apply activation function
        x = self.activation(x)
        
        # 3. Reconstruction: Project back up to original dimension
        x = self.up_project(x)
        
        # 4. Regularization: Apply dropout
        x = self.dropout(x)
        
        # 5. Residual Connection: Add the learned delta to the original features
        return residual + x

    def __repr__(self) -> str:
        """
        Returns a string representation of the module for debugging purposes.
        """
        return f"IBA_Adapter(in_features={self.input_dim}, bottleneck={self.bottleneck_dim})"


class ViTBlockWithAdapter(nn.Module):
    """
    Wrapper Module to Inject an Adapter into a Frozen Transformer Block.

    Purpose
    -------
    This class wraps an existing (frozen) `ViTLayer` or `BertLayer` from the 
    Hugging Face library. It intercepts the forward pass, allows the original 
    block to process the input, and then applies the `IBA_Adapter` to the output 
    hidden states.

    It ensures compatibility with Hugging Face's complex return types 
    (tuples vs ModelOutput objects) so that the rest of the model pipeline 
    remains unaware of the modification.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        """
        Wraps a transformer block with an adapter.

        Parameters
        ----------
        original_block : nn.Module
            The original, frozen Transformer block (e.g., `ViTLayer`).
        adapter : IBA_Adapter
            The trainable adapter instance to be applied after the block.
        """
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self, 
        hidden_states: torch.Tensor,
        *args,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]:
        """
        Forward pass that mimics the signature of a standard Hugging Face ViTLayer.
        
        Note: arguments like `head_mask` or `output_attentions` are implicitly handled 
        or omitted based on the specific requirements of the backbone (e.g., ViTMAE 
        does not support `head_mask`).

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [Batch, SeqLen, Dim].
        args : tuple
            Variable positional arguments required by the pipeline.
        kwargs : dict
            Variable keyword arguments required by the pipeline.

        Returns
        -------
        Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any]]
            The output tuple expected by the transformer model, containing the 
            adapted hidden states and potentially attention weights.
        """
        # 1. Execute the Original Frozen Block
        # We explicitly ignored *args and **kwargs (like head_mask) on purpose 
        # because ViTMAE layers typically reject them.
        outputs = self.original_block(hidden_states)
        
        # 2. Extract the Hidden States
        # Hugging Face models can return:
        # - A tuple: (hidden_states, attention_weights, ...)
        # - A ModelOutput object (like BaseModelOutput)
        # - A raw Tensor
        if isinstance(outputs, tuple):
            x = outputs[0]
        elif hasattr(outputs, "hidden_states"):
            x = outputs.hidden_states
        else:
            x = outputs
        
        # 3. Apply the Adapter
        # The adapter modifies the features in-place (conceptually) via residual connection.
        x = self.adapter(x)
        
        # 4. Repackage Result
        # We must return exactly what the parent model expects to avoid breaking the pipeline.
        if isinstance(outputs, tuple):
            # Reconstruct the tuple: (new_hidden_states, *rest_of_tuple)
            return (x,) + outputs[1:]
        elif hasattr(outputs, "hidden_states"):
            # If it's a ModelOutput object, we try to update it.
            # Some objects are immutable or downstream layers check strict types.
            try:
                outputs.hidden_states = x
                return outputs
            except:
                # Fallback: Return a tuple, which HF pipelines usually accept as a valid alternative.
                return (x,) 
        else:
            # If input was just a Tensor, return the new Tensor.
            return x


def inject_adapters(model: PreTrainedModel, bottleneck_dim: int = 64) -> PreTrainedModel:
    """
    Core Utility: Injects IBA Adapters into the Encoder of a Pre-trained Model.

    This function performs the precise surgery needed to convert a standard 
    pre-trained model (like ViTMAE) into an adapter-tuned model.

    Procedure
    ---------
    1.  **Freeze Backbone**: Sets `requires_grad=False` for ALL original parameters.
    2.  **Locate Encoder**: Identifies the list of transformer layers (`encoder.layer`).
    3.  **Inject Adapters**:
        -   Iterates through each layer.
        -   Creates a new `IBA_Adapter` matching the layer's dimensions.
        -   Wraps the original layer in `ViTBlockWithAdapter`.
        -   Replaces the layer in the model's module list.
    4.  **Activate Adapters**: Ensures only the new adapter parameters are trainable.

    Parameters
    ----------
    model : PreTrainedModel
        The Hugging Face model instance (e.g., `ViTMAEForPreTraining`).
    bottleneck_dim : int, optional
        The dimension of the adapter bottleneck. Defaults to 64.

    Returns
    -------
    PreTrainedModel
        The modified model instance with adapters injected and backbone frozen.

    Raises
    ------
    AttributeError
        If the model structure is not recognized (i.e., cannot find the encoder layers).
    """
    print(f"\n{'='*60}")
    print(f"[System] Starting Adapter Injection Procedure")
    print(f"{'='*60}")

    # 1. Freeze the entire model backbone
    # This ensures we don't destroy the pre-trained knowledge during fine-tuning.
    print("[Config] Freezing original backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Locate the Encoder Module
    # We inspect the model structure to find where the Transformer layers live.
    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        # Standard ViTMAE structure (Hugging Face)
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
        # Create the adapter instance
        adapter = IBA_Adapter(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        
        # CRITICAL: Move adapter to the correct device/dtype.
        # This handles cases where the model is already on GPU or in FP16/BF16.
        # We take the first parameter of the layer as a reference.
        ref_param = next(layer.parameters())
        adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        
        # Wrap the original layer with our adapter-enabled wrapper
        wrapped_layer = ViTBlockWithAdapter(original_block=layer, adapter=adapter)
        
        # Perform the replacement in the ModuleList
        encoder.layer[i] = wrapped_layer
        
        # Progress logging
        if (i + 1) % 4 == 0 or (i + 1) == num_layers:
            print(f"  -> Processed layer {i + 1}/{num_layers}")

    print(f"[System] Injection Complete. Decoder layers ignored (if present).")
    
    # 4. Verification
    # Print a summary of trainable vs frozen parameters to confirm success.
    count_trainable_params(model)
    
    return model


def count_trainable_params(model: nn.Module) -> None:
    """
    Audit Utility: Prints the distribution of Frozen vs Trainable parameters.
    
    Useful for verifying that:
    1.  The backbone is indeed frozen (0 gradients).
    2.  The adapters are trainable (requires_grad=True).
    
    Parameters
    ----------
    model : nn.Module
        The model to inspect.
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
# Main Execution Block (Integration Test)
# =============================================================================
if __name__ == "__main__":
    """
    Test Script to verify the Adapter Injection pipeline.
    
    Steps:
    1.  Load a real ViTMAE model from Hugging Face.
    2.  Inject Adapters.
    3.  Run a dummy forward pass to check for shape mismatches or runtime errors.
    """
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
        print(f"[Success] Standard Forward pass complete. Loss: {loss_val}")

    except Exception as e:
        print(f"[Error] An error occurred during execution: {e}")