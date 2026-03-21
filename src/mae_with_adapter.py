"""Inject residual adapters into the frozen ViT-MAE backbone.

The repository uses the same parameter-efficient recipe everywhere:
1. Load the pretrained MAE model exactly once.
2. Freeze the original backbone weights.
3. Attach lightweight residual adapters to the upper encoder blocks.
4. Train and exchange only those adapter weights during downstream runs.

This keeps the starting representation stable while making communication and
optimization much cheaper than full-model training.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class IBA_Adapter(nn.Module):
    """Apply a residual bottleneck update to one transformer hidden state.

    Each forward pass follows the same four steps:
    1. Compress the token representation into a smaller bottleneck space.
    2. Apply the adapter non-linearity.
    3. Project back to the backbone dimension.
    4. Add the adapter output back to the original hidden state.

    The up-projection starts at zero, so the wrapped backbone behaves like the
    untouched pretrained model at step zero.
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
        self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=True)
        self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the adapter so training starts from the pretrained backbone."""
        nn.init.kaiming_normal_(
            self.down_project.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.down_project.bias is not None:
            nn.init.zeros_(self.down_project.bias)

        nn.init.zeros_(self.up_project.weight)
        if self.up_project.bias is not None:
            nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the bottleneck path and add it back as a residual update."""
        residual = hidden_states
        adapted = self.down_project(hidden_states)
        adapted = self.activation(adapted)
        adapted = self.up_project(adapted)
        adapted = self.dropout(adapted)
        return residual + adapted


class ViTBlockWithAdapter(nn.Module):
    """Wrap one transformer block and append an adapter after the frozen block.

    The wrapper exists so the rest of the model still sees the same interface:
    1. Run the original transformer block untouched.
    2. Adapt only its hidden-state output.
    3. Return the same outer structure the caller expects.
    """

    def __init__(self, original_block: nn.Module, adapter: IBA_Adapter) -> None:
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Any], Any]:
        """Forward through the original block first, then through the adapter.

        ``*args`` and ``**kwargs`` are passed through unchanged so attention
        masks, hidden-state flags, and any future Hugging Face options keep
        working exactly as they did before adapter injection.
        """
        outputs = self.original_block(hidden_states, *args, **kwargs)

        if isinstance(outputs, tuple):
            adapted_hidden = self.adapter(outputs[0])
            return (adapted_hidden,) + outputs[1:]

        if hasattr(outputs, "last_hidden_state"):
            outputs.last_hidden_state = self.adapter(outputs.last_hidden_state)
            return outputs

        if hasattr(outputs, "hidden_states"):
            outputs.hidden_states = self.adapter(outputs.hidden_states)
            return outputs

        return self.adapter(outputs)


def inject_adapters(
    model: PreTrainedModel,
    bottleneck_dim: int = 64,
) -> PreTrainedModel:
    """Freeze the backbone and inject adapters into the upper half of the encoder.

    The helper edits the given model in place:
    1. Freeze every pretrained MAE parameter.
    2. Find the ViT encoder stack.
    3. Wrap the upper half of the blocks with residual adapters.
    4. Leave only the new adapter weights trainable.
    """
    logger.info("Injecting adapters with bottleneck_dim=%s", bottleneck_dim)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "vit") and hasattr(model.vit, "encoder"):
        encoder = model.vit.encoder
        config = model.config
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        encoder = model.encoder
        config = model.config
    else:
        raise AttributeError(
            "Could not find a ViT-style encoder stack on the provided model."
        )

    input_dim = config.hidden_size
    num_layers = len(encoder.layer)
    inject_start_layer = num_layers // 2

    for layer_index in range(inject_start_layer, num_layers):
        original_layer = encoder.layer[layer_index]
        adapter = IBA_Adapter(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
        )

        reference_parameter = next(original_layer.parameters())
        adapter.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )

        encoder.layer[layer_index] = ViTBlockWithAdapter(
            original_block=original_layer,
            adapter=adapter,
        )

    _log_param_stats(model)
    return model


def _log_param_stats(model: nn.Module) -> None:
    """Log how much of the full MAE model remains trainable after injection."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters
    trainable_ratio = (
        100.0 * trainable_parameters / total_parameters if total_parameters else 0.0
    )

    logger.info(
        "Adapter injection complete | total=%s | frozen=%s | trainable=%s (%.2f%%)",
        f"{total_parameters:,}",
        f"{frozen_parameters:,}",
        f"{trainable_parameters:,}",
        trainable_ratio,
    )
