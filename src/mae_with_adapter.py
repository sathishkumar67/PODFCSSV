"""Attach residual adapters to the frozen ViT-MAE backbone.

The repository reuses the same parameter-efficient recipe everywhere:

1. load the pretrained MAE model,
2. freeze the original backbone,
3. insert lightweight residual adapters into the upper encoder blocks, and
4. leave only those adapters trainable during continual learning.

This keeps the pretrained representation stable while dramatically reducing the
number of parameters that must be optimized and communicated.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class IBA_Adapter(nn.Module):
    """Apply one bottleneck-style residual update to a transformer block output.

    Each call follows the same sequence:
    1. down-project the hidden state into a smaller bottleneck space,
    2. apply the adapter non-linearity,
    3. project back to the original hidden size, and
    4. add that adapter output back to the incoming hidden state.

    The up-projection is initialized at zero so training starts from the exact
    pretrained backbone behavior.
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
        """Initialize the adapter to preserve the pretrained backbone at step zero.

        The down-projection is given a normal Kaiming initialization, while the
        up-projection is set to zero so the adapter initially behaves like a
        no-op residual branch.
        """
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
        """Run the bottleneck path and return the residual adapter output."""
        residual = hidden_states
        adapted = self.down_project(hidden_states)
        adapted = self.activation(adapted)
        adapted = self.up_project(adapted)
        adapted = self.dropout(adapted)
        return residual + adapted


class ViTBlockWithAdapter(nn.Module):
    """Wrap one transformer block and insert an adapter after it.

    The wrapper keeps the external interface unchanged:
    1. run the original transformer block exactly as before,
    2. apply the adapter only to the hidden-state output, and
    3. return the same structure expected by the Hugging Face model stack.
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

        ``*args`` and ``**kwargs`` pass through untouched so attention masks,
        hidden-state flags, and future Hugging Face options continue to work in
        exactly the same way as the unwrapped block.
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
    """Freeze the backbone and inject adapters into the upper encoder layers.

    The helper edits the model in place:
    1. freeze every pretrained MAE parameter,
    2. locate the encoder stack inside the model,
    3. compute the midpoint of that stack,
    4. wrap every upper-half block with a residual adapter, and
    5. leave only the new adapter parameters trainable.

    This is the structural step that turns the pretrained MAE into the
    parameter-efficient model used by both run modes.
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
    """Log how much of the MAE model remains trainable after injection.

    The log is meant to make the parameter-efficiency story explicit for every
    run: total parameters, frozen parameters, trainable parameters, and the
    resulting trainable percentage.
    """
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
