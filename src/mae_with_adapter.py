"""Adapter injection for a frozen ViT-MAE backbone.

This file implements the parameter-efficient fine-tuning strategy used
throughout the repository. The design is intentionally simple:
1. Freeze the pre-trained MAE backbone.
2. Insert lightweight residual adapters into the upper half of the encoder.
3. Train only the adapter parameters during federated rounds.

Keeping the adapters residual and zero-initialized makes the wrapped model
start from the same function as the pre-trained backbone, then gradually learn
task-specific corrections.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class IBA_Adapter(nn.Module):
    """Residual information-bottleneck adapter.

    The adapter applies a low-rank transformation to each token embedding:
    1. Project from the backbone dimension ``D`` to a smaller bottleneck
       dimension ``d``.
    2. Apply a non-linearity.
    3. Project back from ``d`` to ``D``.
    4. Add the result back to the original hidden state.

    The up-projection is initialized to zero so the adapter starts as an
    identity mapping.
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
        """Initialize the adapter so it behaves like an identity at step zero."""
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
        """Apply the residual bottleneck transformation to the hidden states."""
        residual = hidden_states
        adapted = self.down_project(hidden_states)
        adapted = self.activation(adapted)
        adapted = self.up_project(adapted)
        adapted = self.dropout(adapted)
        return residual + adapted


class ViTBlockWithAdapter(nn.Module):
    """Wrap a transformer block and append an adapter to its output.

    The wrapper preserves the original return type. That is important because
    Hugging Face transformer blocks can return tuples or structured model output
    objects depending on the caller's flags.
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
        """Run the frozen block first, then apply the adapter to its output.

        The wrapper forwards ``*args`` and ``**kwargs`` unchanged so features
        such as head masks and attention-output flags continue to work exactly
        as they do in the original transformer block.
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

    The function edits the provided model in place and returns the same model
    reference for convenience.
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
    """Log how many parameters remain trainable after adapter injection."""
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
