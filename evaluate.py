"""Post-training checkpoint evaluation for the federated multi-dataset run.

This script stays outside the training loop on purpose:
1. Load a saved adapter checkpoint.
2. Rebuild the exact runtime config stored with that checkpoint.
3. Recover the dataset schedule from checkpoint metadata when available.
4. Run linear-probe evaluation on the saved model.
5. Run the same evaluation on the untouched Hugging Face base model.
6. Print and optionally save a side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import ViTMAEForPreTraining

from main import (
    DATASET_DISPLAY_NAMES,
    MULTI_DATASET_CONFIG,
    build_base_model,
    build_dataset_order_by_stage,
    evaluate_datasets,
    resolve_runtime_config,
    set_random_seed,
)

logger = logging.getLogger("PODFCSSV_Evaluate")

TORCH_DTYPE_LOOKUP: Dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved adapter checkpoint against the base Hugging Face model.",
    )
    parser.add_argument(
        "saved_model_path",
        type=Path,
        help="Path to the saved fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DISPLAY_NAMES.keys()),
        default=None,
        help="Datasets to evaluate. Defaults to the current main.py sequence.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the comparison results as JSON.",
    )
    return parser.parse_args()


def default_dataset_order(config: Dict[str, Any]) -> List[str]:
    """Return the stage-ordered dataset list from the checkpoint or current code."""
    dataset_sequence = config.get("client_dataset_sequence")
    if isinstance(dataset_sequence, dict) and dataset_sequence:
        return build_dataset_order_by_stage(dataset_sequence)
    return build_dataset_order_by_stage()


def deserialize_config(serialized_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert serialized checkpoint config values back into runtime types."""
    config = dict(serialized_config)

    dtype_value = config.get("dtype")
    if isinstance(dtype_value, str) and dtype_value in TORCH_DTYPE_LOOKUP:
        config["dtype"] = TORCH_DTYPE_LOOKUP[dtype_value]

    return config


def build_runtime_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Build the evaluation config from defaults plus checkpoint metadata."""
    config = dict(MULTI_DATASET_CONFIG)

    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint.get("config", {})
        config.update(deserialize_config(checkpoint_config))

    resolve_runtime_config(config)
    return config


def load_finetuned_model(checkpoint_path: Path, config: Dict[str, Any]) -> torch.nn.Module:
    """Load the fine-tuned adapter model from a saved checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_base_model(config)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise KeyError("The checkpoint does not contain a 'model_state_dict' entry.")

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if unexpected_keys:
        logger.warning("Unexpected keys while loading fine-tuned checkpoint: %s", unexpected_keys)
    if missing_keys:
        logger.info(
            "Checkpoint load completed with %s missing keys. This is expected because the checkpoint stores only trainable adapter weights.",
            len(missing_keys),
        )

    model.eval()
    return model


def load_huggingface_base_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Load the original Hugging Face ViT-MAE model without adapters."""
    model = ViTMAEForPreTraining.from_pretrained(config["pretrained_model_name"])
    model = model.to(device=config["device"], dtype=config["dtype"])
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    model_name: str,
    dataset_names: List[str],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate one model on a list of datasets with per-dataset progress logs."""
    results: Dict[str, float] = {}
    for dataset_name in dataset_names:
        display_name = DATASET_DISPLAY_NAMES[dataset_name]
        logger.info("Evaluating %s on %s...", model_name, display_name)
        accuracy = evaluate_datasets(
            model=model,
            dataset_names=[dataset_name],
            config=config,
        )[dataset_name]
        results[dataset_name] = accuracy
        logger.info("%s | %s accuracy=%.4f", model_name, display_name, accuracy)
    return results


def print_comparison_table(
    dataset_names: List[str],
    base_results: Dict[str, float],
    finetuned_results: Dict[str, float],
) -> None:
    """Print a compact dataset-by-dataset comparison table."""
    header = (
        f"{'Dataset':<20} {'HF Base':>10} {'Fine-tuned':>12} {'Delta':>10}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for dataset_name in dataset_names:
        base_accuracy = base_results.get(dataset_name, 0.0)
        finetuned_accuracy = finetuned_results.get(dataset_name, 0.0)
        delta = finetuned_accuracy - base_accuracy
        print(
            f"{DATASET_DISPLAY_NAMES[dataset_name]:<20} "
            f"{base_accuracy:>10.4f} "
            f"{finetuned_accuracy:>12.4f} "
            f"{delta:>10.4f}"
        )
    print(separator)


def save_results(
    output_path: Path,
    checkpoint_path: Path,
    dataset_names: List[str],
    base_results: Dict[str, float],
    finetuned_results: Dict[str, float],
) -> None:
    """Write the comparison results to disk as JSON."""
    comparison = {
        dataset_name: {
            "hf_base_accuracy": base_results.get(dataset_name, 0.0),
            "finetuned_accuracy": finetuned_results.get(dataset_name, 0.0),
            "delta": finetuned_results.get(dataset_name, 0.0)
            - base_results.get(dataset_name, 0.0),
        }
        for dataset_name in dataset_names
    }

    payload = {
        "checkpoint": str(checkpoint_path),
        "datasets": dataset_names,
        "results": comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved comparison JSON to %s", output_path)


def main() -> None:
    """Evaluate the saved fine-tuned checkpoint against the base model."""
    args = parse_args()
    config = build_runtime_config(args.saved_model_path)
    set_random_seed(config.get("seed"))
    dataset_names = args.datasets or default_dataset_order(config)

    logger.info(
        "Starting evaluation | checkpoint=%s | device=%s | datasets=%s",
        args.saved_model_path,
        config["device"],
        ", ".join(DATASET_DISPLAY_NAMES[name] for name in dataset_names),
    )

    finetuned_model = load_finetuned_model(args.saved_model_path, config)
    finetuned_results = evaluate_model(
        model=finetuned_model,
        model_name="Fine-tuned",
        dataset_names=dataset_names,
        config=config,
    )
    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_model = load_huggingface_base_model(config)
    base_results = evaluate_model(
        model=base_model,
        model_name="HF Base",
        dataset_names=dataset_names,
        config=config,
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print_comparison_table(
        dataset_names=dataset_names,
        base_results=base_results,
        finetuned_results=finetuned_results,
    )

    if args.output_json is not None:
        save_results(
            output_path=args.output_json,
            checkpoint_path=args.saved_model_path,
            dataset_names=dataset_names,
            base_results=base_results,
            finetuned_results=finetuned_results,
        )


if __name__ == "__main__":
    main()
