"""Single-model continual baseline for the multi-dataset sequence.

This entrypoint keeps the same adapter-injected ViT-MAE model used by
``main.py`` but removes federation and GPAD entirely:
1. Build one frozen ViT-MAE backbone with the same injected adapters.
2. Train that single model on one dataset at a time using reconstruction loss.
3. Carry the same model and optimizer state across dataset transitions.
4. Save checkpoints, JSON histories, and training-only summary plots.

Evaluation is intentionally left out of this file so it can stay focused on
the continual training path alone.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from main import (
    DATASET_DISPLAY_NAMES,
    MULTI_DATASET_CONFIG,
    build_base_model,
    build_dataset_order_by_stage,
    create_standard_dataloader,
    load_named_dataset,
    prepare_output_dirs,
    save_checkpoint,
    save_history,
    set_random_seed,
    resolve_runtime_config,
)

logger = logging.getLogger("PODFCSSV_Base")

BASE_CONFIG: Dict[str, Any] = {
    **MULTI_DATASET_CONFIG,
    "num_clients": 1,
    "save_dir": "base_outputs",
    "train_samples_per_dataset": None,
}


def initialize_base_history() -> Dict[str, Any]:
    """Create the baseline history containers."""
    return {
        "round_ids": [],
        "dataset_names": [],
        "dataset_rounds": [],
        "round_times": [],
        "avg_total_loss": [],
        "avg_mae_loss": [],
        "stage_summaries": [],
    }


def train_reconstruction_round(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    local_epochs: int,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Train the single model for one round with reconstruction loss only."""
    model.train()
    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    for _ in range(local_epochs):
        for batch in dataloader:
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            inputs = inputs.to(device=device, dtype=dtype)
            outputs = model(inputs)
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("The MAE model did not return a reconstruction loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            total_batches += 1
            total_samples += int(inputs.size(0))

    average_loss = total_loss / max(total_batches, 1)
    return {
        "loss": average_loss,
        "mae_loss": average_loss,
        "num_batches": total_batches,
        "num_samples": total_samples,
    }

def plot_base_training_history(history: Dict[str, Any], plots_dir: Path) -> Path:
    """Plot baseline reconstruction loss and round time."""
    figure_path = plots_dir / "base_training_summary.png"
    rounds = history.get("round_ids", [])
    if not rounds:
        return figure_path

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(rounds, history["avg_total_loss"], marker="o")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")

    axes[1].plot(rounds, history["round_times"], marker="o")
    axes[1].set_title("Round Time")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Seconds")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    """Run the single-model continual baseline."""
    config = dict(BASE_CONFIG)
    resolve_runtime_config(config)
    config["dataset_order_by_stage"] = build_dataset_order_by_stage()
    config["client_dataset_sequence"] = {"0": list(config["dataset_order_by_stage"])}
    config["num_stages"] = len(config["dataset_order_by_stage"])
    config["total_sequential_rounds"] = config["num_stages"] * config["rounds_per_dataset"]

    set_random_seed(config["seed"])
    output_dirs = prepare_output_dirs(config["save_dir"])

    model = build_base_model(config)
    optimizer = optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["client_lr"],
        weight_decay=config["client_weight_decay"],
    )
    history = initialize_base_history()
    global_round_idx = 0

    logger.info(
        "Starting single-model baseline | datasets=%s | rounds_per_dataset=%s | device=%s | output=%s",
        len(config["dataset_order_by_stage"]),
        config["rounds_per_dataset"],
        config["device"],
        output_dirs["root"],
    )

    for stage_index, dataset_name in enumerate(config["dataset_order_by_stage"], start=1):
        logger.info(
            "Starting baseline stage %s/%s | dataset=%s",
            stage_index,
            config["num_stages"],
            DATASET_DISPLAY_NAMES[dataset_name],
        )
        stage_losses = []

        dataset = load_named_dataset(
            dataset_name=dataset_name,
            data_root=config["data_root"],
            image_size=config["image_size"],
            train=True,
            seed=config["seed"],
            max_samples=config["train_samples_per_dataset"],
            min_samples=config["min_train_samples_per_dataset"],
        )
        dataloader = create_standard_dataloader(
            dataset=dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            shuffle=config["dataloader_shuffle"],
        )

        for dataset_round in range(1, config["rounds_per_dataset"] + 1):
            global_round_idx += 1
            round_start = time.time()
            round_result = train_reconstruction_round(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                local_epochs=config["local_epochs"],
                device=config["device"],
                dtype=config["dtype"],
            )
            round_time = time.time() - round_start

            history["round_ids"].append(global_round_idx)
            history["dataset_names"].append(dataset_name)
            history["dataset_rounds"].append(dataset_round)
            history["round_times"].append(round_time)
            history["avg_total_loss"].append(round_result["loss"])
            history["avg_mae_loss"].append(round_result["mae_loss"])
            stage_losses.append(round_result["loss"])

            logger.info(
                "Baseline round %s/%s complete | dataset=%s | round=%s/%s | loss=%.6f | time=%.2fs",
                global_round_idx,
                config["total_sequential_rounds"],
                DATASET_DISPLAY_NAMES[dataset_name],
                dataset_round,
                config["rounds_per_dataset"],
                round_result["loss"],
                round_time,
            )

            save_checkpoint(
                checkpoint_dir=output_dirs["checkpoints"],
                round_idx=global_round_idx,
                base_model=model,
                config=config,
                training_history=history,
                is_final=False,
                include_training_history=False,
            )
            save_history(history, output_dirs["metrics"], filename="base_training_history.json")
            plot_base_training_history(history, output_dirs["plots"])

        history["stage_summaries"].append(
            {
                "stage": stage_index,
                "dataset_name": dataset_name,
                "display_name": DATASET_DISPLAY_NAMES[dataset_name],
                "num_train_samples": len(dataset),
                "average_round_loss": float(sum(stage_losses) / len(stage_losses)),
                "last_round_loss": float(stage_losses[-1]),
            }
        )
        logger.info(
            "Baseline stage %s complete | dataset=%s | avg_round_loss=%.6f | train_samples=%s",
            stage_index,
            DATASET_DISPLAY_NAMES[dataset_name],
            history["stage_summaries"][-1]["average_round_loss"],
            len(dataset),
        )

        save_history(history, output_dirs["metrics"], filename="base_training_history.json")
        plot_base_training_history(history, output_dirs["plots"])

        del dataloader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_checkpoint(
        checkpoint_dir=output_dirs["checkpoints"],
        round_idx=global_round_idx,
        base_model=model,
        config=config,
        training_history=history,
        is_final=True,
        include_training_history=False,
    )
    logger.info(
        "Baseline training complete | rounds=%s | final_checkpoint=%s",
        global_round_idx,
        output_dirs["checkpoints"] / "final_model.pt",
    )


if __name__ == "__main__":
    main()
