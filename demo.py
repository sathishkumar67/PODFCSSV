"""Run single-image reconstruction with the saved baseline and federated adapters.

The script rebuilds the same frozen ViT-MAE-Base backbone used during training,
loads the adapter-only checkpoints from the ``runs`` folder, and evaluates one
input image with both final models. For each mode it prints the MAE
reconstruction loss, prints a full-image MSE for easier comparison, and saves
the reconstructed image to ``demo_outputs``.

Usage:
    python demo.py path/to/image.jpg
    python demo.py path/to/image.jpg --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import ViTMAEForPreTraining

from src.mae_with_adapter import inject_adapters


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "demo_outputs"

DEFAULT_MODEL_NAME = "facebook/vit-mae-base"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ADAPTER_BOTTLENECK = 256


CHECKPOINT_CANDIDATES = {
    "baseline": [
        RUNS_ROOT / "baseline" / "baseline_checkpoint.pt",
        RUNS_ROOT / "baseline" / "checkpoints" / "final_model.pt",
        PROJECT_ROOT / "baseline_outputs" / "checkpoints" / "final_model.pt",
    ],
    "federated": [
        RUNS_ROOT / "federated" / "federated_checkpoint.pt",
        RUNS_ROOT / "federated" / "checkpoints" / "final_model.pt",
        PROJECT_ROOT / "multidataset_outputs_2client" / "checkpoints" / "final_model.pt",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct one image with baseline and federated checkpoints."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="Root folder containing baseline/federated run outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Folder where reconstructed images will be saved.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the MAE random mask during inference.",
    )
    return parser.parse_args()


def first_existing_path(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    candidates = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"No checkpoint found. Checked:\n{candidates}")


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary at {path}, got {type(checkpoint)!r}.")
    return checkpoint


def checkpoint_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    config = checkpoint.get("config", {})
    return config if isinstance(config, dict) else {}


def checkpoint_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "adapter_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError(
        "Checkpoint does not contain model_state_dict, adapter_state_dict, or state_dict."
    )


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> ViTMAEForPreTraining:
    config = checkpoint_config(checkpoint)
    model_name = str(config.get("pretrained_model_name", DEFAULT_MODEL_NAME))
    bottleneck_dim = int(config.get("adapter_bottleneck_dim", DEFAULT_ADAPTER_BOTTLENECK))

    model = ViTMAEForPreTraining.from_pretrained(model_name)
    model = inject_adapters(model, bottleneck_dim=bottleneck_dim)
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def image_size_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    config = checkpoint_config(checkpoint)
    return int(config.get("image_size", DEFAULT_IMAGE_SIZE))


def load_image_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor


def unpatchify(logits: torch.Tensor, model: ViTMAEForPreTraining) -> torch.Tensor:
    patch_size = int(model.config.patch_size)
    channels = int(model.config.num_channels)
    num_patches = logits.shape[1]
    grid_size = int(num_patches**0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot unpatchify {num_patches} patches into a square image.")

    images = logits.reshape(
        logits.shape[0],
        grid_size,
        grid_size,
        patch_size,
        patch_size,
        channels,
    )
    images = images.permute(0, 5, 1, 3, 2, 4)
    return images.reshape(
        logits.shape[0],
        channels,
        grid_size * patch_size,
        grid_size * patch_size,
    )


def patchify(images: torch.Tensor, model: ViTMAEForPreTraining) -> torch.Tensor:
    patch_size = int(model.config.patch_size)
    channels = int(model.config.num_channels)
    if images.shape[2] != images.shape[3] or images.shape[2] % patch_size != 0:
        raise ValueError("Expected square images with side length divisible by patch size.")

    grid_size = images.shape[2] // patch_size
    patches = images.reshape(
        images.shape[0],
        channels,
        grid_size,
        patch_size,
        grid_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(images.shape[0], grid_size * grid_size, patch_size**2 * channels)


def logits_to_image(
    logits: torch.Tensor,
    original_image: torch.Tensor,
    model: ViTMAEForPreTraining,
) -> torch.Tensor:
    if bool(getattr(model.config, "norm_pix_loss", False)):
        target_patches = patchify(original_image, model)
        patch_mean = target_patches.mean(dim=-1, keepdim=True)
        patch_std = target_patches.var(dim=-1, keepdim=True).add(1e-6).sqrt()
        logits = logits * patch_std + patch_mean
    return unpatchify(logits, model)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    return transforms.ToPILImage()(image)


@torch.inference_mode()
def run_reconstruction(
    mode: str,
    checkpoint_path: Path,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    image_size = image_size_from_checkpoint(checkpoint)
    image = load_image_tensor(image_path, image_size, device)
    model = build_model_from_checkpoint(checkpoint, device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    outputs = model(pixel_values=image)
    loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)
    if loss is None or logits is None:
        raise RuntimeError(f"{mode} model did not return both loss and logits.")

    reconstruction = logits_to_image(logits, image, model)
    full_image_mse = F.mse_loss(reconstruction.clamp(0.0, 1.0), image).item()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_{mode}_reconstruction.png"
    tensor_to_pil(reconstruction).save(output_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "mode": mode,
        "checkpoint": checkpoint_path,
        "loss": float(loss.detach().cpu().item()),
        "full_image_mse": float(full_image_mse),
        "reconstruction_path": output_path,
        "input_tensor": image.detach().cpu(),
        "reconstruction_tensor": reconstruction.detach().cpu(),
    }


def save_comparison_image(results: list[Dict[str, Any]], image_path: Path, output_dir: Path) -> Path:
    panels: list[tuple[str, Image.Image]] = [
        ("Input", tensor_to_pil(results[0]["input_tensor"])),
    ]
    for result in results:
        panels.append((result["mode"].title(), tensor_to_pil(result["reconstruction_tensor"])))

    width, height = panels[0][1].size
    label_height = 30
    canvas = Image.new("RGB", (width * len(panels), height + label_height), color="white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * width
        canvas.paste(image.convert("RGB"), (x, label_height))
        draw.text((x + 8, 8), label, fill=(0, 0, 0))

    comparison_path = output_dir / f"{image_path.stem}_baseline_vs_federated.png"
    canvas.save(comparison_path)
    return comparison_path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_paths = {
        mode: first_existing_path(
            [
                args.runs_root / mode / f"{mode}_checkpoint.pt",
                args.runs_root / mode / "checkpoints" / "final_model.pt",
                *CHECKPOINT_CANDIDATES[mode],
            ]
        )
        for mode in ("baseline", "federated")
    }

    results = []
    for mode in ("baseline", "federated"):
        result = run_reconstruction(
            mode=mode,
            checkpoint_path=checkpoint_paths[mode],
            image_path=args.image_path,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        print(f"{mode.title()} checkpoint: {result['checkpoint']}")
        print(f"{mode.title()} MAE reconstruction loss: {result['loss']:.8f}")
        print(f"{mode.title()} full-image MSE: {result['full_image_mse']:.8f}")
        print(f"{mode.title()} reconstruction: {result['reconstruction_path']}")
        print()

    comparison_path = save_comparison_image(results, args.image_path, args.output_dir)
    print(f"Comparison image: {comparison_path}")


if __name__ == "__main__":
    main()
