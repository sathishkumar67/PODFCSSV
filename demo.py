"""Run a federated checkpoint reconstruction check for one image.

This script keeps the demo intentionally small and terminal-only. It loads the
federated adapter checkpoint from the `runs` folder, prepares a single image in
the same image size used by the training pipeline, runs the MAE reconstruction
forward pass, and prints only the federated reconstruction metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import ViTMAEForPreTraining

from src.mae_with_adapter import inject_adapters


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "federated" / "federated_checkpoint.pt"
DEFAULT_MODEL_NAME = "facebook/vit-mae-base"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_ADAPTER_BOTTLENECK = 256


def parse_args() -> argparse.Namespace:
    """Collect the image path and optional runtime settings from the terminal."""

    parser = argparse.ArgumentParser(
        description="Print federated MAE reconstruction metrics for one image."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the federated checkpoint saved in the runs folder.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to make the MAE masking pattern reproducible.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load the checkpoint while remaining compatible with older PyTorch builds."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Federated checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary saved by the training pipeline.")

    return checkpoint


def get_checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return the serialized training config when it exists."""

    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        return config
    return {}


def get_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the adapter model state saved by the federated training run."""

    state = checkpoint.get("model_state_dict")
    if state is None:
        state = checkpoint.get("global_model_state_dict")
    if state is None:
        state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise KeyError(
            "Checkpoint does not contain model_state_dict, "
            "global_model_state_dict, or adapter_state_dict."
        )
    return state


def build_federated_model(checkpoint: dict[str, Any], device: torch.device) -> ViTMAEForPreTraining:
    """Create the MAE base model, inject adapters, and load federated weights."""

    config = get_checkpoint_config(checkpoint)
    model_name = str(config.get("pretrained_model_name", DEFAULT_MODEL_NAME))
    bottleneck_dim = int(config.get("adapter_bottleneck_dim", DEFAULT_ADAPTER_BOTTLENECK))

    model = ViTMAEForPreTraining.from_pretrained(model_name)
    inject_adapters(model, bottleneck_dim=bottleneck_dim)

    checkpoint_state = get_model_state(checkpoint)
    if not any("adapter" in key.lower() for key in checkpoint_state):
        raise RuntimeError("The federated checkpoint does not contain adapter weights.")

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")

    missing_adapter_keys = [key for key in missing_keys if "adapter" in key.lower()]
    if missing_adapter_keys:
        raise RuntimeError(f"Missing adapter keys after checkpoint load: {missing_adapter_keys}")

    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def load_image_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """Read one RGB image and convert it to a float32 tensor in the model format."""

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split an image tensor into flattened non-overlapping MAE patches."""

    batch_size, channels, height, width = images.shape
    if height != width:
        raise ValueError("MAE reconstruction expects square images.")
    if height % patch_size != 0:
        raise ValueError("Image size must be divisible by the model patch size.")

    patches_per_side = height // patch_size
    patches = images.reshape(
        batch_size,
        channels,
        patches_per_side,
        patch_size,
        patches_per_side,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    return patches.reshape(batch_size, patches_per_side * patches_per_side, patch_size**2 * channels)


def unpatchify(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Rebuild an image tensor from flattened MAE patch predictions."""

    batch_size, num_patches, patch_dim = patches.shape
    channels = patch_dim // (patch_size**2)
    patches_per_side = int(num_patches**0.5)
    if patches_per_side * patches_per_side != num_patches:
        raise ValueError("Number of patches must form a square grid.")

    images = patches.reshape(
        batch_size,
        patches_per_side,
        patches_per_side,
        patch_size,
        patch_size,
        channels,
    )
    images = images.permute(0, 5, 1, 3, 2, 4)
    return images.reshape(
        batch_size,
        channels,
        patches_per_side * patch_size,
        patches_per_side * patch_size,
    )


def logits_to_reconstruction(
    model: ViTMAEForPreTraining,
    pixel_values: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Convert MAE patch logits into a full reconstructed image tensor."""

    patch_size = int(model.config.patch_size)
    predicted_patches = logits.to(dtype=torch.float32)

    if bool(getattr(model.config, "norm_pix_loss", False)):
        target_patches = patchify(pixel_values, patch_size)
        patch_mean = target_patches.mean(dim=-1, keepdim=True)
        patch_var = target_patches.var(dim=-1, keepdim=True)
        predicted_patches = predicted_patches * (patch_var + 1e-6).sqrt() + patch_mean

    return unpatchify(predicted_patches, patch_size)


@torch.inference_mode()
def run_federated_reconstruction(
    image_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run the federated model once and return reconstruction metrics."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    checkpoint = load_checkpoint(checkpoint_path)
    config = get_checkpoint_config(checkpoint)
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))

    model = build_federated_model(checkpoint, device)
    pixel_values = load_image_tensor(image_path, image_size, device)
    outputs = model(pixel_values=pixel_values)
    reconstruction = logits_to_reconstruction(model, pixel_values, outputs.logits).clamp(0.0, 1.0)

    return {
        "checkpoint": str(checkpoint_path),
        "image": str(image_path),
        "device": str(device),
        "image_size": image_size,
        "mae_reconstruction_loss": float(outputs.loss.detach().cpu().item()),
        "full_image_mse": float(F.mse_loss(reconstruction, pixel_values).detach().cpu().item()),
    }


def main() -> None:
    """Print only the federated reconstruction result for the requested image."""

    args = parse_args()
    device = torch.device(args.device)
    result = run_federated_reconstruction(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        device=device,
        seed=args.seed,
    )

    print("Federated reconstruction result")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"image: {result['image']}")
    print(f"device: {result['device']}")
    print(f"image_size: {result['image_size']}")
    print(f"mae_reconstruction_loss: {result['mae_reconstruction_loss']:.8f}")
    print(f"full_image_mse: {result['full_image_mse']:.8f}")


if __name__ == "__main__":
    main()
