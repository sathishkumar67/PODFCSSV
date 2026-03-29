"""Download-check every dataset used by the current pipeline.

This script is intentionally separate from ``main.py`` so it can validate the
dataset stack without importing the full training pipeline. The check follows
these steps:
1. Parse the current benchmark and stress schedules directly from ``main.py``.
2. Build the same stage-aware dataset order used by the training code.
3. Download the required splits for one dataset at a time.
4. Materialize each split by calling ``len(...)`` so metadata extraction also
   gets exercised.
5. Delete that dataset directory before moving on to the next dataset.
6. Print a final success or failure summary.
"""

from __future__ import annotations

import ast
import gc
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from torchvision import datasets

DATA_ROOT = Path("./data")
PIPELINE_FILE = Path(__file__).with_name("main.py")
SKIPPED_DATASETS = {
    "eurosat",
    "gtsrb",
    "cifar10",
    "svhn",
    "food101",
    "country211",
}


def load_pipeline_sequences() -> tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Read the benchmark and stress schedules directly from ``main.py``.

    The parser uses only the Python standard library so the download smoke test
    stays independent from the heavier training imports.
    """
    module = ast.parse(PIPELINE_FILE.read_text(encoding="utf-8"))
    values: Dict[str, Dict[int, List[str]]] = {}

    for node in module.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in {
            "BENCHMARK_CLIENT_DATASET_SEQUENCE",
            "FEDERATED_RETENTION_NOISE_SEQUENCE",
        }:
            continue
        values[node.target.id] = ast.literal_eval(node.value)

    return (
        values["BENCHMARK_CLIENT_DATASET_SEQUENCE"],
        values["FEDERATED_RETENTION_NOISE_SEQUENCE"],
    )


def build_dataset_order(
    benchmark_sequence: Dict[int, Sequence[str]],
    stress_sequence: Dict[int, Sequence[str]],
) -> List[str]:
    """Build the current stage-aware order while skipping verified datasets."""
    benchmark_stage_count = len(next(iter(benchmark_sequence.values())))
    stress_stage_count = len(next(iter(stress_sequence.values())))
    ordered_names: List[str] = []

    for stage_index in range(benchmark_stage_count):
        for client_index in sorted(benchmark_sequence):
            dataset_name = benchmark_sequence[client_index][stage_index]
            if dataset_name not in SKIPPED_DATASETS:
                ordered_names.append(dataset_name)
        if stage_index < stress_stage_count:
            for client_index in sorted(stress_sequence):
                dataset_name = stress_sequence[client_index][stage_index]
                if dataset_name not in SKIPPED_DATASETS:
                    ordered_names.append(dataset_name)

    return ordered_names


def dataset_root(dataset_name: str) -> Path:
    """Return the on-disk dataset folder used by the repo loaders."""
    return DATA_ROOT / "multidataset" / dataset_name


def build_download_jobs(dataset_name: str) -> List[Tuple[str, Callable[[], object]]]:
    """Return the split constructors needed to verify one dataset.

    Each job matches a split path referenced by the active pipeline so a
    successful run means the training code will hit the same download and
    extraction paths later.
    """
    root = str(dataset_root(dataset_name))

    if dataset_name == "eurosat":
        return [("all", lambda: datasets.EuroSAT(root=root, download=True))]
    if dataset_name == "food101":
        return [
            ("train", lambda: datasets.Food101(root=root, split="train", download=True)),
            ("test", lambda: datasets.Food101(root=root, split="test", download=True)),
        ]
    if dataset_name == "oxfordiiitpet":
        return [
            ("trainval", lambda: datasets.OxfordIIITPet(root=root, split="trainval", download=True)),
            ("test", lambda: datasets.OxfordIIITPet(root=root, split="test", download=True)),
        ]
    if dataset_name == "gtsrb":
        return [
            ("train", lambda: datasets.GTSRB(root=root, split="train", download=True)),
            ("test", lambda: datasets.GTSRB(root=root, split="test", download=True)),
        ]
    if dataset_name == "country211":
        return [
            ("train", lambda: datasets.Country211(root=root, split="train", download=True)),
            ("test", lambda: datasets.Country211(root=root, split="test", download=True)),
        ]
    if dataset_name == "fgvcaircraft":
        return [
            (
                "trainval",
                lambda: datasets.FGVCAircraft(
                    root=root,
                    split="trainval",
                    annotation_level="variant",
                    download=True,
                ),
            ),
            (
                "test",
                lambda: datasets.FGVCAircraft(
                    root=root,
                    split="test",
                    annotation_level="variant",
                    download=True,
                ),
            ),
        ]
    if dataset_name == "cifar10":
        return [
            ("train", lambda: datasets.CIFAR10(root=root, train=True, download=True)),
            ("test", lambda: datasets.CIFAR10(root=root, train=False, download=True)),
        ]
    if dataset_name == "svhn":
        return [
            ("train", lambda: datasets.SVHN(root=root, split="train", download=True)),
            ("test", lambda: datasets.SVHN(root=root, split="test", download=True)),
        ]
    if dataset_name == "stl10":
        return [
            ("train", lambda: datasets.STL10(root=root, split="train", download=True)),
            ("test", lambda: datasets.STL10(root=root, split="test", download=True)),
        ]
    if dataset_name == "cifar100":
        return [
            ("train", lambda: datasets.CIFAR100(root=root, train=True, download=True)),
            ("test", lambda: datasets.CIFAR100(root=root, train=False, download=True)),
        ]
    if dataset_name == "flowers102":
        return [
            ("train", lambda: datasets.Flowers102(root=root, split="train", download=True)),
            ("val", lambda: datasets.Flowers102(root=root, split="val", download=True)),
            ("test", lambda: datasets.Flowers102(root=root, split="test", download=True)),
        ]
    if dataset_name == "dtd":
        return [
            ("train", lambda: datasets.DTD(root=root, split="train", download=True)),
            ("val", lambda: datasets.DTD(root=root, split="val", download=True)),
            ("test", lambda: datasets.DTD(root=root, split="test", download=True)),
        ]

    raise ValueError(f"Unsupported dataset in smoke test: {dataset_name}")


def delete_dataset_dir(target_dir: Path, retries: int = 5, delay_seconds: float = 1.0) -> None:
    """Delete one dataset directory with a few retries for Windows file handles."""
    if not target_dir.exists():
        return

    last_error: Exception | None = None
    for _ in range(retries):
        try:
            shutil.rmtree(target_dir)
            return
        except Exception as exc:
            last_error = exc
            gc.collect()
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error


def verify_dataset(dataset_name: str) -> Dict[str, int]:
    """Download one dataset, touch every required split, and return split sizes."""
    split_sizes: Dict[str, int] = {}
    for split_name, factory in build_download_jobs(dataset_name):
        dataset = factory()
        split_sizes[split_name] = len(dataset)
        del dataset
        gc.collect()
    return split_sizes


def main() -> None:
    """Run the full download-and-delete smoke test for all active datasets."""
    benchmark_sequence, stress_sequence = load_pipeline_sequences()
    ordered_dataset_names = build_dataset_order(benchmark_sequence, stress_sequence)
    failures: List[Tuple[str, str]] = []

    print("Dataset download smoke test starting")
    print(f"Order: {ordered_dataset_names}")

    for dataset_name in ordered_dataset_names:
        target_dir = dataset_root(dataset_name)
        print(f"\n=== Checking {dataset_name} ===")
        try:
            split_sizes = verify_dataset(dataset_name)
            print(f"Downloaded successfully: {split_sizes}")
        except Exception as exc:
            failures.append((dataset_name, str(exc)))
            print(f"FAILED: {exc}")
        finally:
            try:
                delete_dataset_dir(target_dir)
                print(f"Deleted: {target_dir}")
            except Exception as cleanup_exc:
                failures.append((dataset_name, f"cleanup failed: {cleanup_exc}"))
                print(f"CLEANUP FAILED: {cleanup_exc}")

    if failures:
        print("\nDownload smoke test finished with failures:")
        for dataset_name, message in failures:
            print(f"- {dataset_name}: {message}")
        raise SystemExit(1)

    print("\nDownload smoke test finished successfully for every dataset.")


if __name__ == "__main__":
    main()
