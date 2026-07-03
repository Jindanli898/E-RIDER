#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export a Hugging Face ImageNet-1k subset to ImageFolder layout.

This script is designed for the training code in ``S4-ImageNet-VGG11.py``.
It reads the gated HF dataset ``ILSVRC/imagenet-1k`` using streaming mode,
exports:

1. train: a class-balanced subset with a fixed number of samples per class
2. val: the full validation split by default

The output layout is directly compatible with ``torchvision.datasets.ImageFolder``:

    output_root/
      train/
        0000/
        0001/
        ...
      val/
        0000/
        0001/
        ...

The zero-padded folder names ensure ImageFolder assigns class indices in the
same integer order as the HF ``label`` field.
"""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export HF ImageNet-1k subset to ImageFolder")
    parser.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k")
    parser.add_argument(
        "--output-root",
        type=str,
        default="/data/imagenet_prepared/imagenet1k_hf_train200_valfull",
        help="Output directory containing train/ and val/ subdirectories",
    )
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=-1,
        help="Use -1 to export the full validation split",
    )
    parser.add_argument(
        "--train-shuffle-buffer",
        type=int,
        default=10000,
        help="Streaming shuffle buffer for the train split. Set 0 to disable shuffle.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root before export if it already exists",
    )
    return parser.parse_args()


def prepare_output_root(output_root: Path, overwrite: bool):
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Pass --overwrite to remove it first."
            )
        shutil.rmtree(output_root)

    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)


def get_progress_bar():
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


def class_folder_name(label_idx: int, num_classes: int) -> str:
    width = max(4, len(str(num_classes - 1)))
    return f"{label_idx:0{width}d}"


def save_rgb_jpeg(image, out_path: Path, quality: int):
    image = image.convert("RGB")
    image.save(out_path, format="JPEG", quality=quality)


def get_class_names_from_features(ds):
    label_feature = ds.features["label"]
    if hasattr(label_feature, "names") and label_feature.names is not None:
        return list(label_feature.names)
    raise ValueError("Could not recover class names from dataset features['label']")


def write_label_mapping(output_root: Path, class_names):
    mapping = []
    num_classes = len(class_names)
    for label_idx, class_name in enumerate(class_names):
        mapping.append(
            {
                "label_idx": label_idx,
                "folder_name": class_folder_name(label_idx, num_classes),
                "class_name": class_name,
            }
        )

    (output_root / "label_mapping.json").write_text(json.dumps(mapping, indent=2))

    lines = ["label_idx\tfolder_name\tclass_name"]
    for row in mapping:
        lines.append(f"{row['label_idx']}\t{row['folder_name']}\t{row['class_name']}")
    (output_root / "label_mapping.tsv").write_text("\n".join(lines) + "\n")


def export_train_subset(train_ds, output_root: Path, num_classes: int, args):
    target_per_class = args.train_per_class
    saved_per_class = [0 for _ in range(num_classes)]
    done_classes = 0
    total_target = target_per_class * num_classes

    tqdm = get_progress_bar()
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_target, desc="Export train", unit="img")

    for sample in train_ds:
        label = int(sample["label"])
        if saved_per_class[label] >= target_per_class:
            continue

        class_dir = output_root / "train" / class_folder_name(label, num_classes)
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{saved_per_class[label]:06d}.jpg"
        save_rgb_jpeg(sample["image"], out_path, args.image_quality)

        saved_per_class[label] += 1
        if progress is not None:
            progress.update(1)

        if saved_per_class[label] == target_per_class:
            done_classes += 1

        if done_classes == num_classes:
            break

    if progress is not None:
        progress.close()

    if min(saved_per_class) != target_per_class:
        raise RuntimeError(
            "Train export finished before all classes reached the requested quota. "
            f"Minimum exported per class: {min(saved_per_class)}"
        )

    return saved_per_class


def export_val_subset(val_ds, output_root: Path, num_classes: int, args):
    use_full_val = args.val_per_class <= 0
    saved_per_class = [0 for _ in range(num_classes)]
    done_classes = 0
    total_target = None if use_full_val else args.val_per_class * num_classes

    tqdm = get_progress_bar()
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_target, desc="Export val", unit="img")

    for sample in val_ds:
        label = int(sample["label"])
        if (not use_full_val) and saved_per_class[label] >= args.val_per_class:
            continue

        class_dir = output_root / "val" / class_folder_name(label, num_classes)
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{saved_per_class[label]:06d}.jpg"
        save_rgb_jpeg(sample["image"], out_path, args.image_quality)

        saved_per_class[label] += 1
        if progress is not None:
            progress.update(1)

        if (not use_full_val) and saved_per_class[label] == args.val_per_class:
            done_classes += 1
            if done_classes == num_classes:
                break

    if progress is not None:
        progress.close()

    if (not use_full_val) and min(saved_per_class) != args.val_per_class:
        raise RuntimeError(
            "Validation export finished before all classes reached the requested quota. "
            f"Minimum exported per class: {min(saved_per_class)}"
        )

    return saved_per_class


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    prepare_output_root(output_root, args.overwrite)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  pip install -U datasets huggingface_hub pillow tqdm"
        ) from exc

    train_ds = load_dataset(args.dataset, split="train", streaming=True, token=True)
    val_ds = load_dataset(args.dataset, split="validation", streaming=True, token=True)

    if args.train_shuffle_buffer > 0:
        train_ds = train_ds.shuffle(seed=args.seed, buffer_size=args.train_shuffle_buffer)

    class_names = get_class_names_from_features(train_ds)
    num_classes = len(class_names)
    write_label_mapping(output_root, class_names)

    print(f"[HF] dataset={args.dataset}")
    print(f"[HF] num_classes={num_classes}")
    print(f"[Out] output_root={output_root}")
    print(f"[Train] target_per_class={args.train_per_class}")
    print(f"[Val] target_per_class={'full' if args.val_per_class <= 0 else args.val_per_class}")
    print(f"[Train] shuffle_buffer={args.train_shuffle_buffer}")

    train_counts = export_train_subset(train_ds, output_root, num_classes, args)
    val_counts = export_val_subset(val_ds, output_root, num_classes, args)

    summary = {
        "dataset": args.dataset,
        "output_root": str(output_root),
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "seed": args.seed,
        "image_quality": args.image_quality,
        "num_classes": num_classes,
        "train_total_exported": int(sum(train_counts)),
        "val_total_exported": int(sum(val_counts)),
        "train_min_per_class": int(min(train_counts)),
        "train_max_per_class": int(max(train_counts)),
        "val_min_per_class": int(min(val_counts)),
        "val_max_per_class": int(max(val_counts)),
    }
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2))

    print("[Done] Export finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
