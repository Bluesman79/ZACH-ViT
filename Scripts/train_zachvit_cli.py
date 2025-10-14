#!/usr/bin/env python3
"""
ZACH-ViT: Command-line interface for model training and evaluation.
This script wraps the run_zach_vit_time() pipeline.
"""

import argparse
from zachvit.model_utils import run_zach_vit_time


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ZACH-ViT model on VIS / SSDA datasets."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="../Data/",
        help="Base directory containing train/val/test folders."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=23,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=53,
        help="Pixel threshold (0–255) below which grayscale values are set to zero."
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W0", "W1"),
        help="Optional class weights for imbalanced training, e.g. --class_weights 0.3 0.7"
    )

    args = parser.parse_args()

    # Convert class_weights into dict format if provided
    cw = None
    if args.class_weights:
        cw = {0: args.class_weights[0], 1: args.class_weights[1]}

    print("\n🚀 Starting ZACH-ViT training ...\n")
    model, val_df, test_df = run_zach_vit_time(
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold=args.threshold,
        class_weights=cw,
        base_dir=args.base_dir
    )

    print("\n✅ Training complete. Metrics saved.\n")


if __name__ == "__main__":
    main()
