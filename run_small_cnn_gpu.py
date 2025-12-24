#!/usr/bin/env python3
"""run_small_cnn_gpu.py

CLI helper to run SmallCNN on a GPU with sensible defaults.
"""
from pathlib import Path
import argparse
import logging
import os
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.training.train_small_cnn import train


def parse_args():
    p = argparse.ArgumentParser(description="Run SmallCNN on GPU (or CPU fallback)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--limit", type=int, default=None, help="Limit samples per split for quick tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Device to use (cuda|cpu). If omitted uses cuda when available.")
    p.add_argument("--outdir", type=str, default="experiments/runs")
    return p.parse_args()


def setup_logging(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run_small_cnn_gpu.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return log_path


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    log_path = setup_logging(outdir)

    logging.info("Starting SmallCNN run (GPU helper)")
    logging.info(f"Args: {args}")

    device = args.device or ("cuda" if ("cuda" in os.environ.get("CUDA_VISIBLE_DEVICES", "") or __import__("torch").cuda.is_available()) else "cpu")
    logging.info(f"Using device: {device}")

    # Seed
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Call train
    train(
        data_root="data/raw",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        limit_per_split=args.limit,
        dropout_p=args.dropout,
        num_workers=args.num_workers,
    )

    logging.info("Run finished. Check outputs in %s", outdir)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
