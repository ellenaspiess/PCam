#!/usr/bin/env python3
from __future__ import annotations
"""Select representative stain-reference patches from PCam train split."""

import argparse
import json
from pathlib import Path

import numpy as np
from torchvision import datasets


def _compute_patch_score(img_u8: np.ndarray) -> tuple[float, float]:
    """Return ``(score, tissue_fraction)`` for reference suitability.

    Higher score indicates tissue-rich, color-varied, and non-flat patches.
    """
    x = img_u8.astype(np.float32) / 255.0
    gray = x.mean(axis=2)
    sat = x.max(axis=2) - x.min(axis=2)

    tissue_mask = (gray < 0.9) & (sat > 0.05)
    tissue_fraction = float(tissue_mask.mean())
    if tissue_fraction < 0.4:
        return -1.0, tissue_fraction

    if np.any(tissue_mask):
        sat_mean = float(sat[tissue_mask].mean())
        gray_std = float(gray[tissue_mask].std())
    else:
        sat_mean = 0.0
        gray_std = 0.0

    # Prefer tissue-rich, color-varied, non-flat patches.
    score = tissue_fraction * (sat_mean + 1e-6) * (gray_std + 1e-6)
    return float(score), tissue_fraction


def main() -> None:
    """CLI entry point for selecting and saving stain reference artifacts."""
    parser = argparse.ArgumentParser(description="Select stain normalization reference patches from PCam train split.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="experiments/stain_refs")
    parser.add_argument("--num-candidates", type=int, default=5000)
    parser.add_argument("--num-select", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.PCAM(root=args.data_root, split="train", transform=None, download=True)

    rng = np.random.default_rng(args.seed)
    n = len(ds)
    candidate_count = min(args.num_candidates, n)
    candidate_indices = rng.choice(n, size=candidate_count, replace=False)

    rows = []
    for idx in candidate_indices.tolist():
        img, _ = ds[idx]
        img_u8 = np.asarray(img.convert("RGB"), dtype=np.uint8)
        score, tissue_fraction = _compute_patch_score(img_u8)
        if score > 0:
            rows.append({"index": int(idx), "score": float(score), "tissue_fraction": float(tissue_fraction)})

    if not rows:
        raise RuntimeError("No suitable reference candidates found. Try increasing --num-candidates.")

    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = rows[: min(args.num_select, len(rows))]

    # Use the best-scoring patch as fixed reference image.
    primary_index = selected[0]["index"]
    primary_img, _ = ds[primary_index]
    primary_u8 = np.asarray(primary_img.convert("RGB"), dtype=np.uint8)

    ref_img_path = output_dir / "reference_image.npy"
    np.save(ref_img_path, primary_u8)

    manifest = {
        "data_root": args.data_root,
        "seed": args.seed,
        "num_candidates": candidate_count,
        "num_selected": len(selected),
        "primary_index": primary_index,
        "reference_image_path": str(ref_img_path),
        "selected": selected,
    }

    manifest_path = output_dir / "references.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved reference image: {ref_img_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
