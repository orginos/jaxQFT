#!/usr/bin/env python3
"""Extract gauge field `q` from a jaxQFT MCMC checkpoint."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract gauge field from checkpoint pickle")
    ap.add_argument("--checkpoint", type=str, required=True, help="input checkpoint .pkl path")
    ap.add_argument("--out", type=str, required=True, help="output path (.npy or .pkl/.pickle)")
    ap.add_argument(
        "--key",
        type=str,
        default="q",
        help="payload key to extract from checkpoint dict (default: q)",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)

    with ckpt_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload is not a dict: {type(payload).__name__}")
    if args.key not in payload:
        raise ValueError(f"Checkpoint has no key '{args.key}'")

    q = np.asarray(payload[args.key])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".npy":
        np.save(out_path, q)
    elif ext in (".pkl", ".pickle"):
        with out_path.open("wb") as f:
            pickle.dump(q, f)
    else:
        raise ValueError("Output extension must be .npy or .pkl/.pickle")

    print(f"Wrote: {out_path}")
    print(f"Shape: {tuple(q.shape)} dtype={q.dtype}")


if __name__ == "__main__":
    main()

