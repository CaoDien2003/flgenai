#!/usr/bin/env python3
"""
Run:
    python split_cifar10.py --out ./subset
    python split_cifar10.py --out ./subset --mode disjoint --alpha 0.5
"""

import argparse
import json
import os
import pickle
import tarfile
import urllib.request

import numpy as np
from PIL import Image

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_DIR = "cifar-10-batches-py"
NUM_CLASSES = 10


# Download
def download_cifar10(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    extracted = os.path.join(root, CIFAR_DIR)
    if os.path.isdir(extracted):
        print(f"[download] found {extracted}, skipping")
        return extracted

    archive = os.path.join(root, os.path.basename(CIFAR_URL))
    if not os.path.isfile(archive):
        print(f"[download] {CIFAR_URL} -> {archive}")

        def progress(blocks, block_size, total):
            done = min(blocks * block_size, total)
            print(f"\r[download] {done / 2**20:6.1f} / {total / 2**20:.1f} MB", end="")

        urllib.request.urlretrieve(CIFAR_URL, archive, reporthook=progress)
        print()

    print(f"[download] extracting {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def load_cifar10(extracted: str):
    """Load train + test (60,000 images) as (N,32,32,3) uint8, labels, class names."""
    def unpickle(name):
        with open(os.path.join(extracted, name), "rb") as f:
            return pickle.load(f, encoding="bytes")

    xs, ys = [], []
    for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        d = unpickle(name)
        xs.append(d[b"data"])
        ys.extend(d[b"labels"])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.asarray(ys, dtype=np.int64)
    names = [n.decode() for n in unpickle("batches.meta")[b"label_names"]]
    print(f"[data] {len(y)} images, {NUM_CLASSES} classes: {names}")
    return x, y, names


# Partition
def counts_with_floor(total: int, probs: np.ndarray, floor: int,
                      rng: np.random.Generator) -> np.ndarray:
    """`floor` per bucket first, then the remainder ~ Multinomial(probs)."""
    rest = total - floor * len(probs)
    if rest < 0:
        raise ValueError(f"floor {floor} x {len(probs)} buckets exceeds total {total}")
    return floor + rng.multinomial(rest, probs)


def split_paper(y: np.ndarray, n_subsets: int, alpha: float, floor: int,
                rng: np.random.Generator):
    by_class = [np.flatnonzero(y == c) for c in range(NUM_CLASSES)]
    per_subset = len(y) // n_subsets
    subsets = []
    for _ in range(n_subsets):
        counts = counts_with_floor(per_subset, rng.dirichlet([alpha] * NUM_CLASSES),
                                   floor, rng)
        counts = np.minimum(counts, [len(p) for p in by_class])   # unique within a subset
        subsets.append(np.concatenate([rng.choice(by_class[c], counts[c], replace=False)
                                       for c in range(NUM_CLASSES)]))
    return subsets


def split_disjoint(y: np.ndarray, n_subsets: int, alpha: float, floor: int,
                   rng: np.random.Generator):
    """Each image goes to exactly one subset: each class split across subsets by Dir(alpha)."""
    parts = [[] for _ in range(n_subsets)]
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(y == c))
        counts = counts_with_floor(len(idx), rng.dirichlet([alpha] * n_subsets), floor, rng)
        for s, chunk in enumerate(np.split(idx, np.cumsum(counts)[:-1])):
            parts[s].append(chunk)
    return [np.concatenate(p) for p in parts]


def save_subsets(x, y, names, subsets, out: str):
    """Write PNGs as <out>/subset{k}/class_{i}_{name}/{global_index:05d}.png."""
    for k, idx in enumerate(subsets, start=1):
        root = os.path.join(out, f"subset{k}")
        for c in range(NUM_CLASSES):                 # every class folder, even if empty
            os.makedirs(os.path.join(root, f"class_{c}_{names[c]}"), exist_ok=True)
        for i in idx:
            c = int(y[i])
            Image.fromarray(x[i]).save(
                os.path.join(root, f"class_{c}_{names[c]}", f"{i:05d}.png"))
        print(f"[save] {root}: {len(idx)} images")


def report(y, names, subsets, out: str, mode: str, alpha: float, seed: int):
    table = np.array([np.bincount(y[idx], minlength=NUM_CLASSES) for idx in subsets])
    header = "subset   " + " ".join(f"{c:>5d}" for c in range(NUM_CLASSES)) + "   total"
    print("\n" + header)
    for k, row in enumerate(table, start=1):
        print(f"subset{k:<3d}" + " ".join(f"{v:>5d}" for v in row) + f"   {row.sum():>5d}")

    seen = np.concatenate(subsets)
    n_unique = len(np.unique(seen))
    print(f"\n[stats] {len(seen)} assignments, {n_unique} unique images "
          f"({len(seen) - n_unique} repeats across subsets)")

    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump({"mode": mode, "alpha": alpha, "seed": seed, "class_names": names,
                   "counts": {f"subset{k}": row.tolist() for k, row in enumerate(table, 1)},
                   "indices": {f"subset{k}": idx.tolist() for k, idx in enumerate(subsets, 1)}},
                  f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping distribution.png")
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.8 / NUM_CLASSES
    pos = np.arange(len(subsets))
    for c in range(NUM_CLASSES):
        ax.bar(pos + (c - NUM_CLASSES / 2 + 0.5) * width, table[:, c], width, label=str(c))
    ax.set_xticks(pos, [f"set{k}" for k in range(1, len(subsets) + 1)])
    ax.set_xlabel("Subset"); ax.set_ylabel("Number of images")
    ax.legend(title="Class", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "distribution.png"), dpi=120)
    print(f"[plot] {os.path.join(out, 'distribution.png')}")


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Download CIFAR-10 and split into non-IID subsets")
    p.add_argument("--out", default="./subset", help="output directory for subset1..subsetN")
    p.add_argument("--cache", default="./cifar10_raw", help="where the raw CIFAR-10 is downloaded")
    p.add_argument("--subsets", type=int, default=10)
    p.add_argument("--mode", default="paper", choices=["paper", "disjoint"],
                   help="paper = reproduces subset as in paper; disjoint = no shared images")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Dirichlet concentration (smaller = more skewed); ~1.0")
    p.add_argument("--floor", type=int, default=50, help="minimum images per class per subset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    x, y, names = load_cifar10(download_cifar10(args.cache))
    split = split_paper if args.mode == "paper" else split_disjoint
    subsets = split(y, args.subsets, args.alpha, args.floor, rng)

    os.makedirs(args.out, exist_ok=True)
    save_subsets(x, y, names, subsets, args.out)
    report(y, names, subsets, args.out, args.mode, args.alpha, args.seed)


if __name__ == "__main__":
    main()
