#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sope_wds_tools.py

All-in-one utility for:
1) Converting the SOPE dataset tree to WebDataset shards (~1GB each).
2) Loading it efficiently with webdataset (variable-height batches).
3) Reversing shards back to the original tree layout.
4) Benchmarking dataload speed.

Author: you 🫶
"""
import argparse, io, json, os, re, sys, time, tarfile, hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional, List

import numpy as np

# Optional but recommended
try:
    import cv2  # for EXR reading when generating .npy helper copies
except Exception:
    cv2 = None

# WebDataset
import webdataset as wds

import io, numpy as np, json, torch, webdataset as wds
# from cutoop.data_loader import Dataset as SopeDataset
import sys
sys.path.append("/home/mirshad7/HunyuanWorld-Mirror/training/data/datasets")
from utils import get_intrinsic_matrix, resize_like_vggt, vis_one_batch
from webdataloader_utils import SopeWebDataset

def decode_sope_sample(sample):
    """
    Converts raw WebDataset bytes (with original filenames like 0000_color.png)
    into the same tensors your dataset produced.
    """
    # ---- 1. Locate the right keys dynamically ----
    def find_key(contains):
        for k in sample.keys():
            if contains in k:
                return k
        return None

    base = sample["__key__"].split("/")[-1]  # e.g. '0003'
    color_bytes = sample[f"color.png"]
    depth_bytes = sample[f"depth.exr"]
    meta_bytes  = sample[f"meta.json"]
    heat_bytes  = sample.get(f"heatmap.npz", None)
    pose_bytes  = sample.get(f"pose_map.npz", None)

    # ---- 2. Decode with SopeWebDataset (no io.BytesIO wrappers) ----
    color = SopeWebDataset.load_color(color_bytes)
    depth = SopeWebDataset.load_depth(depth_bytes)
    meta  = SopeWebDataset.load_meta(meta_bytes)

    heat, pose = None, None
    if heat_bytes is not None and pose_bytes is not None:
        heat = np.load(io.BytesIO(heat_bytes))["heatmap"]
        pose = np.load(io.BytesIO(pose_bytes))["abs_pose"]

    # heat, pose = None, None
    # if heat_key and pose_key:
    #     heat = np.load(io.BytesIO(sample[heat_key]))["heatmap"]
    #     pose = np.load(io.BytesIO(sample[pose_key]))["abs_pose"]

    # ---- 3. Compute intrinsics ----
    image_width, image_height = color.shape[1], color.shape[0]
    K = get_intrinsic_matrix(meta.camera.intrinsics, image_width, image_height)

    #print("original RGB, depth, heat, pose, K shapes:", color.shape, depth.shape, heat.shape, pose.shape, K.shape)
    # ---- 4. VGGT-style resize ----
    color, depth, heat, pose, K = resize_like_vggt(
        color, depth=depth, heat=heat, pose=pose, K=K
    )

    # ---- 5. Convert to tensors + normalize ----
    rgb = torch.from_numpy(color.copy()).permute(2, 0, 1).float() / 255.0
    depth = torch.from_numpy(depth.copy()).float()

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    rgb = (rgb - mean) / std

    sample_out = {
        "rgb": rgb,
        "depth": depth,
        "K": torch.from_numpy(K).float(),
        # "prefix": sample["__key__"],
    }

    if heat is not None and pose is not None:
        sample_out["heatmap"] = torch.from_numpy(heat.copy()).unsqueeze(0).float()
        sample_out["pose_map"] = torch.from_numpy(pose.copy()).permute(2, 0, 1).float()

    return sample_out
# ----------------------------
# Helpers
# ----------------------------
def human_bytes(n: int) -> str:
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.2f}{u}"
        n /= 1024

def read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def write_bytes(p: Path, b: bytes):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(b)

def to_key_from_relpath(rel: str) -> str:
    # WebDataset sample __key__ should be unique and safe; avoid slashes.
    # Keep info in index.json for perfect reversibility.
    # Normalize separators and strip extensions; append a short hash for uniqueness.
    # rel example: "00/train/ikea/0000/0000"
    key_core = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
    h = hashlib.md5(rel.encode()).hexdigest()[:8]
    return f"{key_core}_{h}"

def find_sample_prefixes(root: Path, mode = "train") -> List[Path]:
    """
    Find all 'prefixes' like .../0000_<suffix> with base names per frame.
    We discover them by finding *_color.png and stripping the trailing suffix.
    """


    color_files = list(root.glob(f"**/{mode}/**/*_color.png"))
    prefixes = []
    for cf in color_files:
        base = cf.name[: -len("_color.png")]  # e.g. '0000'
        prefixes.append(cf.with_name(base))   # e.g. .../0000
    prefixes = sorted(prefixes)
    return prefixes

def load_depth_to_npy_bytes(exr_bytes: bytes) -> Optional[bytes]:
    """Decode EXR -> float32 numpy and return as .npy bytes (for fast training).
    If OpenCV EXR not available, return None (we still preserve the original EXR)."""
    if cv2 is None:
        return None
    # cv2.imdecode works for many formats including EXR on most builds
    buf = np.frombuffer(exr_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # ensure float32 for compactness/speed
    arr = img.astype(np.float32, copy=False)
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()

def npy_bytes_from_exr_file(path: Path) -> Optional[bytes]:
    try:
        return load_depth_to_npy_bytes(read_bytes(path))
    except Exception:
        return None

def npy_bytes_from_mask_exr_file(path: Path) -> Optional[bytes]:
    # Often mask is single-channel float; store as float32 npy as well.
    return npy_bytes_from_exr_file(path)

# ----------------------------
# CONVERT → WebDataset
# ----------------------------

def do_convert(
    root: Path,
    out_dir: Path,
    shard_size_gb: float = 1.0,
    # include_original_exr: bool = True,
    add_npy_fastpaths: bool = True,
    verbose: bool = True,
    mode: str = "train",
    delete_original : bool = True,
    max_shards: int = 100
):
    """
    Converts SOPE dataset to WebDataset shards that preserve
    the exact filename structure expected by SopeDataset.
    Example tar contents:
        00/train/ikea/0000/0000_color.png
        00/train/ikea/0000/0000_depth.exr
        00/train/ikea/0000/0000_heatmap.npz
        ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "sope-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=int(shard_size_gb * (1024**3)))

    prefixes = find_sample_prefixes(root, mode=mode)
    if verbose:
        print(f"Found {len(prefixes)} samples under {root}")

    count = 0
    shard_count = 0
    for pref in prefixes:

        # Stop if 100 shards already written
        if shard_count >= max_shards:
            print(f"🛑 Reached {max_shards} shards. Stopping conversion.")
            break

        dirname = str(pref.parent.relative_to(root)).replace("\\", "/")  # e.g. 00/train/ikea/0000
        base = pref.name  # e.g. 0000
        rel_prefix = f"{dirname}/{base}"  # used as tar prefix

        files = {
            "color": pref.with_name(f"{base}_color.png"),
            "depth": pref.with_name(f"{base}_depth.exr"),
            "mask":  pref.with_name(f"{base}_mask.exr"),
            "heat":  pref.with_name(f"{base}_heatmap.npz"),
            "pose":  pref.with_name(f"{base}_pose_map.npz"),
            "meta":  pref.with_name(f"{base}_meta.json"),
        }

        sample = {"__key__": f"{dirname}/{base}"}

        if not files["color"].exists():
            continue  # skip incomplete

        def add_file(extname, path):
            if path.exists():
                sample[extname] = read_bytes(path)

        add_file("color.png", files["color"])
        add_file("meta.json", files["meta"])
        add_file("heatmap.npz", files["heat"])
        add_file("pose_map.npz", files["pose"])
        add_file("depth.exr", files["depth"])
        add_file("mask.exr", files["mask"])

        sink.write(sample)
        count += 1

        # Update shard count based on writer’s internal state
        shard_count = sink.shard
        if verbose and count % 1000 == 0:
            print(f"[convert] wrote {count} samples")

    sink.close()
    print(f"✅ Done: wrote {count} samples to {out_dir}")

# ----------------------------
# LOAD (WebDataset → PyTorch)
# ----------------------------

def build_sope_wds_loader(
    shards_glob: str,
    shuffle_shards: int = 50,
    shuffle_samples: int = 5000,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """
    WebDataset loader that streams shards and decodes samples exactly like
    SopeCentersnapDataset.__getitem__.
    """
    # ds = (
    #     wds.WebDataset(shards_glob)
    #     # .shuffle(1000)
    #     # we read raw bytes; no automatic decoding
    #     # .map_dict(__key__=lambda x: x)  # preserve key for prefix
    #     .map(decode_sope_sample)
    # )
    # # from torch.utils.data import DataLoader
    # # loader = DataLoader(ds, batch_size=32, num_workers=4)

    # print("Building WebLoader from shards:", shards_glob)
    # # print("len of webdataset:", len(ds))
    # loader = wds.WebLoader(
    #     ds,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     pin_memory=pin_memory,
    #     persistent_workers=persistent_workers,
    # )

    ds = (
        wds.WebDataset(shards_glob, shardshuffle=10)
        .shuffle(5000)  # ✅ sample-level buffer shuffle (tune buffer size) 
        .map(decode_sope_sample)
        .batched(batch_size, partial=False)
    )

    # ds = (
    #     wds.WebDataset(shards_glob)
    #     .map(decode_sope_sample)
    #     .batched(batch_size, partial=False)
    # )
    return wds.WebLoader(
        ds,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        batch_size=None,
    )

    # ds = (
    #     wds.WebDataset(shards_glob, shardshuffle=10)
    #     .shuffle(1000)  # ✅ sample-level buffer shuffle (tune buffer size) 
    #     .map(decode_sope_sample)
    #     .batched(batch_size, partial=False)
    # )

    # # ds = (
    # #     wds.WebDataset(shards_glob)
    # #     .map(decode_sope_sample)
    # #     .batched(batch_size, partial=False)
    # # )
    # return wds.WebLoader(
    #     ds,
    #     num_workers=num_workers,
    #     batch_size=None,
    # )

    # return loader

# ----------------------------
# REVERSE (WebDataset → original tree)
# ----------------------------
def do_reverse(shards_glob: str, out_root: Path, overwrite: bool = False):
    """
    Reconstruct original tree using index.json and preserved original files.
    If EXR not present in shard (because user disabled), will reconstruct from .npy when possible.
    """
    ds = (
        wds.WebDataset(shards_glob, shardshuffle=False)
        .to_tuple("__key__", "index.json", "png", "json", "depth.exr", "mask.exr",
                  "depth.npy", "mask.npy", "heat.npz", "pose.npz")
    )

    restored = 0
    for item in ds:
        (key, index_b, color_b, meta_b, depth_exr_b, mask_exr_b,
         depth_npy_b, mask_npy_b, heat_b, pose_b) = item

        index = json.loads(index_b.decode("utf-8"))
        rel_dir = index["relative_dir"]
        base = index["base"]

        target_dir = out_root / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        def save_if(data, name):
            if data is None:
                return
            write_bytes(target_dir / f"{base}_{name}", data)

        # Restore original files if present
        save_if(color_b, "color.png")
        save_if(meta_b, "meta.json")
        save_if(heat_b, "heatmap.npz")
        save_if(pose_b, "pose_map.npz")

        if depth_exr_b is not None:
            save_if(depth_exr_b, "depth.exr")
        elif depth_npy_b is not None:
            # fallback: write EXR alternative as .npy (lossless content but different format)
            write_bytes(target_dir / f"{base}_depth.npy", depth_npy_b)

        if mask_exr_b is not None:
            save_if(mask_exr_b, "mask.exr")
        elif mask_npy_b is not None:
            write_bytes(target_dir / f"{base}_mask.npy", mask_npy_b)

        restored += 1
        if restored % 1000 == 0:
            print(f"[reverse] restored {restored} samples")

    print(f"Reverse complete. Restored {restored} samples under {out_root}")

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="SOPE ⇄ WebDataset converter/loader/benchmark/reverser")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert SOPE tree → WebDataset shards")
    c.add_argument("--root", type=Path, required=True, help="SOPE root (folder containing 00..50)")
    c.add_argument("--out", type=Path, required=True, help="Output dir for .tar shards")
    c.add_argument("--shard_size_gb", type=float, default=1.0)
    c.add_argument("--no-original-exr", action="store_true", help="Do not store original EXR bytes")
    c.add_argument("--no-fast-npy", action="store_true", help="Do not add .npy fast copies")
    c.add_argument("--quiet", action="store_true")
    c.add_argument("--mode", type=str, default="train")


    l = sub.add_parser("loadtest", help="iterate the shards quickly (sanity check)")
    l.add_argument("--shards", type=str, required=True, help="Path/glob to shards, e.g. /data/wds/sope-{000000..000127}.tar")
    l.add_argument("--batch_size", type=int, default=4)
    l.add_argument("--num_workers", type=int, default=4)
    l.add_argument("--persistent_workers", action="store_true")
    l.add_argument("--no_pin_memory", action="store_true")
    l.add_argument("--measure", type=int, default=2000)

    r = sub.add_parser("reverse", help="reconstruct original tree from shards")
    r.add_argument("--shards", type=str, required=True)
    r.add_argument("--out", type=Path, required=True)

    b = sub.add_parser("benchmark", help="quick throughput benchmark (WDS)")
    b.add_argument("--shards", type=str, required=True)
    b.add_argument("--batch_size", type=int, default=4)
    b.add_argument("--num_workers", type=int, default=8)
    b.add_argument("--persistent_workers", action="store_true")
    b.add_argument("--no_pin_memory", action="store_true")
    b.add_argument("--warmup", type=int, default=200)
    b.add_argument("--measure", type=int, default=5000)
    args = ap.parse_args()

    if args.cmd == "convert":
        do_convert(
            root=args.root,
            out_dir=args.out,
            shard_size_gb=args.shard_size_gb,
            # include_original_exr=not args.no_original_exr,
            mode = args.mode,
            add_npy_fastpaths=not args.no_fast_npy,
            verbose=not args.quiet,
        )

    elif args.cmd == "loadtest":
        # -----------------------------------------------------
        # use your new loader that mirrors SopeCentersnapDataset
        # -----------------------------------------------------
        from base_dataset import BaseCentersnapDataset

        loader = build_sope_wds_loader(
            shards_glob=args.shards,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            persistent_workers=args.persistent_workers,
        )

        # -----------------------------------------------------
        # benchmark: iterate a few batches
        # -----------------------------------------------------
        import time
        n = 0
        t0 = time.time()

        from tqdm import tqdm
        # time 50 batches
        import time
        t0 = time.time()
        for i, batch in enumerate(tqdm(loader, total=500)):
            print("depth min/max:", batch["depth"].min().item(), batch["depth"].max().item())
            # print("depth min/max:", batch["depth"].min().item(), batch["depth"].max().item())
            # vis_one_batch(batch, i)
            if i == 500:
                break
        t_total = time.time() - t0
        print(f"Time for 500 batches: {t_total:.2f} seconds ({t_total/500:.2f} s/batch)")

if __name__ == "__main__":
    main()
