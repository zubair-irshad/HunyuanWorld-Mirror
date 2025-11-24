import os, io, json, torch, numpy as np, webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import glob

"""
sope_webdataset_loader.py
---------------------------------
Drop-in loader that mimics SopeDataset but works from WebDataset in-memory bytes.

Use this instead of SopeDataset when your data come from .tar shards.
"""
import io, cv2, numpy as np, json
from time import time
from PIL import Image as _Image
from typing import Tuple

from cutoop.image_meta import ImageMetaData

# WebDataset
import io, json
from typing import Tuple
import numpy as np

# Optional but recommended
try:
    import cv2  # for EXR reading when generating .npy helper copies
except Exception:
    cv2 = None

import io, numpy as np, json, torch, webdataset as wds
import sys
# sys.path.append("/home/mirshad7/HunyuanWorld-Mirror/training/data/datasets")
from utils import get_intrinsic_matrix, resize_like_vggt
#from utils import get_intrinsic_matrix, resize_like_vggt
import os
import glob
import random
from torch.utils.data import get_worker_info


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def decode_sope_sample(sample):
    color_bytes = sample.get("color.png")
    depth_bytes = sample.get("depth.exr")
    meta_bytes  = sample.get("meta.json")
    if color_bytes is None or depth_bytes is None or meta_bytes is None:
        raise KeyError(f"Missing required keys: {sample.keys()}")
    color = SopeWebDataset.load_color(color_bytes)
    depth = SopeWebDataset.load_depth(depth_bytes)


    # print(f"depth MB: {depth.nbytes/1e6}")
    meta  = SopeWebDataset.load_meta(meta_bytes)
    heat_bytes = sample.get("heatmap.npz")
    pose_bytes = sample.get("pose_map.npz")
    heat = pose = None
    if heat_bytes and pose_bytes:
        heat = np.load(io.BytesIO(heat_bytes))["heatmap"]
        pose = np.load(io.BytesIO(pose_bytes))["abs_pose"]
    image_width, image_height = color.shape[1], color.shape[0]
    K = get_intrinsic_matrix(meta.camera.intrinsics, image_width, image_height)
    color, depth, heat, pose, K = resize_like_vggt(color, depth=depth, heat=heat, pose=pose, K=K)
    if depth is None or color is None:
        raise ValueError("Color/Depth decode returned None")
    rgb = torch.from_numpy(np.ascontiguousarray(color)).permute(2, 0, 1).float() / 255.0
    depth_t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).float()
    # Clamp depth to reasonable range to avoid memory explosion
    depth_t = depth_t.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 25.0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    out = {"rgb": rgb, "depth": depth_t, "K": torch.from_numpy(K).float()}
    if heat is not None and pose is not None:
        out["heatmap"] = torch.from_numpy(np.ascontiguousarray(heat)).unsqueeze(0).float()
        out["pose_map"] = torch.from_numpy(np.ascontiguousarray(pose)).permute(2, 0, 1).float()
    return out

def _imdecode(data: bytes, mode=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Decode compressed image bytes directly to numpy."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, mode)
    if img is None:
        raise IOError("cv2.imdecode failed")
    return img


def load_color_bytes(data: bytes) -> np.ndarray:
    """Load RGB color image from in-memory bytes (PNG/JPG)."""
    img = _imdecode(data, cv2.IMREAD_COLOR)
    return img[:, :, ::-1]  # BGR→RGB


def load_coord_bytes(data: bytes) -> np.ndarray:
    """Decode NOCS coordinate map (PNG) from bytes."""
    img = _imdecode(data, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.clip(0, 1)
    img[:, :, 2] = 1 - img[:, :, 2]
    return img - 0.5


def load_ir_bytes(data: bytes) -> np.ndarray:
    """Decode single-channel IR image from bytes."""
    arr = np.frombuffer(data, np.uint8)
    pil = _Image.open(io.BytesIO(arr)).convert("L")
    return np.array(pil)

def load_depth_bytes(data: bytes) -> np.ndarray:
    """Decode depth; raise explicit error if it fails."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None or img.size == 0:
        raise IOError("Depth decode failed (cv2.imdecode returned None/empty).")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32, copy=False)

def load_mask_bytes(data: bytes) -> np.ndarray:
    """Decode EXR mask from bytes, returning uint8 mask."""
    img = _imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(img.shape) == 3:
        img = img[:, :, 2]
    return np.array(img * 255, dtype=np.uint8)


def load_mask_sam_bytes(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Decode SAM mask npz from bytes."""
    with io.BytesIO(data) as f:
        d = np.load(f)
        return d["masks"], d["mask_ids"]


def load_normal_bytes(data: bytes) -> np.ndarray:
    """Decode EXR normal map from bytes, range [-1, 1]."""
    img = _imdecode(data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 0] = -img[:, :, 0]
    return img

def load_meta_bytes(data: bytes):
    """Parse JSON bytes directly into an ImageMetaData instance (no temp file)."""
    meta_dict = json.loads(data.decode("utf-8"))
    # construct the same object ImageMetaData.load_json() would return
    return ImageMetaData(**meta_dict)

# --------------------------------------------------------------------------
# Mapping table mirroring SopeDataset.LOADER_MAP
# --------------------------------------------------------------------------

LOADER_MAP = {
    "color": ("color.png", load_color_bytes),
    "coord": ("coord.png", load_coord_bytes),
    "ir_l": ("ir_l.png", load_ir_bytes),
    "ir_r": ("ir_r.png", load_ir_bytes),
    "depth": ("depth.exr", load_depth_bytes),
    "depth_syn": ("depth_syn.exr", load_depth_bytes),
    "mask": ("mask.exr", load_mask_bytes),
    "mask_sam": ("mask_sam.npz", load_mask_sam_bytes),
    "normal": ("normal.exr", load_normal_bytes),
    "meta": ("meta.json", load_meta_bytes),
}

# --------------------------------------------------------------------------
# Convenience API compatible with SopeDataset
# --------------------------------------------------------------------------

class SopeWebDataset:
    """A byte-based variant of SopeDataset for WebDataset samples."""
    @staticmethod
    def load_color(data: bytes): return load_color_bytes(data)
    @staticmethod
    def load_depth(data: bytes): return load_depth_bytes(data)
    @staticmethod
    def load_mask(data: bytes): return load_mask_bytes(data)
    @staticmethod
    def load_meta(data: bytes): return load_meta_bytes(data)
    @staticmethod
    def load_normal(data: bytes): return load_normal_bytes(data)


os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CV2_NUM_THREADS": "0",
})

# import cv2
# def _imdecode(data: bytes, mode=cv2.IMREAD_UNCHANGED) -> np.ndarray:
#     """Decode compressed image bytes directly to numpy."""
#     arr = np.frombuffer(data, np.uint8)
#     img = cv2.imdecode(arr, mode)
#     if img is None:
#         raise IOError("cv2.imdecode failed")
#     return img
# def load_color_bytes(data: bytes) -> np.ndarray:
#     """Load RGB color image from in-memory bytes (PNG/JPG)."""
#     img = _imdecode(data, cv2.IMREAD_COLOR)
#     return img[:, :, ::-1]  # BGR→RGB

# def print_worker_init(worker_id):
#     info = torch.utils.data.get_worker_info()
#     dataset = info.dataset
#     urls = None

#     # WebDataset stores shard list in the first pipeline stage
#     if hasattr(dataset, "pipeline") and len(dataset.pipeline) > 0:
#         stage = dataset.pipeline[0]
#         if hasattr(stage, "urls"):
#             urls = stage.urls

#     print(f"[Worker {worker_id}] assigned shards: {urls}")

# ----------------------------
# 🧱 Minimal decode function
# ----------------------------
# def decode_sample(sample):
#     # print("sample key", sample["__key__"])
#     key = sample["__key__"]
#     color_bytes = sample.get("color.png")
#     color = load_color_bytes(color_bytes)
#     out = {'rgb': color, "key": key}
#     return out
# ----------------------------
# 🧰 Safe shuffle wrapper
# ----------------------------

# def safe_shuffle(data, bufsize=10, initial=10, rng=None):
#     """Safely shuffle a streaming dataset with a bounded buffer.

#     Args:
#         data: an iterable of samples
#         bufsize: maximum number of samples to keep in the buffer
#         initial: number of samples to prefill before yielding
#         rng: numpy random generator (optional)
#     """
#     rng = np.random.default_rng() if rng is None else rng
#     buf = []

#     # 1️⃣ Fill initial buffer
#     for sample in data:
#         buf.append(sample)
#         if len(buf) >= initial:
#             break

#     # 2️⃣ Stream new samples while keeping buffer <= bufsize
#     for sample in data:
#         if len(buf) < bufsize:
#             buf.append(sample)
#         else:
#             k = rng.integers(0, len(buf))
#             yield buf[k]
#             buf[k] = sample

#     # 3️⃣ Drain remaining samples
#     while buf:
#         try:
#             k = rng.integers(0, len(buf))
#             yield buf.pop(k)
#         except ValueError:
#             break
# ----------------------------
# ⚙️ Build loader function
# ----------------------------
# import random
# from torch.utils.data import get_worker_info

# def debug_split_by_worker(src):
#     """Same as wds.split_by_worker, but prints shard assignments once per worker."""
#     info = get_worker_info()
#     worker_id = info.id if info else 0
#     num_workers = info.num_workers if info else 1
#     urls = list(src)
#     # Strided selection (standard split_by_worker)
#     shards = urls[worker_id::num_workers]
#     print(f"[WDS] Worker {worker_id} → {len(shards)} shards")
#     for u in shards:
#         print(f"   └─ {u}")
#     for s in shards:
#         yield s

# def build_sope_wds_loader(
#     shards_glob,
#     num_workers=4,
#     batch_size=4,
#     bufsize=2000,
#     initial=500,
#     shuffle_shards=True,
#     shuffle_samples=True,
#     seed=42,
#     resampled=False,
# ):
#     """Single-node WebDataset loader with correct worker sharding and IID shuffling."""
#     shard_paths = sorted(glob.glob(f"{shards_glob}/*.tar"))
#     assert len(shard_paths) > 0, f"No shards found in {shards_glob}"

#     if shuffle_shards:
#         random.Random(seed).shuffle(shard_paths)

#     # Important: wrap in lambda to make it lazily evaluated inside worker processes
#     shard_source = (
#         (lambda: wds.SimpleShardList(shard_paths))
#         if not resampled
#         else (lambda: wds.ResampledShards(shard_paths))
#     )

#     # Build the pipeline
#     pipeline = [
#         shard_source,
#         debug_split_by_worker,  # ensures unique shard subsets per worker
#         wds.tarfile_to_samples(handler=wds.handlers.warn_and_continue),
#     ]

#     if shuffle_samples:
#         pipeline.append(wds.shuffle(bufsize, initial=initial))

#     pipeline += [
#         wds.map(decode_sample),
#         wds.batched(batch_size, partial=True),
#     ]

#     dataset = wds.DataPipeline(*pipeline)

#     loader = wds.WebLoader(
#         dataset,
#         num_workers=num_workers,
#         batch_size=None,          # batching done inside pipeline
#         pin_memory=True,
#         prefetch_factor=(2 if num_workers > 0 else None),
#         persistent_workers=False,
#     )

#     print(f"[WDS] Initialized loader with {len(shard_paths)} shards, {num_workers} workers")
#     return loader

def build_sope_wds_loader(
    shards_glob,
    num_workers=2,
    batch_size=2,
    bufsize=5000,
    initial=500,
    shuffle_shards=True,
    shuffle_samples=True,
    seed=42,
    resampled=False,
):
    """Single-node WebDataset loader with correct worker sharding and IID shuffling."""
    shard_paths = sorted(glob.glob(f"{shards_glob}/*.tar"))
    assert len(shard_paths) > 0, f"No shards found in {shards_glob}"

    # Global shard shuffle
    if shuffle_shards:
        random.Random(seed).shuffle(shard_paths)

    # Lazily generate shard list per worker
    shard_source = (
        (lambda: wds.SimpleShardList(shard_paths))
        if not resampled
        else (lambda: wds.ResampledShards(shard_paths))
    )

    # Build the pipeline
    pipeline = [
        shard_source,
        wds.split_by_worker,  # ✅ unique shards per worker
        wds.tarfile_to_samples(handler=wds.handlers.warn_and_continue),
    ]

    if shuffle_samples:
        pipeline.append(wds.shuffle(bufsize, initial=initial))

    pipeline += [
        wds.map(decode_sope_sample),
        wds.batched(batch_size, partial=True),
    ]

    dataset = wds.DataPipeline(*pipeline)

    # Build PyTorch-style DataLoader
    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,  # batching handled by WebDataset
        pin_memory=True,
        prefetch_factor=(2 if num_workers > 0 else None),
        persistent_workers=False,
    )

    print(f"[WDS] Loader initialized: {len(shard_paths)} shards | {num_workers} workers | batch {batch_size}")
    return loader

# def build_test_wds_loader(shards_glob, num_workers=2, batch_size=4, epoch=0):

#     rng = np.random.default_rng(epoch)

#     # shard_list = list(wds.shardlists.expand_urls("/home/mirshad7/Downloads/data/Omni6DPose/SOPE_test/sope-{000000..000003}.tar"))
#     # print("Found shards:", shard_list)

#     # shards = list(wds.shardlists.expand_urls(shards_glob))
#     # shards = shards * num_workers  # duplicate list to avoid truncation
#     # ds = wds.WebDataset(
#     #     shards,
#     #     nodesplitter=wds.split_by_worker,  # each worker gets its own shard subset
#     #     handler=wds.handlers.warn_and_continue,
#     #     shardshuffle=0,
#     #     resampled=False,
#     # )

#     print("shards_glob", shards_glob)
#     # ds = wds.WebDataset(
#     #     shards_glob,
#     #     # nodesplitter=wds.split_by_worker,
#     #     nodesplitter=wds.split_by_node,
#     #     handler=wds.handlers.warn_and_continue,
#     #     shardshuffle=0,   # deterministic order of shards
#     #     resampled=False,
#     # )
#     # Replace .shuffle() with our safe version
#     # ds = ds.compose(lambda src: safe_shuffle(src, bufsize=10, initial=10, rng=rng))
#     # ds = ds.map(decode_sample).batched(batch_size, partial=True)

# # glob(f'{root}/{base_fname}/*.tar'
     
#     # ds = wds.WebDataset(
#     #     shards_glob,
#     #     nodesplitter=wds.split_by_node,  # ✅ full dataset per node
#     #     handler=wds.handlers.warn_and_continue,
#     #     shardshuffle=0,
#     #     resampled=False,
#     # )
#     # ds = ds.compose(lambda src: safe_shuffle(src, bufsize=10, initial=10, rng=rng))
#     # ds = ds.map(decode_sample).batched(batch_size, partial=True)

# #   File "/home/mirshad7/HunyuanWorld-Mirror/training/data/datasets/test_wds_loader.py", line 124, in build_test_wds_loader
# #     shards_glob_list = glob(f'{shards_glob}/*.tar')
# # TypeError: 'module' object is not callable

#     # shards_glob_list = glob(f'{shards_glob}/*.tar')
#     # print("Found shards:", shards_glob_list)


#     shards_glob_list = glob.glob(f'{shards_glob}/*.tar')
#     print("Found shards:", shards_glob_list)
#     ds = wds.WebDataset(
#         shards_glob_list,
#         nodesplitter=wds.split_by_node,  # ✅ full dataset per node
#         handler=wds.handlers.warn_and_continue,
#         shardshuffle=False,
#         resampled=False,
#     ).shuffle(5, initial=5)

#     # ds = ds.compose(lambda src: safe_shuffle(src, bufsize=10, initial=10, rng=rng))
#     ds = ds.map(decode_sample).batched(batch_size, partial=True)

#     # shards = list(wds.shardlists.expand_urls(shards_glob))
#     # shards = shards * max(1, num_workers)   # duplicate to avoid contention
#     # ds = (wds.WebDataset(
#     #     shards,
#     #     nodesplitter=wds.split_by_worker,
#     #     handler=wds.handlers.warn_and_continue,
#     #     shardshuffle=0,
#     #     resampled=False,
#     # )
#     # .compose(lambda src: safe_shuffle(src, bufsize=1000, initial=100, rng=rng))
#     # .map(decode_sample)
#     # .batched(batch_size, partial=True)
#     # )

#     loader = wds.WebLoader(
#         ds,
#         num_workers=num_workers,
#         batch_size=None,
#         pin_memory=False,
#         prefetch_factor=(2 if num_workers > 0 else None),
#         persistent_workers=False,
#     )
#     return loader

# def make_dataset(shards_glob):
#     return wds.WebDataset(
#         shards_glob,
#         handler=wds.handlers.warn_and_continue,
#         resampled=False,
#     ).map(decode_sample)

# def build_test_wds_loader(shards_glob, num_workers=0, batch_size=4, epoch=0):
#     def worker_init_fn(worker_id):
#         info = torch.utils.data.get_worker_info()
#         num_workers = info.num_workers
#         worker_id = info.id

#         shards = list(wds.shardlists.expand_urls(shards_glob))
#         per_worker = len(shards) // num_workers
#         shards_split = shards[worker_id * per_worker : (worker_id + 1) * per_worker]

#         print(f"[Worker {worker_id}] assigned shards: {shards_split}")

#         info.dataset.pipeline[0].urls = shards_split

#     ds = make_dataset(shards_glob)
#     ds = ds.compose(lambda src: safe_shuffle(src, bufsize=10, initial=10, rng=np.random.default_rng(epoch)))
#     ds = ds.batched(batch_size, partial=True)

#     loader = wds.WebLoader(
#         ds,
#         num_workers=num_workers,
#         worker_init_fn=worker_init_fn,
#         batch_size=None,
#         pin_memory=False,
#         prefetch_factor=(2 if num_workers > 0 else None),
#         persistent_workers=False,
#     )
#     return loader

# ----------------------------
# 🚀 Test loop
# ----------------------------
if __name__ == "__main__":
    # shards = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_test/sope-{000000..000003}.tar"

    root = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset"

    for epoch in range(3):  # few epochs
        print(f"\n=== Epoch {epoch} ===")
        loader = build_sope_wds_loader(root, num_workers=2, batch_size=20)

        # unique_keys = set()
        # num_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            rgb = batch["rgb"]
            # key = batch["key"]
            # if num_batches ==0:
            #     print(f"First batch rgb shape: {rgb.shape}")
            #     print("key:", key)
            # num_batches += 1
            # unique_keys.update(key)

            # x = batch["dummy_tensor"]
            # Simulate forward pass
            # _ = x.mean()
        # print(f"✅ Unique samples seen in epoch {epoch}: {len(unique_keys)}")
        # print(f"✅ Finished epoch {epoch} with {num_batches} batches")
