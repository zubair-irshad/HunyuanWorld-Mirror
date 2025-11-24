"""
SOPE → CenterSnap dataset with VGGT-style resizing (width≈518, keep AR).
Preserves alignment of RGB / depth / heatmap / pose_map / intrinsics.
"""
import os
import cv2
import torch
import numpy as np
from cutoop.data_loader import Dataset as SopeDataset
from cutoop.data_types import Pose
from .base_dataset import BaseCentersnapDataset
from .utils import (
    get_intrinsic_matrix,
)
from preprocessing.utils import (
    compute_heatmaps_from_masks,
    compute_nocs_abspose_field,
    save_targets,
)
from tqdm import tqdm

class SopeCentersnapDataset():
    def __init__(self, root):
        self.prefixes = SopeDataset.glob_prefix(root, mode="test")
    def __len__(self):
        return len(self.prefixes)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        print("processing:", prefix)

        mask = SopeDataset.load_mask(prefix + "mask.exr")
        meta = SopeDataset.load_meta(prefix + "meta.json")

        # --- Build mask list ---
        mask_ids = np.unique(mask)  # skip background (0)
        mask_ids = mask_ids[mask_ids != 0]

        # print(f"Mask unique IDs: {np.unique(mask)}")
        masks_list = [(mask == mid).astype(np.uint8) for mid in mask_ids]

        # --- Build mask_id → metadata mapping ---
        meta_dict = {obj.mask_id: obj for obj in meta.objects}
        meta_list = [meta_dict.get(mid, None) for mid in mask_ids]

        # --- Poses (still useful for abs_pose_target) ---
        poses = [Pose(tuple(o.quaternion_wxyz), tuple(o.translation)) for o in meta.objects]

        bbox_side_lengths = [o.meta.bbox_side_len for o in meta.objects]

        # --- Intrinsics ---
        intrinsics = meta.camera.intrinsics

        # --- Compute heatmaps safely (matching mask_id ↔ object) ---
        heatmaps = compute_heatmaps_from_masks(
            masks_list, meta_objects=meta_list, intrinsics=intrinsics
        )
        combined_heatmap = np.max(np.stack(heatmaps, axis=0), axis=0)

        # --- Compute absolute pose field (still uses all objects) ---
        abs_pose_target = compute_nocs_abspose_field(poses, heatmaps, bbox_side_lengths)

        print(
            "computed heatmap and abs_pose_target shapes:",
            combined_heatmap.shape,
            abs_pose_target.shape,
        )

        save_targets(
            prefix,
            combined_heatmap.astype(np.float32),
            abs_pose_target.astype(np.float32),
        )

#faster multiprocessing

dataset = None

def init_worker(global_root):
    """Initialize dataset once per process (keeps memory local)."""
    global dataset
    dataset = SopeCentersnapDataset(global_root)


# ---------------------------------------------------------
# Per-item processing function
# ---------------------------------------------------------
def process_idx(idx):
    prefix = dataset.prefixes[idx]
    # print("processing:", prefix)
    heatmap_path = prefix + "heatmap.npz"
    pose_path = prefix + "pose_map.npz"

    # Skip if already exists
    # if os.path.exists(heatmap_path) and os.path.exists(pose_path):
    #     return 0

    try:
        dataset[idx]  # triggers computation + saving
        return 1
    except Exception as e:
        print(f"[WARN] Failed {prefix}: {e}")
        return 0


from multiprocessing import Pool, cpu_count
# ---------------------------------------------------------
# Main precomputation
# ---------------------------------------------------------
def precompute_all(root):
    dataset = SopeCentersnapDataset(root)
    total = len(dataset)
    # nproc = min(cpu_count(), 16)  # tune as needed
    nproc = 32

    print(f"Starting with {nproc} workers over {total} items...")
    with Pool(nproc, initializer=init_worker, initargs=(root,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_idx, range(total)), total=total))

    n_done = sum(results)
    print(f"\n✅ Done. Processed {n_done} / {total} scenes.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    precompute_all("/home/mirshad7/Downloads/data/Omni6DPose/SOPE")


# def precompute_all(root):
#     dataset = SopeCentersnapDataset(root)
#     for i in tqdm(range(len(dataset))):
#         prefix = dataset.prefixes[i]
#         print("processing:", prefix)

        


#         # heatmap_path = prefix + "heatmap.npz"
#         # pose_path = prefix + "pose_map.npz"
#         # if not (os.path.exists(heatmap_path) and os.path.exists(pose_path)):
#         #     sample = dataset[i]  # triggers computation + saving

#         #delete npz npy files 
#         for ext in ["heatmap.npz", "pose_map.npz", "heatmap.npy", "pose_map.npy", "heatmap.npz.npy", "pose_map.npz.npy"]:
#             path = prefix + ext
#             if os.path.exists(path):
#                 os.remove(path)
#                 print("deleted:", path)

# def precompute_specific(root, prefix = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE/train/scene_00001/frame_00000000_"):
#     dataset = SopeCentersnapDataset(root)
#     print("dataset.prefixes[0]:", dataset.prefixes[0])
#     if prefix in dataset.prefixes:
#         idx = dataset.prefixes.index(prefix)
#         print("processing:", prefix)
#         dataset[idx]  # triggers computation + saving
#     else:
#         print("Prefix not found in dataset.")
    
    # dataset = SopeCentersnapDataset(root)
    # for i in range(len(dataset)):
    #     prefix = dataset.prefixes[i]  # however you form your prefix
    #     pose_path = prefix + "pose_map.npz"

    #     if os.path.exists(pose_path):
    #         if os.path.getsize(pose_path) == 0:
    #             print("Empty file found:", pose_path)

    #     heatmap_path = prefix + "heatmap.npz"
    #     if os.path.exists(heatmap_path):
    #         if os.path.getsize(heatmap_path) == 0:
    #             print("Empty file found:", heatmap_path)
                
        

# if __name__ == "__main__":
#     # benchmark_loader("/home/mirshad7/Downloads/data/Omni6DPose/SOPE")
#     # precompute_all("/home/mirshad7/Downloads/data/Omni6DPose/SOPE")
#     precompute_specific("/home/mirshad7/Downloads/data/Omni6DPose/SOPE", prefix= '/home/mirshad7/Downloads/data/Omni6DPose/SOPE/39/train/matterport3d/0050/0017_')