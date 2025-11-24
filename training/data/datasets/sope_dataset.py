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

import sys
sys.path.append("/home/mirshad7/HunyuanWorld-Mirror/training/data/datasets")

from base_dataset import BaseCentersnapDataset
from utils import (
    get_intrinsic_matrix,
)

# from .base_dataset import BaseCentersnapDataset
# from .utils import (
#     get_intrinsic_matrix,
# )

import time

class SopeCentersnapDataset(BaseCentersnapDataset):
    def __init__(self, root, img_width=640, patch_size=32, norm_rgb=True):
        super().__init__(root, img_width, patch_size, norm_rgb=norm_rgb)
        self.prefixes = SopeDataset.glob_prefix(root, mode="train")

        #shuffle prefixes for randomness
        # np.random.seed(42)
        # np.random.shuffle(self.prefixes)

        self.norm_rgb = norm_rgb

    def __len__(self):
        return len(self.prefixes)

    # ------------------------------------------------------------------
    def load_targets(self, prefix):
        heat_path, pose_path = prefix + "heatmap.npz", prefix + "pose_map.npz"
        heat = np.load(heat_path)["heatmap"]
        pose = np.load(pose_path)["abs_pose"]
        return heat, pose

    # def __getitem__(self, idx):
    #     prefix = self.prefixes[idx]
    #     timings = {}
    #     t0 = time.time()

    #     # ---- Load raw data ----
    #     t = time.time()
    #     color = SopeDataset.load_color(prefix + "color.png")
    #     depth = SopeDataset.load_depth(prefix + "depth.exr")
    #     mask  = SopeDataset.load_mask(prefix + "mask.exr")
    #     meta  = SopeDataset.load_meta(prefix + "meta.json")
    #     timings["load_images_json"] = time.time() - t

    #     # ---- Intrinsics ----
    #     t = time.time()
    #     image_width, image_height = color.shape[1], color.shape[0]
    #     K = get_intrinsic_matrix(meta.camera.intrinsics, image_width, image_height)
    #     timings["intrinsics"] = time.time() - t

    #     # ---- Load targets ----
    #     t = time.time()
    #     heat, pose = self.load_targets(prefix)
    #     timings["npz_load"] = time.time() - t

    #     # ---- Resize ----
    #     t = time.time()
    #     color, depth, heat, pose, K = self.resize_like_vggt(
    #         color, depth=depth, heat=heat, pose=pose, K=K
    #     )
    #     timings["resize"] = time.time() - t

    #     # ---- Tensor conversion ----
    #     t = time.time()
    #     sample = {
    #         "rgb": torch.from_numpy(color.copy()).permute(2, 0, 1).float() / 255.0,
    #         "depth": torch.from_numpy(depth.copy()).unsqueeze(0).float(),
    #         "K": torch.from_numpy(K).float(),
    #         "prefix": prefix,
    #     }
    #     if heat is not None and pose is not None:
    #         sample["heatmap"] = torch.from_numpy(heat.copy()).unsqueeze(0).float()
    #         sample["pose_map"] = torch.from_numpy(pose.copy()).permute(2, 0, 1).float()
    #     timings["tensor_conversion"] = time.time() - t

    #     timings["total"] = time.time() - t0
    #     sample["timings"] = timings
    #     return sample


    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        # print("self.prefixes[idx]:", self.prefixes[idx])
        # print("Loading sample:", prefix)
        color = SopeDataset.load_color(prefix + "color.png")
        depth = SopeDataset.load_depth(prefix + "depth.exr")
        # mask = SopeDataset.load_mask(prefix + "mask.exr")
        meta = SopeDataset.load_meta(prefix + "meta.json")

        image_width, image_height = color.shape[1], color.shape[0]
        K = get_intrinsic_matrix(meta.camera.intrinsics, image_width, image_height)

        # print("color shape:", color.shape)
        # print("depth shape:", depth.shape)
        # print("Intrinsics K:\n", K)

        heat, pose = self.load_targets(prefix)

        print("heat shape before resize:", heat.shape, "pose shape before resize:", pose.shape)
        # --- VGGT-style resize / pad (keeps aspect ratio) ---
        color, depth, heat, pose, K = self.resize_like_vggt(
            color, depth=depth, heat=heat, pose=pose, K=K
        )

        # print("color shape after resize:", color.shape)

        # ---- Convert to tensors ----
        rgb = torch.from_numpy(color.copy()).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth.copy()).unsqueeze(0).float()

        # ---- Apply ImageNet normalization (CenterSnap-style) ----
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        rgb = (rgb - mean) / std

        sample = {
            "rgb": rgb,
            "depth": depth,
            "K": torch.from_numpy(K).float(),
            "prefix": prefix,
        }

        if heat is not None and pose is not None:
            sample["heatmap"] = torch.from_numpy(heat.copy()).unsqueeze(0).float()
            sample["pose_map"] = torch.from_numpy(pose.copy()).permute(2, 0, 1).float()

        return sample


# ----------------------------------------------------------------------
# Quick smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np

    root = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE"
    ds = SopeCentersnapDataset(root, norm_rgb=True)

    N = min(len(ds), 200)
    times = {k: [] for k in ["load_images_json", "npz_load", "resize", "tensor_conversion", "total"]}

    for i in tqdm(range(N)):
        t = ds[i]["timings"]
        for k in times: times[k].append(t[k])

    for k,v in times.items():
        v = np.array(v)
        print(f"{k:20s} mean={v.mean():.4f}s std={v.std():.4f}s max={v.max():.4f}s")
