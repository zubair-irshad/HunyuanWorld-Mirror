"""
DynamicCenterSnapLoader
-----------------------
VGGT-style dynamic DataLoader for variable image heights (same width≈518).
Batches images of differing H by zero-padding them on the fly.
"""
import random
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import cv2
from cutoop.utils import draw_3d_bbox, draw_pose_axes
import cv2
from .utils import extract_peaks_from_centroid, extract_abs_pose_from_peaks, draw_peaks
from cutoop.data_types import CameraIntrinsicsBase

# def pad_collate(batch):
#     """
#     Pad variable-sized CenterSnap samples to the max HxW in the batch.
#     Returns a dict of stacked tensors plus 'valid_mask' for loss masking.
#     """
#     # ---- find max height/width in batch ----
#     max_h = max(b["rgb"].shape[1] for b in batch)
#     max_w = max(b["rgb"].shape[2] for b in batch)

#     # ---- helper: pad one tensor to max size ----
#     def pad_to_max(t: torch.Tensor, pad_val=0.0):
#         """Pad tensor [C,H,W] to [C,max_h,max_w]."""
#         pad_h = max_h - t.shape[1]
#         pad_w = max_w - t.shape[2]
#         if pad_h == 0 and pad_w == 0:
#             return t
#         return F.pad(t, (0, pad_w, 0, pad_h), value=pad_val)

#     collated = {}
#     # ---- valid-pixel masks (1 where real, 0 where padded) ----
#     masks = []
#     for b in batch:
#         H, W = b["rgb"].shape[1:]
#         mask = torch.zeros((1, max_h, max_w), dtype=torch.bool)
#         mask[:, :H, :W] = True
#         masks.append(mask)
#     collated["valid_mask"] = torch.stack(masks)        # [B,1,Hmax,Wmax]

#     # ---- pad & stack each key ----
#     for key in batch[0].keys():
#         if key in ["prefix"]:                          # non-tensor metadata
#             collated[key] = [b[key] for b in batch]
#             continue

#         # modalities needing spatial padding
#         if key in ["rgb", "depth", "heatmap", "pose_map"]:
#             collated[key] = torch.stack([pad_to_max(b[key]) for b in batch])
#             continue

#         # intrinsics or other same-shape tensors
#         if isinstance(batch[0][key], torch.Tensor):
#             collated[key] = torch.stack([b[key] for b in batch])
#         else:
#             collated[key] = [b[key] for b in batch]

#     return collated

# Can bring it back later

# # ---------------------------------------------------------
# #  Dynamic Sampler (similar spirit to VGGT DynamicBatchSampler)
# # ---------------------------------------------------------
# class DynamicAspectSampler(Sampler):
#     """
#     Samples dataset indices while dynamically selecting random aspect ratios.
#     This doesn’t change number of images per sample (always 1), but emulates
#     VGGT dynamic aspect logic for single-view data.
#     """

#     def __init__(self, dataset, aspect_range=(0.33, 1.0), seed=42):
#         self.dataset = dataset
#         self.aspect_range = aspect_range
#         self.seed = seed
#         self.rng = random.Random(seed)

#     def set_epoch(self, epoch):
#         self.rng.seed(self.seed + epoch)

#     def __iter__(self):
#         n = len(self.dataset)
#         order = list(range(n))
#         self.rng.shuffle(order)
#         for idx in order:
#             # sample random aspect ratio (not actually used in dataset, placeholder)
#             aspect_ratio = round(self.rng.uniform(*self.aspect_range), 2)
#             yield (idx, aspect_ratio)

#     def __len__(self):
#         return len(self.dataset)


# ---------------------------------------------------------
#  DynamicCenterSnapLoader class
# ---------------------------------------------------------
class DynamicCenterSnapLoader:
    """
    Builds a DataLoader that can handle variable-height inputs (VGGT style).
    """
    def __init__(
        self,
        dataset,
        num_workers=2,
        batch_size=4,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        aspect_range=(0.33, 1.0),
        seed=42,
        prefetch_factor=10,
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
        self.aspect_range = aspect_range
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        # self.sampler = DynamicAspectSampler(dataset, aspect_range, seed)


    def get_loader(self, epoch=0):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        sampler = RandomSampler(self.dataset, generator=g)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            # collate_fn=pad_collate
        )

# ---------------------------------------------------------
#  Quick smoke test
# ---------------------------------------------------------
if __name__ == "__main__":

    #webdataset

    from .sope_dataset import SopeCentersnapDataset
    from tqdm import tqdm
    root = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE"
    ds = SopeCentersnapDataset(root, norm_rgb=True)
    loader_builder = DynamicCenterSnapLoader(ds, num_workers=6, batch_size=10)
    dl = loader_builder.get_loader()

    #time 50 batches
    # import time
    # t0 = time.time()
    # for i, batch in enumerate(tqdm(dl, total=500)):
    #     if i == 500:
    #         break
    # t_total = time.time() - t0
    # print(f"Time for 500 batches: {t_total:.2f} seconds ({t_total/500:.2f} s/batch)")

    # visualize

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for i, batch in enumerate(tqdm(dl)):
        rgb = batch["rgb"].to(device)


        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(device)
        rgb_vis = rgb * std + mean
        
        depth = batch["depth"].to(device)
        print(f"Batch {i}: RGB {rgb.shape}, Depth {depth.shape}")

        for k,v in batch.items():
            print(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")

        #save RGB, depth, mask images for verification
        import torchvision
        import os
        save_dir = "./dynamic_dataloader_test_outputs"
        os.makedirs(save_dir, exist_ok=True)
        #the grid should have 4 images in each column, row should have RGB, depth, mask, heatmap


        heatmap = batch["heatmap"].to(device)
        abs_pose_output = batch["pose_map"].permute(0,2,3,1).to(device)  # [B,H,W,12]

        print("abs_pose_output", abs_pose_output.shape)
        print("heatmap", heatmap.shape)
        intrinsics = batch["K"].to(device)
        

        # --- Convert to numpy arrays ---
        depth_np = depth.squeeze(1).detach().cpu().numpy()
        heatmap_np = heatmap.squeeze(1).detach().cpu().numpy()

        colored_depth_list = []
        colored_heatmap_list = []
        overlay_list = []
        pose_vis = []
        peaks_vis = []
        colored_rgb_list = []

        for idx in range(len(rgb)):
            # --- Depth colormap ---


            d = depth_np[idx]
            d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)  # normalize 0–1
            d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # --- Heatmap colormap ---
            h = np.clip(heatmap_np[idx], 0.0, 1.0) * 255.0
            h_color = cv2.applyColorMap(h.astype(np.uint8), cv2.COLORMAP_JET)
            h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # --- Overlay (RGB + Heatmap) ---
            rgb_np = rgb_vis[idx].detach().cpu().permute(1, 2, 0).numpy()
            rgb_np = np.clip(rgb_np, 0.0, 1.0)

            colored_rgb_list.append(torch.from_numpy(rgb_np).permute(2, 0, 1))


            overlay = cv2.addWeighted(rgb_np, 0.6, h_color, 0.4, 0)

            # --- Stack results ---
            colored_depth_list.append(torch.from_numpy(d_color).permute(2, 0, 1))
            colored_heatmap_list.append(torch.from_numpy(h_color).permute(2, 0, 1))
            overlay_list.append(torch.from_numpy(overlay).permute(2, 0, 1))

            heatmap_peaks = heatmap[idx, 0]

            peaks = extract_peaks_from_centroid(heatmap_peaks, min_distance=10, min_confidence=0.20)

            heatmap_pred = draw_peaks(heatmap_peaks.detach().cpu().numpy(), peaks)
            peaks_vis.append(torch.from_numpy(heatmap_pred).permute(2,0,1))
            # Extract predicted absolute poses
            abs_poses, sizes = extract_abs_pose_from_peaks(peaks, abs_pose_output[idx], scale_factor=2)
            intrinsic_vis = CameraIntrinsicsBase(fx=intrinsics[idx,0,0].item(),
                                                         fy=intrinsics[idx,1,1].item(),
                                                         cx=intrinsics[idx,0,2].item(),
                                                         cy=intrinsics[idx,1,2].item(),
                                                         width=rgb_np.shape[1],
                                                         height=rgb_np.shape[0])
            bbox_task = []
            # Visualize predicted poses
            for j, (T, size) in enumerate(zip(abs_poses, sizes)):
                # print("T", T)
                # print("size", size)
                bbox_task.append(
                    partial(
                        draw_3d_bbox,
                        intrinsics=intrinsic_vis,
                        sRT_4x4=T,
                        bbox_side_len=size,
                    )
                )
                bbox_task.append(
                    partial(
                        draw_pose_axes,
                        intrinsics=intrinsic_vis,
                        sRT_4x4=T,
                        length=0.1,
                    )
                )
            vis_img = rgb_np.copy()
            for filter in bbox_task:
                vis_img = filter(img=vis_img)
                # vis_img = draw_3d_bbox(vis_img, intrinsic_vis, sRT_4x4=T, bbox_side_len=size)
                # vis_img = draw_pose_axes(vis_img, intrinsics=intrinsic_vis, sRT_4x4=T, length=0.1)

            pose_vis.append(torch.from_numpy(vis_img).permute(2,0,1))

        
        rgb = torch.stack(colored_rgb_list).to(device)
        colored_depth = torch.stack(colored_depth_list).to(device)
        colored_heatmap = torch.stack(colored_heatmap_list).to(device)
        overlay_heatmap = torch.stack(overlay_list).to(device)
        pose_vis = torch.stack(pose_vis).to(device)
        peaks_vis = torch.stack(peaks_vis).to(device)

        grid = torchvision.utils.make_grid(
            torch.cat([
                rgb,
                colored_depth,
                # batch["valid_mask"].to(device).float().repeat(1,3,1,1),
                overlay_heatmap, 
                pose_vis,
                peaks_vis
            ], dim=0),
            nrow=4,
            padding=2,
            normalize=True,
            value_range=(0,1)
        )
        torchvision.utils.save_image(grid, os.path.join(save_dir, f"batch_{i}_grid.png"))
        if i == 5:
            break
