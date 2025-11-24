# """
# Base dataset utilities: VGGT-style resize (width≈518, keep aspect ratio)
# and consistent transforms for RGB / depth / heatmap / pose_map / intrinsics.
# """
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BaseCentersnapDataset(Dataset):
    def __init__(self, root, img_width=640, patch_size=14, norm_rgb=True):
        self.root = root
        self.img_width = img_width
        self.patch = patch_size
        self.norm_rgb = norm_rgb

    def updated_intrinsics_after_crop(
        self, K: np.ndarray,
        crop_top: int, crop_left: int,
    ) -> np.ndarray:
        """Update intrinsics K after a crop operation.

        :param K: Original 3x3 intrinsics matrix.
        :param orig_h: Original image height.
        :param orig_w: Original image width.
        :param crop_top: Top pixel of the crop box.
        :param crop_left: Left pixel of the crop box.
        :param crop_h: Height of the crop box.
        :param crop_w: Width of the crop box.

        :return: Updated 3x3 intrinsics matrix after cropping.
        """
        K_new = K.copy()
        # Adjust principal point
        K_new[0, 2] -= crop_left
        K_new[1, 2] -= crop_top
        return K_new

    # ------------------------------------------------------------------
    # @staticmethod
    # def _pad_to_patch_multiple(arr, patch=32, pad_val=0):
    #     """Pad arr (H,W[,C]) so both dims are multiples of patch."""
    #     if arr is None:
    #         return None
    #     H, W = arr.shape[:2]
    #     pad_h = (patch - (H % patch)) % patch
    #     pad_w = (patch - (W % patch)) % patch
    #     pad_cfg = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (arr.ndim - 2)
    #     return np.pad(arr, pad_cfg, mode="constant", constant_values=pad_val)

    # ------------------------------------------------------------------
    def resize_like_vggt(self, img, depth=None, heat=None, pose=None, K=None):
        """
        Resize so that WIDTH≈518 (multiple of patch), preserve aspect ratio.
        Pad to patch multiple.  Update intrinsics accordingly.
        """
        H, W = img.shape[:2]
        scale = self.img_width / W
        # print("scaling factor:", scale)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        # print("new size:", (new_h, new_w))
        if not new_h == H:
            # need to resize
            # print("Resizing from", (H, W), "to", (new_h, new_w))
            img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            depth_r = (
                cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if depth is not None
                else None
            )
            heat_r = (
                cv2.resize(heat, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if heat is not None
                else None
            )
            pose_r = (
                cv2.resize(pose, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if pose is not None
                else None
            )

            if K is not None:
                K = K.copy()
                K[0, 0] *= scale
                K[1, 1] *= scale
                K[0, 2] *= scale
                K[1, 2] *= scale

        else:
            img_r = img
            depth_r = depth
            heat_r = heat
            pose_r = pose
                
        #remember pose_r has shape (H/2,W/2,12)

        # if self.cropping_strategy == "center":

        #employing a center cropping to height of 480 if the height is 512, only pplicable for SOPE dataset synthetic


        # --- rigid center crop if height != 480 ---
        target_h = 480
        if img_r.shape[0] != target_h:
            # full-res crop
            start_h = (img_r.shape[0] - target_h) // 2
            end_h = start_h + target_h

            img_r = img_r[start_h:end_h]
            if depth_r is not None:
                depth_r = depth_r[start_h:end_h]
            if heat_r is not None:
                heat_r = heat_r[start_h:end_h]

            # half-res crop for pose
            if pose_r is not None:
                start_h_p = start_h // 2
                end_h_p = end_h // 2
                pose_r = pose_r[start_h_p:end_h_p]

            if K is not None:

                K = self.updated_intrinsics_after_crop(
                    K,
                    crop_top=start_h,
                    crop_left=0,
                )
        
        # print("final resized shape", img_r.shape, depth_r.shape, heat_r.shape, pose_r.shape)

        # # --- pad to patch multiple ---
        # img_r = self._pad_to_patch_multiple(img_r, self.patch, pad_val=0)
        # depth_r = self._pad_to_patch_multiple(depth_r, self.patch, pad_val=0)
        # heat_r = self._pad_to_patch_multiple(heat_r, self.patch, pad_val=0)
        # pose_r = self._pad_to_patch_multiple(pose_r, self.patch, pad_val=0)

        # --- update intrinsics ---

        return img_r, depth_r, heat_r, pose_r, K
