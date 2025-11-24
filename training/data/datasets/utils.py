import numpy as np
import torch
import cv2
from cutoop.utils import draw_3d_bbox, draw_pose_axes
import torchvision
from cutoop.data_types import CameraIntrinsicsBase
from functools import partial
import os

def updated_intrinsics_after_crop(
    K: np.ndarray,
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

def resize_like_vggt(img, depth=None, heat=None, pose=None, K=None):
    """
    Resize so that WIDTH≈518 (multiple of patch), preserve aspect ratio.
    Pad to patch multiple.  Update intrinsics accordingly.
    """
    img_r = img
    depth_r = depth
    heat_r = heat
    pose_r = pose

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

            K = updated_intrinsics_after_crop(
                K,
                crop_top=start_h,
                crop_left=0,
            )

    return img_r, depth_r, heat_r, pose_r, K

def get_intrinsic_matrix(cam, image_width=None, image_height=None):
    """Convert a SOPE CameraIntrinsicsBase object to a 3x3 numpy array."""
    width, height = cam.width, cam.height
    # print("Image size in Intrinsics:", width, "x", height)
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)

    if image_width is not None and image_height is not None:
        # Adjust K if the image size is different from the original
        scale_x = image_width / width
        scale_y = image_height / height
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        # print("Adjusted Intrinsics for image size:", image_width, "x", image_height)
    return K

import numpy as np
from skimage.feature import peak_local_max

# def extract_peaks_from_centroid(centroid_heatmap, min_distance=10, min_confidence=0.20):
#   peaks = peak_local_max(centroid_heatmap, min_distance=min_distance, threshold_abs=min_confidence)
#   peaks = peaks[peaks[:,1].argsort()]
#   return peaks

def draw_peaks(centroid_target, peaks):
  centroid_target = np.clip(centroid_target, 0.0, 1.0) * 255.0
  heatmap_img = cv2.applyColorMap(centroid_target.astype(np.uint8), cv2.COLORMAP_JET)
  for ii in range(peaks.shape[0]):
    point = (int(peaks[ii, 1]), int(peaks[ii, 0]))
    heatmap_img = cv2.putText(heatmap_img,str(ii), 
    point, 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255,255,255),
    2)
    cv2.line(heatmap_img, (point), ([0,0]), (0, 255, 0), thickness=3, lineType=8)
  return heatmap_img

def extract_peaks_from_centroid(centroid_heatmap, min_distance=10, min_confidence=0.20):
    """
    Find local maxima (object centers) in a heatmap tensor.
    Automatically handles PyTorch tensors on GPU or CPU.
    """
    # --- Ensure input is a NumPy array on CPU ---
    if isinstance(centroid_heatmap, torch.Tensor):
        centroid_heatmap = centroid_heatmap.detach().cpu().squeeze().numpy()

    # --- Compute peaks ---
    peaks = peak_local_max(
        centroid_heatmap,
        min_distance=min_distance,
        threshold_abs=min_confidence
    )

    # --- Sort peaks by X (optional, for consistency) ---
    if peaks.size > 0:
        peaks = peaks[peaks[:, 1].argsort()]

    return peaks


def extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=4):
    """
    Extract absolute 4x4 poses and object sizes at detected heatmap peaks.
    Compatible with compute_nocs_abspose_field() which stores [rot6d, trans, size].

    Args:
        peaks: (N, 2) array of [y, x] pixel coordinates from centroid heatmap.
        abs_pose_output: (H, W, 12) numpy array containing 6D rot + 3D trans + 3D size.
        scale_factor: downsampling factor between heatmap and abs_pose_output resolution.

    Returns:
        abs_poses: list of 4x4 numpy arrays (object poses)
        sizes: list of (3,) numpy arrays (bbox side lengths)
    """
    assert abs_pose_output.shape[-1] == 12, \
        f"Expected 12 channels (6 rot + 3 trans + 3 size), got {abs_pose_output.shape[-1]}"

    abs_poses, sizes = [], []

    for ii in range(peaks.shape[0]):
        # Map heatmap coords → abs_pose_output resolution
        # v = int(peaks[i, 0] / scale_factor)
        # u = int(peaks[i, 1] / scale_factor)
        # v = np.clip(v, 0, abs_pose_output.shape[0] - 1)
        # u = np.clip(u, 0, abs_pose_output.shape[1] - 1)

        # Extract per-pixel pose vector
        # values = abs_pose_output[v, u, :]

        index = np.zeros([2])
        # index[0] = int(peaks[ii, 0])
        # index[1] = int(peaks[ii, 1])
        
        #incorporate scale factor
        index[0] = int(peaks[ii, 0] / scale_factor)
        index[1] = int(peaks[ii, 1] / scale_factor)

        # print("index 0 and 1", index[0], index[1])
        # print("abs_pose_output shape", abs_pose_output.shape)
        # index = index.astype(np.int)
        index = index.astype(np.int64)
        values = abs_pose_output[index[0], index[1], :]
        
        values = values.cpu().numpy() if isinstance(values, torch.Tensor) else values

        # --- Split into rotation(6), translation(3), and size(3) ---
        rot6d = values[0:6].reshape(3, 2)  # (3,2)
        t = values[6:9]
        s = values[9:12]


        # --- Reconstruct valid rotation via Gram–Schmidt ---
        a1 = rot6d[:, 0]
        a2 = rot6d[:, 1]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 /= (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        R = np.stack((b1, b2, b3), axis=1)  # (3,3)

        # --- Compose 4x4 affine transformation ---
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        abs_poses.append(T)
        sizes.append(s)

    return abs_poses, sizes


def vis_one_batch(batch, i):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = batch["rgb"].to(device)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(device)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(device)
    rgb_vis = rgb * std + mean
    
    depth = batch["depth"].to(device)
    print(f"Batch {i}: RGB {rgb.shape}, Depth {depth.shape}")

    for k,v in batch.items():
        print(f"  {k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")

    #save RGB, depth, mask images for verification

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
        return  # only save first 5 batches
