import numpy as np
from scipy.stats import multivariate_normal


def save_targets(prefix, heatmap, abs_pose):
    heatmap_path = prefix + "heatmap.npz"
    pose_path = prefix + "pose_map.npz"

    # Save as .npz files (keyed arrays)
    np.savez_compressed(heatmap_path, heatmap=heatmap)
    np.savez_compressed(pose_path, abs_pose=abs_pose)

_PEAK_CONCENTRATION = 0.8

_DOWNSCALE_VALUE = 2


def compute_nocs_abspose_field(poses, heat_maps, bbox_side_lengths=None):
    """Exact CenterSnap version using SOPE Pose.to_affine()."""
    abs_pose_target = np.zeros(
        [len(poses), heat_maps[0].shape[0], heat_maps[0].shape[1], 12], dtype=np.float32
    )
    # print("abs_pose_target", abs_pose_target.shape)
    heatmap_indices = np.argmax(np.array(heat_maps), axis=0)

    for pose, ii in zip(poses, range(len(heat_maps))):
        mask = (heatmap_indices == ii)
        affine = pose.to_affine()
        R_gt = affine[:3, :3]
        t_gt = affine[:3, 3]
        s_gt = np.array(bbox_side_lengths[ii])

        rot6d_gt = R_gt[:, :2].reshape(-1)  # first two columns flattened → (6,)
        abs_pose_values = np.concatenate([rot6d_gt, t_gt, s_gt])
        abs_pose_target[ii, mask] = abs_pose_values

    return np.sum(abs_pose_target, axis=0)[::_DOWNSCALE_VALUE, ::_DOWNSCALE_VALUE].copy()

# ---------------------------------------------------------------------
# CenterSnap-style heatmap computation
# ---------------------------------------------------------------------

def compute_heatmaps_from_masks(masks, meta_objects=None, intrinsics=None):
    """
    Compute heatmaps for multiple objects.

    Parameters
    ----------
    masks : list[np.ndarray]
        List of (H, W) binary masks.
    meta_objects : Optional[list]
        List of metadata entries (same order as masks).
    intrinsics : Optional[np.ndarray]
        3x3 camera intrinsics.

    Returns
    -------
    heatmaps : list[np.ndarray]
        List of per-object heatmaps (each (H, W)).
    """
    heatmaps = []
    for i, mask in enumerate(masks):
        obj_meta = meta_objects[i] if meta_objects is not None else None
        hm = compute_heatmap_from_mask(mask, obj_meta=obj_meta, intrinsics=intrinsics)
        heatmaps.append(hm)
    return heatmaps

def get_intrinsic_matrix(cam):
    """Convert a SOPE CameraIntrinsicsBase object to a 3x3 numpy array."""
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K


def resize_intrinsics(K, new_h, new_w, orig_h, orig_w):
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    K_resized = K.copy()
    K_resized[0, 0] *= scale_x  # fx
    K_resized[1, 1] *= scale_y  # fy
    K_resized[0, 2] *= scale_x  # cx
    K_resized[1, 2] *= scale_y  # cy
    return K_resized



def compute_heatmap_from_mask(mask, obj_meta=None, intrinsics=None):
    """
    Compute a CenterSnap-style Gaussian heatmap for a single object mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary (H, W) array for the object (uint8 or bool).
    obj_meta : Optional[object]
        SOPE object metadata entry, must have `.translation` (3,) in camera frame.
        Used to compute a better geometric center.
    intrinsics : Optional[np.ndarray]
        3x3 camera intrinsics matrix.

    Returns
    -------
    heat_map : np.ndarray
        (H, W) float32 heatmap normalized to [0, 1].
    """

    H, W = mask.shape
    n_pix = np.count_nonzero(mask)
    if n_pix < 8:
        # skip tiny or invalid masks
        print("Warning: mask too small, returning zero heatmap.")
        return np.zeros((H, W), dtype=np.float32)

    # --- Compute 2D centroid of the visible mask (pixel mean) ---
    coords = np.column_stack(np.nonzero(mask))
    mask_center = np.floor(np.mean(coords, axis=0))  # (y, x)

    # --- Optionally project 3D object center ---
    if obj_meta is not None and intrinsics is not None:
        # project 3D translation to pixel space

        # if hasattr(intrinsics, "fx"):
        #     K = get_intrinsic_matrix(intrinsics)
        # else:
        #     K = np.array(intrinsics, dtype=np.float32).reshape(3, 3)

        if hasattr(intrinsics, "fx"):
            K = get_intrinsic_matrix(intrinsics)
            orig_w, orig_h = intrinsics.width, intrinsics.height
        else:
            K = np.array(intrinsics, dtype=np.float32).reshape(3, 3)
            orig_w, orig_h = 1280, 1024  # fallback if not given

        # scale to match current mask shape
        K = resize_intrinsics(K, H, W, orig_h, orig_w)
        
        cam_pt = np.array(obj_meta.translation, dtype=np.float32).reshape(3)
        px = K @ cam_pt

        if px[2] > 1e-6:
            px = (px[:2] / px[2])[::-1]  # (y, x)
            mean_value = 0.5 * (mask_center + px)
        else:
            mean_value = mask_center

    # --- Compute covariance (2x2) from pixel distribution ---
    cov = np.cov((coords - mean_value).T)
    if not np.isfinite(cov).all() or np.linalg.det(cov) <= 1e-8:
        # handle degenerate covariance
        cov = np.eye(2, dtype=np.float32) * 4.0
    cov *= _PEAK_CONCENTRATION

    # --- Build Gaussian density over pixel grid ---
    yy, xx = np.mgrid[0:H, 0:W]
    pos = np.stack([yy, xx], axis=-1).reshape(-1, 2)
    multi_var = multivariate_normal(mean=mean_value, cov=cov, allow_singular=True)
    density = multi_var.pdf(pos)
    heat_map = np.zeros((H, W), dtype=np.float32)
    heat_map[yy.ravel(), xx.ravel()] = density

    # --- Normalize to [0, 1] ---
    maxv = np.max(heat_map)
    if maxv > 0:
        heat_map /= maxv
    return heat_map