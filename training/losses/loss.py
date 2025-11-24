import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Masked L1 for pose map (no confidence)
# ---------------------------------------------------------
# ---------------------------------------------------------
# Masked L1 for pose map (no confidence)
# ---------------------------------------------------------
class MaskedL1Loss(nn.Module):
    def __init__(self, centroid_threshold=0.3, pose_weight=10.0):
        super().__init__()
        self.centroid_threshold = centroid_threshold
        self.pose_weight = pose_weight
        self.loss = nn.L1Loss(reduction="none")
        # self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target, heatmap_gt, valid_mask=None, downsample_factor=2):
        """
        pred: [B,C,H,W]  (pose prediction)
        target: [B,C,H,W]
        heatmap_gt: [B,H,W]
        valid_mask: [B,1,H,W] or None
        """
        B, C, H, W = pred.shape

        # --- Make sure masks & targets align ---
        if valid_mask is None:
            valid_mask = torch.ones((B, 1, H, W), device=pred.device, dtype=pred.dtype)
        elif valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]

        if heatmap_gt.ndim == 4:
            heatmap_gt = heatmap_gt.squeeze(1)  # [B,H,W]

        if target.ndim == 5:
            target = target.squeeze(1)  # remove seq dim if exists
        if target.shape[1] != C:
            # Permute if target is [B,H,W,C]
            target = target.permute(0, 3, 1, 2).contiguous()

        valid_mask = valid_mask.float()
        heat_mask = (heatmap_gt > self.centroid_threshold).float()
        valid_mask = valid_mask[:, :, ::downsample_factor, ::downsample_factor]  # downsample if needed
        heat_mask = heat_mask[:, ::downsample_factor, ::downsample_factor]  # downsample if needed
        combined_mask = valid_mask.squeeze(1) * heat_mask  # [B,H,W]
        mask = combined_mask.unsqueeze(1).expand_as(pred)  # [B,C,H,W]

        # Assuming pred has shape [B, 12, H, W] and target has same shape
        # rot_pred, trans_pred, size_pred = pred[:, :6], pred[:, 6:9], pred[:, 9:12]
        # rot_gt, trans_gt, size_gt = target[:, :6], target[:, 6:9], target[:, 9:12]

        # Define helper to compute masked average loss
        def masked_loss(pred, target, mask):
            loss = self.loss(pred, target)
            loss = loss * mask
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom

        # Compute each loss component
        rot_loss = masked_loss(pred[:, :6], target[:, :6], mask[:, :6])
        trans_loss = masked_loss(pred[:, 6:9], target[:, 6:9], mask[:, 6:9])
        size_loss = masked_loss(pred[:, 9:12], target[:, 9:12], mask[:, 9:12])

        # Optionally combine them (e.g., equal weighting)
        total_loss = rot_loss + trans_loss + size_loss

        # For logging
        log_dict = {
            'rot_loss': self.pose_weight * rot_loss.item(),
            'trans_loss': self.pose_weight * trans_loss.item(),
            'size_loss': self.pose_weight * size_loss.item(),
            'pose_loss': self.pose_weight * total_loss.item()
        }

        return total_loss, log_dict


        # loss = self.loss(pred, target)
        # loss = loss * mask
        # denom = mask.sum().clamp(min=1.0)
        # return loss.sum() / denom


# ---------------------------------------------------------
# Main loss computation
# ---------------------------------------------------------
def compute_loss(preds, batch, heat_weight=100.0, pose_weight=10.0, centroid_threshold=0.3):
    """
    Simple loss without confidence weighting.
    - Heatmap: MSE
    - Pose: Masked L1 (only where valid + within object region)
    """
    device = preds["heatmap"].device

    # -------------------------------
    # Unpack predictions
    # -------------------------------
    heat_pred = preds["heatmap"]      # [B,S,C,H,W] or [B,S,H,W,1]
    pose_pred = preds["pose_map"]     # [B,S,C,H,W] or [B,S,H,W,C]

    # Remove sequence dim if present
    if heat_pred.ndim == 5:
        heat_pred = heat_pred.squeeze(1)
    if pose_pred.ndim == 5:
        pose_pred = pose_pred.squeeze(1)

    # Permute if needed
    if heat_pred.shape[1] == 1 and heat_pred.ndim == 4:
        heat_pred = heat_pred.squeeze(1)  # [B,H,W]
    elif heat_pred.ndim == 4 and heat_pred.shape[1] > 1:
        heat_pred = heat_pred[:, 0]       # take first channel

    if pose_pred.ndim == 4 and pose_pred.shape[1] != 12:
        # If shape [B,H,W,12], permute to [B,12,H,W]
        pose_pred = pose_pred.permute(0, 3, 1, 2).contiguous()

    # -------------------------------
    # Ground truth
    # -------------------------------
    heatmap_gt = batch["heatmap"].to(device)   # [B,1,H,W]
    pose_gt = batch["pose_map"].to(device)     # [B,12,H,W]
    valid_mask = batch.get("valid_mask", None)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    # Align dims
    heatmap_gt = heatmap_gt.squeeze(1)  # [B,H,W]
    if pose_gt.ndim == 5:
        pose_gt = pose_gt.squeeze(1)
    if pose_gt.shape[1] != 12:
        pose_gt = pose_gt.permute(0, 3, 1, 2).contiguous()

    # -------------------------------
    # Heatmap loss (MSE)
    # -------------------------------
    if valid_mask is None:
        valid_mask = torch.ones((heatmap_gt.shape[0], 1, *heatmap_gt.shape[1:]), device=device)

    heatmap_loss = F.mse_loss(
        heat_pred * valid_mask.squeeze(1),
        heatmap_gt * valid_mask.squeeze(1),
    )

    # -------------------------------
    # Pose masked L1 loss
    # -------------------------------
    mask_l1 = MaskedL1Loss(centroid_threshold=centroid_threshold, pose_weight=pose_weight)

    # print("pose_pred shape:", pose_pred.shape, pose_gt.shape, heatmap_gt.shape, valid_mask.shape)  # Debug print
    pose_loss, log_dict = mask_l1(pose_pred, pose_gt, heatmap_gt, valid_mask)

    # -------------------------------
    # Combine
    # -------------------------------
    total_loss = heat_weight * heatmap_loss + pose_weight * pose_loss

    return total_loss, {
        "heatmap_loss": heat_weight * float(heatmap_loss.detach()),
        "total_loss": float(total_loss.detach()),
        **log_dict
    }


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MaskedL1Loss(nn.Module):
#     def __init__(self, centroid_threshold=0.3):
#         super().__init__()
#         self.loss = nn.L1Loss(reduction='none')
#         self.centroid_threshold = centroid_threshold

#     def forward(self, output, target, heatmap_mask, valid_mask):
#         """
#         output: [B,C,H,W]
#         target: [B,C,H,W]
#         heatmap_mask: [B,H,W]
#         valid_mask: [B,1,H,W]
#         """
#         valid_mask = valid_mask.squeeze(1).float()  # [B,H,W]
#         heat_mask = (heatmap_mask > self.centroid_threshold).float()  # [B,H,W]
#         combined_mask = valid_mask * heat_mask  # [B,H,W]

#         mask = combined_mask.unsqueeze(1).expand_as(output)  # [B,C,H,W]

#         loss = self.loss(output, target)
#         loss = loss * mask
#         valid_count = mask.sum().clamp(min=1.0)
#         return loss.sum() / valid_count


# def compute_loss(preds, batch):
#     """
#     Compute total loss for CenterSnap-like model:
#     - Heatmap: MSE (weighted x100)
#     - Pose: Masked L1 over all 12 channels
#     - Both respect valid_mask from pad_collate
#     """
#     device = preds["heatmap"].device

#     # Fix prediction layout ----------------------------------------------------
#     heat_pred = preds["heatmap"]
#     pose_pred = preds["pose_map"]

#     # Handle shapes like [B,1,H,W,1] or [B,1,H,W,12]
#     if heat_pred.ndim == 5:  # [B,1,H,W,1]
#         heat_pred = heat_pred.squeeze(1).squeeze(-1)  # → [B,H,W]
#     elif heat_pred.ndim == 4 and heat_pred.shape[1] == 1:
#         heat_pred = heat_pred.squeeze(1)  # → [B,H,W]

#     if pose_pred.ndim == 5:  # [B,1,H,W,12]
#         pose_pred = pose_pred.squeeze(1).permute(0, 3, 1, 2).contiguous()  # → [B,12,H,W]
#     elif pose_pred.ndim == 4 and pose_pred.shape[1] == 1:
#         pose_pred = pose_pred.squeeze(1).permute(0, 3, 1, 2).contiguous()  # fallback

#     # -------------------------------------------------------------------------

#     heatmap_gt = batch["heatmap"].to(device).squeeze(1)  # [B,H,W]
#     pose_gt = batch["pose_map"].to(device)               # [B,12,H,W]
#     valid_mask = batch["valid_mask"].to(device)          # [B,1,H,W]

#     # --- Heatmap loss ---
#     # Resize mask if necessary (e.g. model predicts smaller HxW)
#     if valid_mask.shape[-2:] != heat_pred.shape[-2:]:
#         valid_mask = F.interpolate(valid_mask.float(), size=heat_pred.shape[-2:], mode="nearest")

#     heatmap_loss = F.mse_loss(
#         heat_pred * valid_mask.squeeze(1),
#         heatmap_gt * valid_mask.squeeze(1),
#     )

#     # --- Pose loss ---
#     mask_l1 = MaskedL1Loss(centroid_threshold=0.3)
#     pose_loss = mask_l1(pose_pred, pose_gt, heatmap_gt, valid_mask)

#     # --- Total loss (weight heatmap by 100x) ---
#     total_loss = 100.0 * heatmap_loss + pose_loss

#     return total_loss, {
#         "heatmap_loss": heatmap_loss.item(),
#         "pose_loss": pose_loss.item(),
#         "total_loss": total_loss.item(),
#     }
