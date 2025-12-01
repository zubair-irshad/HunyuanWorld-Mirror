import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import numpy as np
import cv2
import wandb
from tqdm import tqdm

from models.models.centersnap_foundation_pose import WorldMirrorCenterSnap
from training.data.datasets.sope_dataloader import DynamicCenterSnapLoader
from training.data.datasets.sope_dataset import SopeCentersnapDataset
from cutoop.utils import draw_3d_bbox, draw_pose_axes
from training.data.datasets.utils import extract_peaks_from_centroid, extract_abs_pose_from_peaks, draw_peaks
from cutoop.data_types import CameraIntrinsicsBase

from training.losses.loss import compute_loss

from models.utils.priors import normalize_depth


# ------------------------------------------------------------------
#  Helper: visualization for wandb
# ------------------------------------------------------------------
@torch.no_grad()
def make_visualizations(batch, preds, draw_gt = False, save_dir=None):
    """Builds RGB, depth, overlay heatmap, pose, and peaks visualization grid."""

    rgb = batch["rgb"].detach().cpu()
    depth = batch["depth"].detach().cpu()
    intrinsics = batch["K"].detach().cpu()
    valid_mask = batch["valid_mask"].cpu()

    print("rgb, depth, intrinsics, valid_mask shapes:", rgb.shape, depth.shape, intrinsics.shape, valid_mask.shape)

    if draw_gt:
        pose_pred = batch["pose_map"].detach().to(torch.float32).cpu()  # <-- cast here
        heat_pred = batch["heatmap"].detach().to(torch.float32).cpu()   # <-- cast here

        print("GT")
        print("heatmap, pose shapes:", heat_pred.shape, pose_pred.shape)

    else:
        pose_pred = preds["pose_map"].detach().to(torch.float32).cpu()  # <-- cast here
        heat_pred = preds["heatmap"].detach().to(torch.float32).cpu()   # <-- cast here
        heat_pred = heat_pred.squeeze(1)          # [B,H,W]
        pose_pred = pose_pred.squeeze(1)        # [B,12,H,W]
        print("Predictions")
        print("heatmap, pose shapes:", heat_pred.shape, pose_pred.shape)

    B = rgb.shape[0]

    colored_depth_list, colored_heatmap_list, overlay_list, pose_vis, peaks_vis = [], [], [], [], []

    for idx in range(B):
        rgb_np = rgb[idx].permute(1, 2, 0).numpy().clip(0, 1)
        d = depth[idx, 0].numpy()
        d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
        d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # heatmap color
        h_pred = heat_pred[idx, 0].numpy().clip(0, 1)
        h_color = cv2.applyColorMap((h_pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = cv2.addWeighted(rgb_np, 0.6, h_color, 0.4, 0)

        # peaks extraction & drawing
        heatmap_peaks = torch.from_numpy(h_pred)
        peaks = extract_peaks_from_centroid(heatmap_peaks, min_distance=10, min_confidence=0.2)
        peaks_img = draw_peaks(h_pred, peaks)
        peaks_vis.append(torch.from_numpy(peaks_img).permute(2, 0, 1))

        # abs_pose_output = pose_pred[idx].permute(1, 2, 0).numpy()
        print("pose pred shape:", pose_pred.shape)  # <-- debug print
        abs_pose_output = pose_pred[idx].permute(1, 2, 0).numpy().astype(np.float32)
        abs_poses, sizes = extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=4)

        intr = CameraIntrinsicsBase(
            fx=intrinsics[idx, 0, 0].item(),
            fy=intrinsics[idx, 1, 1].item(),
            cx=intrinsics[idx, 0, 2].item(),
            cy=intrinsics[idx, 1, 2].item(),
            width=rgb_np.shape[1],
            height=rgb_np.shape[0],
        )
        vis_img = rgb_np.copy()
        for T, size in zip(abs_poses, sizes):
            vis_img = draw_3d_bbox(vis_img, intrinsics=intr, sRT_4x4=T, bbox_side_len=size)
            vis_img = draw_pose_axes(vis_img, intrinsics=intr, sRT_4x4=T, length=0.1)

        colored_depth_list.append(torch.from_numpy(d_color).permute(2, 0, 1))
        colored_heatmap_list.append(torch.from_numpy(h_color).permute(2, 0, 1))
        overlay_list.append(torch.from_numpy(overlay).permute(2, 0, 1))
        pose_vis.append(torch.from_numpy(vis_img).permute(2, 0, 1))

    grid = torchvision.utils.make_grid(
        torch.cat(
            [
                rgb,
                torch.stack(colored_depth_list),
                valid_mask.float().repeat(1, 3, 1, 1),
                torch.stack(overlay_list),
                torch.stack(pose_vis),
                torch.stack(peaks_vis),
            ],
            dim=0,
        ),
        nrow=B,
        padding=2,
        normalize=True,
        value_range=(0, 1),
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(save_dir, "train_vis.png"))
    return grid


# ------------------------------------------------------------------
#  Training script
# ------------------------------------------------------------------
def train():
    wandb.init(project="centersnap-sope", name="wm_centersnap_v1")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and loader
    root = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE"
    train_ds = SopeCentersnapDataset(root=root, compute_targets=True)
    loader_builder = DynamicCenterSnapLoader(train_ds, num_workers=6, batch_size=8)
    train_loader = loader_builder.get_loader(epoch=0)

    # Model
    model = WorldMirrorCenterSnap(
        img_size=518,
        patch_size=14,
        embed_dim=512,
        use_depth_condition=True,
    ).to(device)

    enc_params, dec_params, enc_trainable, dec_trainable = model.param_counts()
    wandb.config.update(
        {"encoder_params": enc_params, "decoder_params": dec_params, "total_params": enc_params + dec_params}
    )
    print(f"Encoder: {enc_params/1e6:.2f}M | Decoders: {dec_params/1e6:.2f}M")
    print(f"Trainable Encoder: {enc_trainable/1e6:.2f}M | Trainable Decoders: {dec_trainable/1e6:.2f}M")

    num_training_steps = len(train_loader) * 20


    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=2e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-8)
    scaler = GradScaler(enabled=True)

    # Training loop
    model.train()
    global_step = 0
    for epoch in range(50):
        loader = loader_builder.get_loader(epoch)
        epoch_loss = 0
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape} | min: {v.min().item():.4f}, max: {v.max().item():.4f}")

    
            depth = normalize_depth(depth)

            print(f"  depth after norm: {depth.shape} | min: {depth.min().item():.4f}, max: {depth.max().item():.4f}")

            with autocast(dtype=torch.bfloat16):
                preds = model(rgb, depth)

                print("Predictions:")
                for k,v in preds.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape} | min: {v.min().item():.4f}, max: {v.max().item():.4f}")
                print("===============================================================================\n\n")

                loss, loss_dict = compute_loss(preds, batch)
                loss_heat = 100* loss_dict["heatmap_loss"]
                loss_pose = loss_dict["pose_loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/loss_heat": loss_heat,
                    "train/loss_pose": loss_pose,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

            epoch_loss += loss.item()
            global_step += 1

            if global_step ==1 or global_step % 250 == 0:
                grid_gt = make_visualizations(batch, preds, draw_gt = True)
                wandb.log({"train/visualizations_gt": wandb.Image(grid_gt)}, step=global_step)

                grid_pred = make_visualizations(batch, preds, draw_gt = False)
                wandb.log({"train/visualizations_pred": wandb.Image(grid_pred)}, step=global_step)

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss}, step=global_step)

        # save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"checkpoints/centersnap_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}, ckpt_path)
            wandb.save(ckpt_path)

    print("Training completed!")


if __name__ == "__main__":
    train()
