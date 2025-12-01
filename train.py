import os
import gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.amp import autocast, GradScaler
import torchvision
import numpy as np
import cv2
import wandb
from tqdm import tqdm

# Dataloader
from training.data.datasets.webdataloader_utils import build_sope_wds_loader

# Model
from models.models.centersnap_foundation_pose import WorldMirrorCenterSnap

# Utils
from training.data.datasets.utils import (
    extract_peaks_from_centroid,
    extract_abs_pose_from_peaks,
    draw_peaks,
)
from cutoop.utils import draw_3d_bbox, draw_pose_axes
from cutoop.data_types import CameraIntrinsicsBase
from training.losses.loss import compute_loss
from models.utils.priors import normalize_depth_fixed


# ==========================================================
# HParams used for WDS loader
# ==========================================================
class HParams:
    def __init__(self):
        self.batch_size = 20
        self.num_workers = 3

        # paths to WDS shards
        # self.shards = "/mnt/ssd/SOPE_webdataset"
        # self.test_shards = "/mnt/ssd/SOPE_webdataset_test"

        self.shards = "/mnt/ssd/SOPE_webdataset"
        self.test_shards = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset_test"

        # training settings
        self.num_epochs = 20
        self.depth_norm = True


# ==========================================================
# Visualization
# ==========================================================
@torch.no_grad()
def make_visualizations(batch, preds, draw_gt=False, save_dir=None):
    rgb = batch["rgb"].cpu()                 # [B,3,H,W]
    depth = batch["depth"].cpu()             # [B,1,H,W]
    intrinsics = batch["K"].cpu()

    if draw_gt:
        heat_pred = batch["heatmap"].cpu()
        pose_pred = batch["pose_map"].cpu()
    else:
        heat_pred = preds["heatmap"].detach().cpu()  # [B,1,H,W]
        pose_pred = preds["pose_map"].detach().cpu()  # [B,12,H,W]
        heat_pred = heat_pred.squeeze(1)  # [B,H,W]
        pose_pred = pose_pred.squeeze(1)  # [B,12,H,W]
    
    # print("rgb, depth, heat_pred, pose_pred shapes:", rgb.shape, depth.shape, heat_pred.shape, pose_pred.shape)
    B = min(4, rgb.shape[0])

    colored_depth_list = []
    colored_heatmap_list = []
    overlay_list = []
    pose_vis = []
    peaks_vis = []

    for idx in range(B):
        rgb_np = rgb[idx].permute(1, 2, 0).numpy().clip(0, 1)

        # --- depth ---
        d = depth[idx, 0].numpy()
        d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
        d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # --- heatmap ---
        # h_pred = heat_pred[idx, 0].numpy().clip(0, 1)
        h_pred = heat_pred[idx, 0].to(torch.float32).cpu().numpy()
        h_color = cv2.applyColorMap((h_pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = cv2.addWeighted(rgb_np, 0.6, h_color, 0.4, 0)

        # --- peaks ---
        peaks = extract_peaks_from_centroid(torch.from_numpy(h_pred), 10, 0.2)
        peaks_img = draw_peaks(h_pred, peaks)
        peaks_vis.append(torch.from_numpy(peaks_img).permute(2, 0, 1))

        # --- pose extraction ---
        # abs_pose_output = pose_pred[idx].permute(1, 2, 0).numpy().astype(np.float32)
        abs_pose_output = pose_pred[idx].permute(1, 2, 0).to(torch.float32).cpu().numpy()
        abs_poses, sizes = extract_abs_pose_from_peaks(peaks, abs_pose_output, scale_factor=2)

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
            vis_img = draw_3d_bbox(vis_img, intr, T, size)
            vis_img = draw_pose_axes(vis_img, intr, T, length=0.1)

        colored_depth_list.append(torch.from_numpy(d_color).permute(2, 0, 1))
        colored_heatmap_list.append(torch.from_numpy(h_color).permute(2, 0, 1))
        overlay_list.append(torch.from_numpy(overlay).permute(2, 0, 1))
        pose_vis.append(torch.from_numpy(vis_img).permute(2, 0, 1))

    # final grid
    grid = torchvision.utils.make_grid(
        torch.cat(
            [
                rgb[:B],
                torch.stack(colored_depth_list),
                torch.stack(overlay_list),
                torch.stack(pose_vis),
                torch.stack(peaks_vis)
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
        torchvision.utils.save_image(grid, f"{save_dir}/vis.png")

    return grid


# ==========================================================
# Training
# ==========================================================
def train():
    hparams = HParams()
    wandb.init(project="centersnap-sope-transformer", name="wm_centersnap_wds_v1")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Model
    # -------------------------
    model = WorldMirrorCenterSnap(
        img_size=384,
        patch_size=16,
        embed_dim=384,
        patch_embed="dinov3_vits16",
        use_depth_condition=True,
    ).to(device)

    print("model param counts (M):")
    enc_params, dec_params, enc_trainable, dec_trainable = model.param_counts()
    print(f" Encoder:  {enc_params/1e6:.2f}M total | {enc_trainable/1e6:.2f}M trainable")
    print(f" Decoders: {dec_params/1e6:.2f}M total | {dec_trainable/1e6:.2f}M trainable")

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    # scaler = GradScaler(enabled=True)

    total_training_steps = hparams.num_epochs * 18000  # rough estimate
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=1e-8)

    global_step = 0
    log_interval = 100  # how often to log
    running_loss = 0.0
    running_dict = {
        "heatmap_loss": 0.0,
        "abs_rot_loss": 0.0,
        "tran_size_loss": 0.0,
        "pose_loss": 0.0,
    }
    # ==========================================================
    # Epoch Loop
    # ==========================================================
    for epoch in range(hparams.num_epochs):

        # -----------------------------------------
        # Build new epoch loader (WDS style)
        # -----------------------------------------
        train_loader = build_sope_wds_loader(
            shards_glob=hparams.shards,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            bufsize=2000,
            initial=500,
            epoch=epoch,
            normalize_rgb=False,   # RAW RGB — IMPORTANT
        )

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            rgb = batch["rgb"].to(device)  # [B,3,H,W]
            depth = normalize_depth_fixed(batch["depth"]).to(device)

            optimizer.zero_grad(set_to_none=True)

            # with autocast(dtype=torch.bfloat16, device_type='cuda'):
            # with autocast(dtype=torch.float16, device_type='cuda'):
            preds = model(rgb, depth)  # transformer takes [B,S,3,H,W]
            loss, loss_dict = compute_loss(preds, batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            # ---- Mixed precision ----
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()

           # ---- Accumulate for averaging ----
            running_loss += loss.detach()
            for k in running_dict.keys():
                if k in loss_dict:
                    running_dict[k] += loss_dict[k]

            epoch_loss += loss.detach()
                    
            num_batches += 1
            global_step += 1
            
            # ---- Log every N steps ----
            if (global_step + 1) % log_interval == 0:
                avg_loss = (running_loss / log_interval)
                avg_dict = {k: (v / log_interval) for k, v in running_dict.items()}

                wandb.log(
                    {
                        "train/total_loss": avg_loss,
                        "train/heatmap_loss": avg_dict["heatmap_loss"],
                        "train/rot_loss": avg_dict["abs_rot_loss"],
                        "train/tran_size_loss": avg_dict["tran_size_loss"],
                        "train/pose_loss": avg_dict["pose_loss"],
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
                # reset accumulators
                running_loss = 0.0
                for k in running_dict.keys():
                    running_dict[k] = 0.0
            # wandb.log(
            #     {
            #         "train/loss": loss.item(),
            #         "train/heatmap_loss": loss_dict["heatmap_loss"],
            #         "train/pose_loss": loss_dict["pose_loss"],
            #         "lr": optimizer.param_groups[0]["lr"],
            #     },
            #     step=global_step,
            # )

            # epoch_loss += loss.item()
            # global_step += 1

            # --- Visualization ---
            if global_step == 1 or global_step % 14000 == 0:
                grid_gt = make_visualizations(batch, preds, draw_gt=True)
                wandb.log({"train/vis_gt": wandb.Image(grid_gt)}, step=global_step)

                grid_pred = make_visualizations(batch, preds, draw_gt=False)
                wandb.log({"train/vis_pred": wandb.Image(grid_pred)}, step=global_step)
                del grid_gt, grid_pred
        
        avg_loss = (epoch_loss / num_batches)
        wandb.log({"epoch_loss": avg_loss}, step=global_step)
        print(f"Epoch {epoch}: avg_loss {avg_loss:.4f}")

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

        # ==========================================================
        # Test loop (added)
        # ==========================================================
        model.eval()
        with torch.no_grad():
            test_loader = build_sope_wds_loader(
                shards_glob=hparams.test_shards,
                batch_size=hparams.batch_size,
                num_workers=hparams.num_workers,
                bufsize=500,
                initial=100
            )

            test_epoch_loss = 0
            # test_loss_each = {"heatmap_loss": 0, "pose_loss": 0, "rot_loss": 0, "trans_loss": 0, "size_loss": 0}
            test_loss_each = {"heatmap_loss": 0, "abs_rot_loss": 0, "tran_size_loss": 0, "pose_loss": 0}
            # pbar_test = tqdm(test_loader, desc=f"Test Epoch {epoch}")
            # for batch in pbar_test:
            num_test_batches = 0
            for batch in tqdm(test_loader, desc=f"Test Epoch {epoch}"):
                depth = normalize_depth_fixed(batch["depth"])
                rgb = batch["rgb"].to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                preds = model(rgb, depth)
                loss, loss_dict = compute_loss(preds, batch)
                test_epoch_loss += loss.detach()
                num_test_batches += 1

                if num_test_batches%1200 == 0:
                    #Visualization only on rank 0 to avoid multi-process contention.
                    grid_gt_test = make_visualizations(batch, preds, draw_gt=True)
                    wandb.log({"test/visualizations_gt": wandb.Image(grid_gt_test)}, step=global_step)
                    grid_pred_test = make_visualizations(batch, preds, draw_gt=False)
                    wandb.log({"test/visualizations_pred": wandb.Image(grid_pred_test)}, step=global_step)
                    del grid_gt_test, grid_pred_test

                #compute all different losses rot, trans, size, heatmap, pose
                #add them all up and visualize at end of testing epoch
                for k in test_loss_each.keys():
                    test_loss_each[k] += loss_dict[k]


            avg_test_loss = (test_epoch_loss / num_test_batches)
            if torch.is_tensor(avg_test_loss):
                avg_test_loss = avg_test_loss.item()
            for k in test_loss_each.keys():
                test_loss_each[k] /= num_test_batches
                wandb.log({f"test/{k}": test_loss_each[k]}, step=global_step)
            print(f"Epoch {epoch}: avg_test_loss={avg_test_loss:.4f}")
            wandb.log({"test/epoch_loss": avg_test_loss}, step=global_step)
                
            # End of test epoch
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

        # save checkpoint
        if (epoch + 1) % 2 == 0:
            os.makedirs("checkpoints_transformer", exist_ok=True)
            path = f"checkpoints_transformer/transformer_epoch_{epoch+1}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)
            wandb.save(path)

    print("Training complete!")


if __name__ == "__main__":
    train()
