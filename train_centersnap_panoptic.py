import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import numpy as np
import cv2
import wandb
from tqdm import tqdm
from functools import partial
import torch.distributed as dist
import gc
# from torch.amp import autocast, GradScaler
from training.data.datasets.webdataloader_utils import build_sope_wds_loader
from cutoop.utils import draw_3d_bbox, draw_pose_axes
from training.data.datasets.utils import extract_peaks_from_centroid, extract_abs_pose_from_peaks, draw_peaks
from cutoop.data_types import CameraIntrinsicsBase
from training.losses.loss import compute_loss
from models.utils.priors import normalize_depth, normalize_depth_fixed

# ---- CenterSnap imports ----
from models.models.centersnap.panoptic_backbone import build_resnet_fpn_backbone, ShapeSpec, output_shape, SemSegFPNHead, PoseFPNHead
from models.models.centersnap.basic_stem import RGBDStem


# ==========================================================
# HParams — mirrors CenterSnap CLI args
# ==========================================================
class HParams:
    def __init__(self):
        # --- model ---
        self.model_name = "res_fpn"
        self.model_norm = "GN"
        self.num_filters_scale = 4

        # --- optimization ---
        self.optim_learning_rate = 0.0006
        self.optim_momentum = 0.9
        self.optim_weight_decay = 1e-4
        self.optim_poly_exp = 0.9
        self.optim_warmup_epochs = 1

        # --- losses ---
        self.loss_seg_mult = 1.0
        self.loss_depth_mult = 1.0
        self.loss_vertex_mult = 0.1
        self.loss_rotation_mult = 0.1
        self.loss_heatmap_mult = 1000.0
        self.loss_latent_emb_mult = 0.1
        self.loss_abs_pose_mult = 0.1
        self.loss_z_centroid_mult = 0.1

        self.batch_size = 32
        self.num_workers = 3

        # self.shards = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset/sope-{000000..000850}.tar"
        # self.test_shards = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset_test/sope-{000000..000099}.tar"

        self.shards = "/mnt/ssd/SOPE_webdataset"
        self.test_shards = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE_webdataset_test"
        self.no_pin_memory = False
        self.persistent_workers = False

# ==========================================================
# CenterSnap Panoptic network (raw tensor outputs)
# ==========================================================
class CenterSnapPanopticRaw(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Backbone
        input_shape = ShapeSpec(channels=3, height=512, width=640)
        self.stem = RGBDStem(hparams)
        self.backbone = build_resnet_fpn_backbone(
            input_shape,
            self.stem,
            model_norm=hparams.model_norm,
            num_filters_scale=hparams.num_filters_scale,
        )
        shape = output_shape(self.backbone)

        # Heads
        self.heatmap_head = SemSegFPNHead(shape, num_classes=1, model_norm=hparams.model_norm)
        self.pose_head = PoseFPNHead(shape, num_classes=12, model_norm=hparams.model_norm)

    def forward(self, rgb, depth):
        """Return raw tensors (no wrapped outputs)."""
        x = torch.cat([rgb, depth], dim=1)
        features, _ = self.backbone.forward(x)
        # small_disp_output = small_disp_output.squeeze(1)

        # depth_pred = self.depth_head.forward(features)
        heatmap_pred = self.heatmap_head.forward(features)
        pose_pred = self.pose_head.forward(features)

        return {
            # "depth": depth_pred,
            # "small_depth": small_disp_output,
            "heatmap": heatmap_pred,
            "pose_map": pose_pred,
        }


# ==========================================================
# Visualization (unchanged)
# ==========================================================
@torch.no_grad()
def make_visualizations(batch, preds, draw_gt=False, save_dir=None):
    rgb_vis = batch["rgb"].detach().cpu()
    depth = batch["depth"].detach().cpu()
    intrinsics = batch["K"].detach().cpu()
    # valid_mask = batch["valid_mask"].cpu()

        #     # ---- Apply ImageNet normalization (CenterSnap-style) ----
        # mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        # std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        # rgb = (rgb - mean) / std

    #can I unnormalize RGB for visualization?
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    rgb_vis = rgb_vis * std + mean

    if draw_gt:
        pose_pred = batch["pose_map"].detach().to(torch.float32).cpu()
        heat_pred = batch["heatmap"].detach().to(torch.float32).cpu()
        print("GT, rgb, depth, pose_pred, heat_pred shapes:", rgb_vis.shape, depth.shape, pose_pred.shape, heat_pred.shape)
    else:
        pose_pred = preds["pose_map"].detach().to(torch.float32).cpu()
        heat_pred = preds["heatmap"].detach().to(torch.float32).cpu()
        heat_pred = heat_pred
        pose_pred = pose_pred.squeeze(1)

        print("Pred, rgb, depth, pose_pred, heat_pred shapes:", rgb_vis.shape, depth.shape, pose_pred.shape, heat_pred.shape
)
    #only visualize first 4 images

    B = min(4, rgb_vis.shape[0])
    # B = rgb.shape[0]
    colored_depth_list, colored_heatmap_list, overlay_list, pose_vis, peaks_vis = [], [], [], [], []
    colored_rgb_list = []
    # valid_mask_list = []
    for idx in range(B):
        rgb_np = rgb_vis[idx].permute(1, 2, 0).numpy().clip(0, 1)
        d = depth[idx, 0].numpy()
        d_norm = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-8)
        d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h_pred = heat_pred[idx, 0].numpy().clip(0, 1)
        h_color = cv2.applyColorMap((h_pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = cv2.addWeighted(rgb_np, 0.6, h_color, 0.4, 0)

        heatmap_peaks = torch.from_numpy(h_pred)
        peaks = extract_peaks_from_centroid(heatmap_peaks, min_distance=10, min_confidence=0.2)
        peaks_img = draw_peaks(h_pred, peaks)
        peaks_vis.append(torch.from_numpy(peaks_img).permute(2, 0, 1))

        abs_pose_output = pose_pred[idx].permute(1, 2, 0).numpy().astype(np.float32)
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


        bbox_task = []
        # Visualize predicted poses
        for j, (T, size) in enumerate(zip(abs_poses, sizes)):
            bbox_task.append(
                partial(
                    draw_3d_bbox,
                    intrinsics=intr,
                    sRT_4x4=T,
                    bbox_side_len=size,
                )
            )
            bbox_task.append(
                partial(
                    draw_pose_axes,
                    intrinsics=intr,
                    sRT_4x4=T,
                    length=0.1,
                )
            )
        print("len bbox_task", len(bbox_task))

        # vis_img = color.copy()
        for filter in bbox_task:
            vis_img = filter(img=vis_img)
            
        # for T, size in zip(abs_poses, sizes):
        #     vis_img = draw_3d_bbox(vis_img, intrinsics=intr, sRT_4x4=T, bbox_side_len=size)
        #     vis_img = draw_pose_axes(vis_img, intrinsics=intr, sRT_4x4=T, length=0.1)

        colored_depth_list.append(torch.from_numpy(d_color).permute(2, 0, 1))
        colored_heatmap_list.append(torch.from_numpy(h_color).permute(2, 0, 1))
        overlay_list.append(torch.from_numpy(overlay).permute(2, 0, 1))
        pose_vis.append(torch.from_numpy(vis_img).permute(2, 0, 1))
        colored_rgb_list.append(torch.from_numpy(rgb_np).permute(2, 0, 1))
        # valid_mask_list.append(valid_mask[idx])

    grid = torchvision.utils.make_grid(
        torch.cat(
            [
                torch.stack(colored_rgb_list),
                torch.stack(colored_depth_list),
                # valid_mask.float().repeat(1, 3, 1, 1),
                # torch.stack(valid_mask_list).float().repeat(1, 3, 1, 1),
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


# ==========================================================
# Training loop
# ==========================================================

import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)


def train():
    hparams = HParams()
    # Determine distributed rank (if using DDP). DataLoader num_workers != distributed ranks.
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Initialize wandb only on rank 0; disable on others to avoid hanging.
    if rank == 0:
        wandb.init(project="centersnap-panoptic", name="centersnap_raw_v2")
    else:
        # Disable wandb for non-zero ranks to prevent concurrent logging issues.
        wandb.init(project="centersnap-panoptic", name="centersnap_raw_v2_worker{rank}", mode="disabled")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batches_per_epoch = 367780//hparams.batch_size  # or your computed number
    # batches_per_epoch = int(batches_per_epoch/hparams.num_workers)

    # test_batches_per_epoch = 40820//hparams.batch_size
    # test_batches_per_epoch = int(test_batches_per_epoch/hparams.num_workers)

    model = CenterSnapPanopticRaw(hparams).to(device)
    print("total params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    print("trainable params (M):", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    print("backbone, heatmap_head, pose_head trainable params (M):", sum(p.numel() for p in model.backbone.parameters() if p.requires_grad) / 1e6, sum(p.numel() for p in model.heatmap_head.parameters() if p.requires_grad) / 1e6, sum(p.numel() for p in model.pose_head.parameters() if p.requires_grad) / 1e6)
    
    optimizer = AdamW(model.parameters(), lr=hparams.optim_learning_rate, weight_decay=hparams.optim_weight_decay)
    # num_training_steps = len_trainset * 10
    total_training_steps = 11000 * 30
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps, eta_min=1e-8)
    global_step = 0

    # scaler = GradScaler()
    log_interval = 100  # how often to log
    running_loss = 0.0
    running_dict = {
        "heatmap_loss": 0.0,
        "abs_rot_loss": 0.0,
        "tran_size_loss": 0.0,
        "pose_loss": 0.0,
    }

    # print("len train loader:", len(train_loader))
    for epoch in range(30):
        train_loader = build_sope_wds_loader(
            shards_glob=hparams.shards,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            bufsize=2000,
            initial=500,
            epoch=epoch
        )

        # loader = loader_builder.get_loader(epoch)
        epoch_loss = 0.0
        model.train()   

        # Iterate until natural exhaustion (all shards consumed)
        # pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=batches_per_epoch)
        # for batch in pbar:
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            depth = normalize_depth_fixed(batch["depth"])
            rgb = batch["rgb"].to(device, non_blocking=True)
            
            depth = depth.to(device, non_blocking=True)


            # with autocast(device_type="cuda"):
            #     preds = model(rgb, depth)
            #     loss, loss_dict = compute_loss(preds, batch)

            preds = model(rgb, depth)
            loss, loss_dict = compute_loss(preds, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()    
            scheduler.step()

            # optimizer.zero_grad(set_to_none=True)
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
            if (global_step + 1) % log_interval == 0 and rank == 0:
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
                    
            # if rank == 0:  # Only rank 0 logs scalars to wandb
            #     wandb.log(
            #         {
            #             "train/loss": loss.item(),
            #             "train/loss_heat": loss_dict["heatmap_loss"],
            #             "train/loss_pose": loss_dict["pose_loss"],
            #             "train/loss_rot": loss_dict["rot_loss"],
            #             "train/loss_trans": loss_dict["trans_loss"],
            #             "train/loss_size": loss_dict["size_loss"],
            #             "lr": optimizer.param_groups[0]["lr"],
            #         },
            #         step=global_step,
            #     )

            # epoch_loss += loss.item()
            # global_step += 1
            # num_batches += 1
                
            # Visualization only on rank 0 to avoid multi-process contention.
            if rank == 0 and global_step % 8000 == 0:
                with torch.no_grad():
                    grid_gt = make_visualizations(batch, preds, draw_gt=True)
                    wandb.log({"train/visualizations_gt": wandb.Image(grid_gt)}, step=global_step)
                    grid_pred = make_visualizations(batch, preds, draw_gt=False)
                    wandb.log({"train/visualizations_pred": wandb.Image(grid_pred)}, step=global_step)

                    del grid_gt, grid_pred
        
        # avg_loss = epoch_loss / num_batches

        avg_epoch_loss = (epoch_loss / num_batches)
        if torch.is_tensor(avg_epoch_loss):
            avg_epoch_loss = avg_epoch_loss.item()
            
        print(f"Epoch {epoch}: avg_loss={avg_epoch_loss:.4f} steps={num_batches}")
        if rank == 0:
            wandb.log({"epoch_loss": avg_epoch_loss}, step=global_step)

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

                if num_test_batches%800 == 0:
                    #Visualization only on rank 0 to avoid multi-process contention.
                    if rank == 0:
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
                if rank == 0:
                    wandb.log({f"test/{k}": test_loss_each[k]}, step=global_step)
            print(f"Epoch {epoch}: avg_test_loss={avg_test_loss:.4f}")
            if rank == 0:
                wandb.log({"test/epoch_loss": avg_test_loss}, step=global_step)
                
            # End of test epoch
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()

        if rank == 0 and (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/centersnap_raw_epoch_{epoch+1}.pt"
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()},
                ckpt_path,
            )
            wandb.save(ckpt_path)

    print("Training completed!")


if __name__ == "__main__":
    train()
