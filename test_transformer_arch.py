import torch
import numpy as np
from tqdm import tqdm

# Dataloader
from training.data.datasets.webdataloader_utils import build_sope_wds_loader

# Model
from models.models.centersnap_foundation_pose import WorldMirrorCenterSnap

# Loss
from training.losses.loss import compute_loss

# Utils
from models.utils.priors import normalize_depth_fixed


# ==========================================================
# Minimal HParams for testing
# ==========================================================
class HParams:
    def __init__(self):
        self.batch_size = 4
        self.num_workers = 2
        self.shards = "/mnt/ssd/SOPE_webdataset"          # <--- make sure path is correct
        # self.test_shards = "/mnt/ssd/SOPE_webdataset_test"


# ==========================================================
# Test Script
# ==========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = HParams()

    print("\n============================================")
    print(" Loading ONE batch from WebDataset...")
    print("============================================")

    test_loader = build_sope_wds_loader(
        shards_glob=hparams.shards,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        bufsize=2000,
        initial=200,
        epoch=0,
        normalize_rgb=False,  # Raw RGB, model normalizes itself
    )

    # batch = next(iter(test_loader))

    for batch in tqdm(test_loader, desc="Loading batch", total=1):
        break
    print("Batch keys:", batch.keys())
    print()

    # ------------------------------------------------------
    # Print shapes + stats
    # ------------------------------------------------------
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:<12} {tuple(v.shape)}   min={v.min().item():.4f}   max={v.max().item():.4f}")

    print("\n============================================")
    print(" Initializing Transformer model...")
    print("============================================")

    model = WorldMirrorCenterSnap(
        img_size=512,
        patch_size=16,
        embed_dim=384,
        patch_embed="dinov3_vitb16",
        use_depth_condition=True,
    ).to(device)

    print("Model parameters, backbone + heads: in Millions")
    enc, dec, enc_trainable, dec_trainable = model.param_counts()
    print(f" Encoder:  {enc/1e6:.2f}M total | {enc_trainable/1e6:.2f}M trainable")
    print(f" Decoders: {dec/1e6:.2f}M total | {dec_trainable/1e6:.2f}M trainable")
    # ------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------
    rgb = batch["rgb"].to(device)  # [B,3,H,W]
    depth = normalize_depth_fixed(batch["depth"]).to(device)  # [B,1,H,W]

    print("\nDepth normalized:")
    print("  shape:", depth.shape)
    print("  min/max:", depth.min().item(), depth.max().item())

    # The transformer expects [B,S,3,H,W]
    # rgb_seq = rgb.unsqueeze(1)
    # depth_seq = depth.unsqueeze(1)

    print("\nModel input:")
    print(" rgb_seq:", rgb.shape)
    print(" depth_seq:", depth.shape)

    print("\n============================================")
    print(" Running forward pass...")
    print("============================================")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        preds = model(rgb, depth)

    print("\nPredictions returned:")
    for k, v in preds.items():
        print(f"{k:<12} {tuple(v.shape)}   min={v.min().item():.4f}   max={v.max().item():.4f}")

    # ------------------------------------------------------
    # Compute loss
    # ------------------------------------------------------
    print("\n============================================")
    print(" Computing Loss...")
    print("============================================")

    loss, loss_dict = compute_loss(preds, batch)

    print(f"\nTotal Loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"{k:<15} {v:.6f}")

    print("\n============================================")
    print(" Test completed successfully ✔")
    print("============================================")


if __name__ == "__main__":
    main()
