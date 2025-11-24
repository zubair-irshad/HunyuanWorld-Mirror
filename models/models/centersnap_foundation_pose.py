# centersnap_worldmirror.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

# re-use your modules exactly as-is
from models.models.visual_transformer import VisualGeometryTransformer
from models.heads.dense_head import DPTHead

# ------- Loss helpers -------
def mse_loss(pred: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None):
    """
    Standard MSE. If valid mask is provided (same HxW broadcastable),
    we compute mean over valid pixels only.
    """
    if valid is not None:
        # valid assumed to be {0,1}, shape [B,1,H,W] or [B,H,W]
        w = (valid > 0).float()
        num = torch.clamp(w.sum(), min=1.0)
        return ((pred - target) ** 2 * w).sum() / num
    return torch.mean((pred - target) ** 2)

def masked_l1(pred: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None):
    """
    Masked L1 for pose maps. If no valid provided, reduces to mean L1.
    """
    if valid is not None:
        w = (valid > 0).float()
        num = torch.clamp(w.sum(), min=1.0)
        # broadcast mask to channel dims if needed
        while w.ndim < pred.ndim:
            w = w.expand(-1, pred.size(1), -1, -1)
        return (torch.abs(pred - target) * w).sum() / num
    return torch.mean(torch.abs(pred - target))


class WorldMirrorCenterSnap(nn.Module):
    """
    WorldMirror variant for CenterSnap-style supervision.
    - Inputs:
        rgb:  [B, 3, H, W]    (values in [0,1])
        depth:[B, 1, H, W]    (optional; recommended)
    - Outputs:
        heatmap:  [B, 1, H, W]
        pose_map: [B, 12, H, W]
    Notes:
        * seq_len is fixed to 1 (we just add an artificial S dim internally)
        * depth can be used as a 'pow3r' conditioning through VisualGeometryTransformer
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 512,
        patch_embed: str = "dinov2_vitb14_reg",
        fixed_patch_embed: bool = False,
        enable_interpolation: bool = False,
        max_resolution: int = 2044,
        condition_strategy: List[str] = ("token", "pow3r", "token"),
        use_depth_condition: bool = True,
        # Heads
        heatmap_activation: str = "sigmoid",   # good for gaussian heatmaps in [0,1]
        pose_activation: Optional[str] = None, # identity
    ):
        super().__init__()
        self.patch_size = patch_size
        self.use_depth_condition = use_depth_condition


        self.encoder = VisualGeometryTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=512,
            enable_cond=True,       # to enable depth conditioning
        )

        dec_in = self.encoder.model_dim   # <-- ViT-B/14 => 768
        print("dec_in", dec_in)

        # 1) Heatmap head: 1 channel
        self.heatmap_head = DPTHead(
            dim_in=dec_in,
            output_dim=1,
            patch_size=patch_size,
            activation="linear",  # "sigmoid"
        )

        # 2) Pose map head: 12 channels
        self.pose_head = DPTHead(
            dim_in=dec_in,
            output_dim=12,
            patch_size=patch_size,
            activation="linear",  # identity / None
        )

    @torch.no_grad()
    def param_counts(self) -> Tuple[int, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.heatmap_head.parameters()) \
              + sum(p.numel() for p in self.pose_head.parameters())

        # also return trainable parameters
        enc_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        dec_trainable = sum(p.numel() for p in self.heatmap_head.parameters() if p.requires_grad) \
              + sum(p.numel() for p in self.pose_head.parameters() if p.requires_grad)
        return enc, dec, enc_trainable, dec_trainable


    def forward(
        self,
        rgb: torch.Tensor,            # [B,3,H,W] in [0,1]
        depth: Optional[torch.Tensor] = None,  # [B,1,H,W]
    ) -> Dict[str, torch.Tensor]:

        assert rgb.ndim == 4 and rgb.size(1) == 3, "rgb must be [B,3,H,W]"
        B, _, H, W = rgb.shape
        # add seq-dim=1 to match VGGT interface
        imgs = rgb.unsqueeze(1)  # [B,1,3,H,W]

        if self.use_depth_condition and depth is not None:
            token_list, patch_start_idx = self.encoder(imgs, depth_maps=depth, cond_flag=True)
        else:
            token_list, patch_start_idx = self.encoder(imgs)

        heatmap_pred = self.heatmap_head(token_list, images=imgs, patch_start_idx=patch_start_idx)
        pose_pred = self.pose_head(token_list, images=imgs, patch_start_idx=patch_start_idx)

        return {
            "heatmap": heatmap_pred,
            "pose_map": pose_pred
        }

# -------------------- sanity check --------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldMirrorCenterSnap(
        img_size=518, patch_size=14, embed_dim=1024,
        use_depth_condition=True
    ).to(device).eval()

    B = 4
    H, W = 406, 518
    rgb = torch.rand(B, 3, H, W, device=device)
    depth = torch.rand(B, 1, H, W, device=device)

    with torch.no_grad():
        out = model(rgb, depth)
        print("heatmap:", out["heatmap"].shape)
        print("pose_map:", out["pose_map"].shape)

    enc_params, dec_params = model.param_counts()
    print(f"#params - encoder: {enc_params/1e6:.2f}M | decoders: {dec_params/1e6:.2f}M")
