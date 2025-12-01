import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.layers import PatchEmbed, PatchEmbed_Mlp
from models.layers.block import Block
from models.layers.rope import RotaryPositionEmbedding2D, PositionGetter
# from models.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from dinov3.dinov3.models.vision_transformer import vit_base, vit_small

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class DINOv3Backbone(nn.Module):
    """
    Wraps DinoVisionTransformer so that forward(x) returns the same
    dict interface as your old DINOv2 backbone, including
    "x_norm_patchtokens".
    """

    def __init__(
        self,
        img_size=592,
        patch_size=16,
        checkpoint_path=None,
    ):
        super().__init__()

        # This creates a DinoVisionTransformer
        # self.model = vit_base(
        #     patch_size=patch_size,
        #     img_size=img_size,
        # )
        self.model = vit_small(
            patch_size=patch_size,
            img_size=img_size,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # allow extra keys: storage_tokens, bias_mask, ls1/ls2.gamma, etc.
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print("Loaded DINOv3 vit_base:")
            print("  missing keys   :", missing)
            print("  unexpected keys:", unexpected)

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad_(False)

        # expose for VGGT
        self.embed_dim = self.model.embed_dim

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W]

        DinoVisionTransformer.forward_features(x, masks=None) returns a dict:
        {
            "x_norm_clstoken": ...,
            "x_storage_tokens": ...,
            "x_norm_patchtokens": ...,
            "x_prenorm": ...,
            "masks": ...
        }
        """
        out = self.model.forward_features(x, masks=None)
        return out  # <-- IMPORTANT: a dict, not CLS tensor
    
# def DINOv3PatchEmbed(
#     img_size=592,
#     patch_size=16,
#     num_register_tokens=4,
#     checkpoint_path=None,
# ):
#     """
#     Load DINOv3 ViT-B/16 via torch.hub so the architecture matches
#     the pretrained checkpoint exactly.
#     """
#     # Use official DINOv3 hub entry
#     # model = torch.hub.load(
#     #     'facebookresearch/dinov3',
#     #     'dinov3_vitb16',
#     #     pretrained=False
#     # )

#     model = vit_base(
#         patch_size=patch_size,
#         img_size=img_size,
#         num_register_tokens=num_register_tokens,
#     )

#     if checkpoint_path is not None:
#         state_dict = torch.load(checkpoint_path, map_location="cpu")
#         # allow extra keys: storage_tokens, bias_mask, ls1/ls2.gamma, etc.
#         missing, unexpected = model.load_state_dict(state_dict, strict=False)
#         print("Loaded DINOv3 vit_base:")
#         print("  missing keys   :", missing)
#         print("  unexpected keys:", unexpected)


#     # Resize positional embeddings for your custom image size
#     # model.patch_embed.img_size = (img_size, img_size)
#     # model.pos_embed = nn.Parameter(
#     #     torch.nn.functional.interpolate(
#     #         model.pos_embed.unsqueeze(0),
#     #         size=(img_size // patch_size, img_size // patch_size),
#     #         mode='bicubic',
#     #         align_corners=False
#     #     ).squeeze(0)
#     # )

#     # Freeze
#     for p in model.parameters():
#         p.requires_grad = False

#     return model

# def DINOv3PatchEmbed(img_size=592, patch_size=16, num_register_tokens=4, checkpoint_path=None):
#     """
#     Wrapper for DINOv3 ViT-B/16 returning patch tokens only.
#     img_size and patch_size can be arbitrary (positional
#     embeddings automatically resized).
#     """

#     # Load pretrained DINOv3 ViT-B/16
#     model = vit_base(
#         patch_size=patch_size,
#         img_size=img_size,
#         num_register_tokens=num_register_tokens,
#     )

#     # Load weights (either a URL or local path)
#     # if download_url is not None:
#     #     state_dict = torch.hub.load_state_dict_from_url(download_url, map_location="cpu")
#     #     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     #     print("Loaded DINOv3, missing:", missing, "unexpected:", unexpected)
#     if checkpoint_path is not None:
#         state_dict = torch.load(checkpoint_path, map_location="cpu")
#         model.load_state_dict(state_dict)
#         # missing, unexpected = model.load_state_dict(state_dict, strict=True)


#     # IMPORTANT: Freeze
#     for p in model.parameters():
#         p.requires_grad = False

#     return model
# class DINOv3PatchEmbed(nn.Module):
#     """
#     Wrapper for DINOv3 ViT-B/16 returning patch tokens only.
#     img_size and patch_size can be arbitrary (positional
#     embeddings automatically resized).
#     """

#     def __init__(self, img_size=592, patch_size=16, num_register_tokens=4, download_url="/home/mirshad7/HunyuanWorld-Mirror/models/dinov3/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"):
#         super().__init__()

#         # Load pretrained DINOv3 ViT-B/16
#         self.model = vit_base(
#             patch_size=patch_size,
#             img_size=img_size,
#             num_register_tokens=num_register_tokens,
#         )

#         # Load weights (either a URL or local path)
#         if download_url is not None:
#             state_dict = torch.hub.load_state_dict_from_url(download_url, map_location="cpu")
#             missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
#             print("Loaded DINOv3, missing:", missing, "unexpected:", unexpected)

#         # IMPORTANT: Freeze
#         for p in self.model.parameters():
#             p.requires_grad = False

        # self.embed_dim = self.model.embed_dim
        # self.patch_size = patch_size
        # self.img_size = img_size

    # def forward(self, x):
    #     """
    #     x: [B, C, H, W]
    #     Returns patch tokens only.
    #     """
    #     out = self.model.prepare_tokens(x)
    #     return out  # [B, N, embed_dim]

class VisualGeometryTransformer(nn.Module):
    """
    Depth-only Visual Geometry Transformer (VGGT-lite)
    --------------------------------------------------
    A simplified VGGT using only depth conditioning (no camera, no rays).
    """

    # model = DinoVisionTransformer(
    #     patch_size=patch_size,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     ffn_ratio=4,
    #     **kwargs,
    # )
    
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        block_fn: nn.Module = Block,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        patch_embed: str = "dinov3",
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        enable_cond: bool = False,     # depth conditioning toggle
        sampling_strategy: str = "uniform",
        fixed_patch_embed: bool = True,
        # intermediate_idxs: List[int] = [2, 5, 7, 11],
        intermediate_idxs=[1, 2, 3]
    ):
        super().__init__()

        self.enable_cond = enable_cond
        self.sampling_strategy = sampling_strategy
        self.intermediate_idxs = intermediate_idxs
        self.depth = depth
        self.patch_size = patch_size

        # Patch embedding
        self.patch_embed = self._init_patch_embedding_module(
            patch_embed, img_size, patch_size, num_register_tokens,
            embed_dim=embed_dim, is_fixed=fixed_patch_embed
        )

        # Determine true embedding dim from backbone
        self.model_dim = getattr(self.patch_embed, "embed_dim", embed_dim)

        # Depth conditioning branch
        if self.enable_cond:
            self.depth_embed = self._init_patch_embedding_module(
                "conv+mlp", img_size, patch_size, num_register_tokens,
                embed_dim=self.model_dim, in_chans=1
            )

        # Rotary positional embedding
        self._init_rotary_position_embedding(rope_freq)

        # Transformer blocks (frame only)
        self._init_transformer_blocks(
            block_fn, self.model_dim, num_heads, mlp_ratio,
            qkv_bias, proj_bias, ffn_bias, init_values, qk_norm
        )

        # Register tokens only (no camera)
        self._init_learnable_tokens(self.model_dim, num_register_tokens)

        # Patch start index = number of register tokens
        self.patch_start_idx = num_register_tokens

        # Normalization constants
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False

    # --------------------------------------------------------- #
    #  Helper builders
    # --------------------------------------------------------- #
    def _init_patch_embedding_module(
        self, patch_embed_type, img_size, patch_size, num_reg_tokens,
        interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0,
        init_values=1.0, embed_dim=1024, is_fixed=False, in_chans=3
    ):
        if "conv" in patch_embed_type:
            if "mlp" in patch_embed_type:
                mod = PatchEmbed_Mlp(
                    img_size=img_size, patch_size=patch_size,
                    in_chans=in_chans, embed_dim=embed_dim
                )
            else:
                mod = PatchEmbed(
                    img_size=img_size, patch_size=patch_size,
                    in_chans=in_chans, embed_dim=embed_dim
                )

        elif "dinov3" in patch_embed_type:
            # from src.models.layers.dinov3_patch_embed import DINOv3PatchEmbed

            checkpoint_paths = {
                "dinov3_vitb16": "/home/mirshad7/HunyuanWorld-Mirror/dinov3/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                "dinov3_vits16": "/home/mirshad7/HunyuanWorld-Mirror/dinov3/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            }

            mod = DINOv3Backbone(
                img_size=img_size,
                patch_size=patch_size,
                checkpoint_path=checkpoint_paths.get(patch_embed_type, None),
            )
            # print("hereee")
            # checkpoint_paths = {
            #     "dinov3_vitb16": "/home/mirshad7/HunyuanWorld-Mirror/dinov3/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"  # put link here
            # }

            # mod = DINOv3PatchEmbed(
            #     img_size=img_size,
            #     patch_size=patch_size,
            #     num_register_tokens=num_reg_tokens,
            #     checkpoint_path=checkpoint_paths.get(patch_embed_type, None)
            # )

        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }
            mod = vit_models[patch_embed_type](
                img_size=img_size, patch_size=patch_size,
                num_register_tokens=num_reg_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks, init_values=init_values,
            )
            if hasattr(mod, "mask_token"):
                mod.mask_token.requires_grad_(False)

        if is_fixed:
            for p in mod.parameters():
                p.requires_grad_(False)
        return mod

    def _init_rotary_position_embedding(self, rope_freq: int):
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.pos_getter = PositionGetter() if self.rope is not None else None

    def _init_transformer_blocks(
        self, block_fn, embed_dim, num_heads, mlp_ratio,
        qkv_bias, proj_bias, ffn_bias, init_values, qk_norm
    ):
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                init_values=init_values, qk_norm=qk_norm, rope=self.rope
            )
            for _ in range(self.depth)
        ])

    def _init_learnable_tokens(self, embed_dim, num_reg_tokens):
        self.reg_token = nn.Parameter(torch.zeros(1, 2, num_reg_tokens, embed_dim))
        nn.init.normal_(self.reg_token, std=1e-6)

    # --------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------- #
    def forward(
        self,
        images: torch.Tensor,                 # [B,S,3,H,W]
        depth_maps: Optional[torch.Tensor] = None,  # [B,S,1,H,W]
        cond_flag: bool = False,
    ) -> Tuple[List[torch.Tensor], int]:

        b, seq_len, ch, h, w = images.shape
        assert ch == 3, f"Expected RGB images with 3 channels, got {ch}"

        # print("Input images shape:", images.shape)

        # normalize and patchify
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images = (images - self._resnet_mean) / self._resnet_std
            images = images.view(b * seq_len, ch, h, w)
            # print("Normalized images shape:", images.shape)
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                # print("DICT PATCH TOKENS++++++++++++++++++++++++++++++++++++++++++++++++, taking x_norm_patchtokens")
                patch_tokens = patch_tokens["x_norm_patchtokens"]
        #print("Patch tokens shape:", patch_tokens.shape)
        _, patch_count, tok_dim = patch_tokens.shape
        assert tok_dim == self.model_dim, f"Patch tokens {tok_dim}, expected {self.model_dim}"

        # register tokens
        reg_tokens = expand_and_flatten_special_tokens(self.reg_token, b, seq_len)

        # depth conditioning (additive)
        if self.enable_cond and cond_flag and depth_maps is not None:
            depth_maps = depth_maps.view(b * seq_len, 1, h, w)
            depth_tokens = self.depth_embed(depth_maps)
            if depth_tokens.dim() == 4:
                ph, pw = h // self.patch_size, w // self.patch_size
                depth_tokens = depth_tokens.permute(0, 2, 3, 1).reshape(b * seq_len, ph * pw, self.model_dim)
            patch_tokens = patch_tokens + depth_tokens

        # combine register + patch tokens
        all_tokens = torch.cat([reg_tokens, patch_tokens], dim=1)
        _, patch_count, embed_dim = all_tokens.shape

        # position embedding
        pos_emb = None
        if self.rope is not None:
            pos_emb = self.pos_getter(
                b * seq_len, h // self.patch_size, w // self.patch_size, device=images.device
            )
            if self.patch_start_idx > 0:
                special_pos = torch.zeros(
                    b * seq_len, self.patch_start_idx, 2,
                    device=images.device, dtype=pos_emb.dtype
                )
                pos_emb = torch.cat([special_pos, pos_emb], dim=1)

        # forward through transformer
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs, tokens = [], all_tokens
            for idx in range(self.depth):
                tokens = self._process_block(tokens, b, seq_len, patch_count, embed_dim, idx, pos_emb)
                if idx in self.intermediate_idxs:
                    outputs.append(tokens)

        return outputs, self.patch_start_idx

    def _process_block(
        self, tokens, b, seq_len, patch_count, embed_dim, idx, pos=None
    ):
        target_shape = (b * seq_len, patch_count, embed_dim)
        pos_shape = (b * seq_len, patch_count, 2) if pos is not None else None
        tokens = tokens.view(*target_shape)
        if pos is not None:
            pos = pos.view(*pos_shape)

        # if self.training:
        #     tokens = checkpoint(self.blocks[idx], tokens, pos=pos, use_reentrant=self.use_reentrant)
        # else:
        #     tokens = self.blocks[idx](tokens, pos=pos)

        tokens = self.blocks[idx](tokens, pos=pos)
        return tokens.view(b, seq_len, patch_count, embed_dim)


# --------------------------------------------------------- #
#  Utility
# --------------------------------------------------------- #
def expand_and_flatten_special_tokens(token_tensor, b, seq_len):
    """Expand register tokens for each frame (no camera token)."""
    first = token_tensor[:, 0:1, ...].expand(b, 1, *token_tensor.shape[2:])
    rest = token_tensor[:, 1:, ...].expand(b, seq_len - 1, *token_tensor.shape[2:])
    combined = torch.cat([first, rest], dim=1)
    return combined.reshape(b * seq_len, *combined.shape[2:])



