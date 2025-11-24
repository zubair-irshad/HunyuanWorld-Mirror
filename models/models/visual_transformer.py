import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.layers import PatchEmbed, PatchEmbed_Mlp
from src.models.layers.block import Block
from src.models.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from src.models.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class VisualGeometryTransformer(nn.Module):
    """
    Depth-only Visual Geometry Transformer (VGGT-lite)
    --------------------------------------------------
    A simplified VGGT using only depth conditioning (no camera, no rays).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        block_fn: nn.Module = Block,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        patch_embed: str = "dinov2_vitb14_reg",
        qk_norm: bool = True,
        rope_freq: int = 100,
        init_values: float = 0.01,
        enable_cond: bool = False,     # depth conditioning toggle
        sampling_strategy: str = "uniform",
        fixed_patch_embed: bool = True,
        intermediate_idxs: List[int] = [2, 5, 7, 11],
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

        # normalize and patchify
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            images = (images - self._resnet_mean) / self._resnet_std
            images = images.view(b * seq_len, ch, h, w)
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

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

        if self.training:
            tokens = checkpoint(self.blocks[idx], tokens, pos=pos, use_reentrant=self.use_reentrant)
        else:
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



