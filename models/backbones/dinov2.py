import os

import torch
import torch.nn as nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # PyTorch SDPA is portable; prebuilt xFormers kernels may not support the GPU.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.patch_size = 14
        if not 0 <= num_trainable_blocks <= len(self.model.blocks):
            raise ValueError("num_trainable_blocks must be between 0 and the backbone depth")
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        self.model.requires_grad_(False)
        for block in self.model.blocks[len(self.model.blocks) - num_trainable_blocks:]:
            block.requires_grad_(True)


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Image dimensions must be divisible by {self.patch_size}")

        x = self.model.prepare_tokens_with_masks(x)
        first_trainable = len(self.model.blocks) - self.num_trainable_blocks

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:first_trainable]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[first_trainable:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // self.patch_size, W // self.patch_size, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f
