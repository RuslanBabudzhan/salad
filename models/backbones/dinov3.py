import torch.nn as nn


class DINOv3(nn.Module):
    """DINOv3 ViT patch features and CLS token, with only the last blocks trained."""

    def __init__(self, model_name="dinov3_vitb16", weights=None, num_trainable_blocks=4):
        super().__init__()
        from transformers import AutoModel

        weights = weights or f"facebook/{model_name.replace('_', '-')}-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(weights)
        if self.model.config.model_type != "dinov3_vit":
            raise ValueError("DINOv3 requires a ViT checkpoint")
        if not 0 <= num_trainable_blocks <= len(self.model.layer):
            raise ValueError("num_trainable_blocks must be between 0 and the backbone depth")
        self.num_channels = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = self.model.config.num_register_tokens

        self.model.requires_grad_(False)
        for block in self.model.layer[len(self.model.layer) - num_trainable_blocks:]:
            block.requires_grad_(True)
        self.model.train(self.training)

    def forward(self, x):
        batch, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"Image dimensions must be divisible by {self.patch_size}")
        # The frozen prefix builds no autograd graph for ordinary image inputs.
        tokens = self.model(pixel_values=x).last_hidden_state
        patches = tokens[:, 1 + self.num_register_tokens:]
        features = patches.transpose(1, 2).reshape(
            batch, self.num_channels, height // self.patch_size, width // self.patch_size,
        )
        return features, tokens[:, 0]
