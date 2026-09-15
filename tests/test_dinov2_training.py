import os
from unittest.mock import patch
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from models.backbones.dinov2 import DINOv2
from train_megaloc import make_backbone_config, make_train_transform


class FakeDINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.norm = nn.LayerNorm(4)

    def prepare_tokens_with_masks(self, images):
        batch, _, height, width = images.shape
        tokens = 1 + height // 14 * (width // 14)
        return images.mean((1, 2, 3)).view(batch, 1, 1).expand(batch, tokens, 4)


class DINOv2TrainingTests(unittest.TestCase):
    def test_megaloc_config_and_trainable_blocks(self):
        config = OmegaConf.load("configs/train_megalocV1.5.yaml")
        normalize = make_train_transform(config.augmentation).transforms[-1]
        self.assertEqual(make_backbone_config(config.model), {
            "num_trainable_blocks": 4, "norm_layer": True, "return_token": True,
        })
        self.assertEqual(tuple(config.augmentation.image_size), (224, 224))
        self.assertEqual(normalize.mean, [0.485, 0.456, 0.406])
        self.assertEqual(normalize.std, [0.229, 0.224, 0.225])

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XFORMERS_DISABLED", None)
            with patch.dict("models.backbones.dinov2.DINOV2_ARCHS", {"dinov2_vitb14": 4}), \
                    patch("torch.hub.load", return_value=FakeDINOv2()):
                backbone = DINOv2(num_trainable_blocks=1, norm_layer=True, return_token=True)
            self.assertEqual(os.environ["XFORMERS_DISABLED"], "1")
        features, token = backbone(torch.randn(2, 3, 28, 42))
        self.assertEqual(features.shape, (2, 4, 2, 3))
        self.assertEqual(token.shape, (2, 4))
        self.assertTrue(all(not p.requires_grad for p in backbone.model.blocks[0].parameters()))
        self.assertTrue(all(not p.requires_grad for p in backbone.model.blocks[1].parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.blocks[2].parameters()))
        self.assertTrue(all(not p.requires_grad for p in backbone.model.norm.parameters()))

        with self.assertRaisesRegex(ValueError, "divisible"):
            backbone(torch.randn(1, 3, 29, 28))


if __name__ == "__main__":
    unittest.main()
