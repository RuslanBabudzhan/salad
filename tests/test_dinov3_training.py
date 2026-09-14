from copy import deepcopy
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from models.backbones.dinov3 import DINOv3
from train_megaloc import make_train_transform
from vpr_model import VPRModel


class DINOv3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.weights = str(Path(cls.temporary.name) / "backbone")
        torch.manual_seed(17)
        config = DINOv3ViTConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, patch_size=16, num_register_tokens=4,
            pos_embed_rescale=None,
        )
        DINOv3ViTModel(config).save_pretrained(cls.weights)

    def make_model(self, chunk_size=3, output_dim=None):
        return VPRModel(
            backbone_arch="dinov3_vitb16",
            backbone_config={"weights": self.weights, "num_trainable_blocks": 1},
            agg_arch="SALAD",
            agg_config={
                "num_clusters": 2, "cluster_dim": 3, "token_dim": 4,
                "dropout": 0, "output_dim": output_dim,
            },
            optimizer="sgd", lr=0.01, momentum=0, weight_decay=0,
            lr_sched_args={"start_factor": 1, "end_factor": 0.2, "total_iters": 1},
            grad_cache_chunk_size=chunk_size,
        )

    def test_patch_grid_and_cls_exclude_registers(self):
        backbone = DINOv3(weights=self.weights, num_trainable_blocks=1).eval()
        images = torch.randn(2, 3, 32, 48)
        with torch.no_grad():
            tokens = backbone.model(pixel_values=images).last_hidden_state
            features, cls = backbone(images)
        self.assertEqual(features.shape, (2, 32, 2, 3))
        torch.testing.assert_close(features.flatten(2).transpose(1, 2), tokens[:, 5:])
        torch.testing.assert_close(cls, tokens[:, 0])

    def test_only_selected_blocks_receive_gradients(self):
        for count in (0, 1, 3):
            backbone = DINOv3(weights=self.weights, num_trainable_blocks=count).train()
            features, cls = backbone(torch.randn(2, 3, 32, 32))
            if count:
                (features[:, 0].mean() + cls[:, 1].mean()).backward()
            for i, block in enumerate(backbone.model.layer):
                self.assertTrue(all(p.requires_grad == (i >= 3 - count) for p in block.parameters()))
                if i < 3 - count:
                    self.assertTrue(all(p.grad is None for p in block.parameters()))
                else:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()))
            self.assertTrue(all(p.grad is None for p in backbone.model.embeddings.parameters()))
            self.assertTrue(all(p.grad is None for p in backbone.model.norm.parameters()))

    def test_invalid_block_count_and_image_size(self):
        with self.assertRaises(ValueError):
            DINOv3(weights=self.weights, num_trainable_blocks=4)
        backbone = DINOv3(weights=self.weights, num_trainable_blocks=1)
        with self.assertRaisesRegex(ValueError, "divisible"):
            backbone(torch.randn(1, 3, 33, 32))

    def test_satellite_config_uses_checkpoint_normalization(self):
        config = OmegaConf.load("configs/train_megalocV1.3.yaml")
        normalize = make_train_transform(config.augmentation).transforms[-1]
        self.assertEqual(config.model.backbone, "dinov3_vitl16")
        self.assertEqual(config.model.backbone_weights,
                         "facebook/dinov3-vitl16-pretrain-sat493m")
        self.assertEqual(normalize.mean, [0.430, 0.411, 0.296])
        self.assertEqual(normalize.std, [0.213, 0.156, 0.143])

    def test_multiple_subset_losses_one_update_and_checkpoint(self):
        torch.manual_seed(3)
        model = self.make_model(output_dim=6)
        initial_projection = model.aggregator.projection.weight.detach().clone()
        reference = deepcopy(model).train()
        batch = {f"subset{i}": (torch.randn(8, 3, 32, 32), torch.arange(8) // 4) for i in range(3)}
        optimizers, _ = reference.configure_optimizers()
        expected_loss = 0
        for images, labels in batch.values():
            descriptors = reference(images)
            loss = reference.loss_fn(descriptors, labels, reference.miner(descriptors, labels))
            expected_loss += loss.detach()
            loss.backward()
        optimizers[0].step()

        metrics_dir = Path(self.temporary.name) / "metrics"
        logger = pl.loggers.CSVLogger(
            metrics_dir, name="", version="", flush_logs_every_n_steps=1,
        )
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, precision="32-true", max_steps=1,
            limit_val_batches=0, num_sanity_val_steps=0, logger=logger,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(model.lr_schedulers().last_epoch, 1)
        self.assertFalse(torch.equal(model.aggregator.projection.weight, initial_projection))
        torch.testing.assert_close(trainer.callback_metrics["loss"], expected_loss)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        with (metrics_dir / "metrics.csv").open(newline="") as file:
            rows = [row for row in csv.DictReader(file) if row["loss"]]
        self.assertEqual(len(rows), 1)
        self.assertEqual({f"loss/subset{i}" for i in range(3)} - rows[0].keys(), set())

        checkpoint = Path(self.temporary.name) / "training.ckpt"
        trainer.save_checkpoint(checkpoint)
        restored = VPRModel.load_from_checkpoint(checkpoint).eval()
        model.eval()
        with torch.no_grad():
            descriptors = restored(batch["subset0"][0])
            self.assertEqual(descriptors.shape, (8, 6))
            torch.testing.assert_close(descriptors.norm(dim=-1), torch.ones(8))
            torch.testing.assert_close(descriptors, model(batch["subset0"][0]))
        state = torch.load(checkpoint, weights_only=False)
        self.assertEqual(len(state["optimizer_states"]), 1)
        self.assertEqual(state["lr_schedulers"][0]["last_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
