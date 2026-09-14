"""Train DINOv3 + SALAD on MegaLoc schedules with gradient caching."""

import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataloaders.MegaLocDataset import MegaLocDataset
from vpr_model import VPRModel


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_train_transform(config):
    return T.Compose([
        T.Resize(tuple(config.image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=config.randaugment_ops,
                      interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(config.get("image_mean", IMAGENET_MEAN),
                    config.get("image_std", IMAGENET_STD)),
    ])


def make_run_dir(logs_dir):
    started_at = datetime.now().astimezone()
    run_dir = logs_dir / started_at.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, started_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="MegaLoc data manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/train_megaloc.yaml"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"), default="16-mixed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required for MegaLoc training")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")

    config = OmegaConf.load(args.config)
    dataset = MegaLocDataset(args.data, make_train_transform(config.augmentation))
    max_steps = len(dataset) if config.training.iterations is None else config.training.iterations
    if max_steps < 1:
        parser.error("training.iterations must be at least 1")
    if config.training.grad_cache_chunk_size < 0:
        parser.error("training.grad_cache_chunk_size must be nonnegative")
    if config.training.checkpoint_every_n_steps < 1:
        parser.error("training.checkpoint_every_n_steps must be positive")

    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=1 if args.num_workers else None,
    )
    run_dir, started_at = make_run_dir(args.logs_dir)
    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    resolved_config.training.iterations = max_steps
    resolved_config.run = {
        "started_at": started_at.isoformat(),
        "source_config": str(args.config.resolve()),
        "data_config": str(args.data.resolve()),
        "subsets": list(dataset.subsets),
        "available_iterations": len(dataset),
        "num_workers": args.num_workers,
        "precision": args.precision,
    }
    OmegaConf.save(resolved_config, run_dir / "config.yaml")
    resolved_data = OmegaConf.create(OmegaConf.to_container(dataset.config, resolve=True))
    resolved_data.path = str(dataset.root)
    OmegaConf.save(resolved_data, run_dir / "data.yaml")

    print(
        f"{len(dataset)} scheduled iterations across {len(dataset.subsets)} subsets "
        f"({', '.join(dataset.subsets)}); training for {max_steps} steps",
        flush=True,
    )
    print(f"Run directory: {run_dir.resolve()}", flush=True)

    pl.seed_everything(config.training.seed, workers=True)
    scheduler_config = {
        "start_factor": 1,
        "end_factor": config.scheduler.end_factor,
        "total_iters": max_steps,
    }
    model = VPRModel(
        backbone_arch=config.model.backbone,
        backbone_config={
            "weights": config.model.backbone_weights,
            "num_trainable_blocks": config.model.trainable_blocks,
        },
        agg_arch="SALAD",
        agg_config={
            "num_clusters": config.model.num_clusters,
            "cluster_dim": config.model.cluster_dim,
            "token_dim": config.model.token_dim,
            "output_dim": config.model.output_dim,
        },
        lr=config.optimizer.learning_rate,
        optimizer=config.optimizer.name,
        weight_decay=config.optimizer.weight_decay,
        lr_sched=config.scheduler.name,
        lr_sched_args=scheduler_config,
        loss_name=config.loss.name,
        miner_name=config.loss.miner,
        miner_margin=config.loss.miner_margin,
        grad_cache_chunk_size=config.training.grad_cache_chunk_size,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step-{step:06d}",
        auto_insert_metric_name=False,
        every_n_train_steps=config.training.checkpoint_every_n_steps,
        save_on_train_epoch_end=False,
        save_top_k=-1,
        save_last=False,
        save_on_exception=True,
        save_weights_only=False,
        enable_version_counter=False,
    )
    logger = pl.loggers.CSVLogger(
        save_dir=run_dir,
        name="",
        version="",
        flush_logs_every_n_steps=1,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=run_dir,
        logger=logger,
        precision=args.precision,
        max_epochs=-1,
        max_steps=max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint],
        log_every_n_steps=1,
    )
    trainer.fit(model=model, train_dataloaders=loader)
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")


if __name__ == "__main__":
    main()
