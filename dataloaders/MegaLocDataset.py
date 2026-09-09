from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MegaLocDataset(Dataset):
    """One item is one scheduled iteration: {subset: (images, labels)}.

    Each subset yields images [128, C, H, W] and labels [128], with four
    consecutive images per place. Labels are local to each subset/iteration;
    compute the loss separately for each subset.

    ``config`` is a YAML manifest selecting dataset folders and subsets.
    ``transform`` must convert an RGB PIL image to a tensor of fixed shape.
    Use DataLoader(batch_size=None, shuffle=False): items are already batched.
    """

    def __init__(self, config, transform):
        self.config_path = Path(config)
        self.config = OmegaConf.load(self.config_path)
        self.root = (self.config_path.parent / self.config.get("path", ".")).resolve()
        self.transform = transform

        self.batches = {}
        self.paths = {}
        self.roots = {}
        for source in self.config.data:
            root = self.root / source.name
            for subset in source.subsets:
                if subset in self.batches:
                    raise ValueError(f"Subset {subset!r} is selected more than once")
                batches = np.load(root / "batches" / f"{subset}.npy", mmap_mode="r")
                if batches.ndim != 3 or batches.shape[1:] != (32, 4):
                    raise ValueError(f"{subset}: expected schedule [iterations, 32, 4], got {batches.shape}")
                self.batches[subset] = batches
                metadata = pd.read_parquet(
                    root / "meta" / f"{subset}.parquet",
                    columns=["image_id", "rel_path"],
                )
                paths = metadata.set_index("image_id")["rel_path"]
                if not paths.index.is_unique:
                    raise ValueError(f"{subset}: duplicate image IDs in metadata")
                self.paths[subset] = paths
                self.roots[subset] = root

        self.subsets = tuple(self.batches)
        if not self.subsets:
            raise ValueError("At least one subset is required")
        if len({len(batches) for batches in self.batches.values()}) != 1:
            raise ValueError("All subsets must have the same number of iterations")
        self.labels = torch.arange(32).repeat_interleave(4)

    def __len__(self):
        return len(self.batches[self.subsets[0]])

    def __getitem__(self, index):
        batch = {}
        for subset in self.subsets:
            image_ids = self.batches[subset][index].reshape(-1)
            paths = self.paths[subset].loc[image_ids]
            images = []
            for path in paths:
                with Image.open(self.roots[subset] / path) as image:
                    images.append(self.transform(image.convert("RGB")))
            batch[subset] = torch.stack(images), self.labels.clone()
        return batch
