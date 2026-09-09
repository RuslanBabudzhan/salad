from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor

from dataloaders.MegaLocDataset import MegaLocDataset


class MegaLocDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.subsets = ("msls", "sf_xl_frontal")
        self.sources = ("dataset_a", "dataset_b")
        self.config = self.root / "data.yaml"
        ids = np.arange(128, dtype=np.uint32) * 7 + 1000
        schedule = np.stack([ids, ids[::-1]]).reshape(2, 32, 4)
        for offset, (source, subset) in enumerate(zip(self.sources, self.subsets)):
            root = self.root / source
            (root / "batches").mkdir(parents=True)
            (root / "meta").mkdir()
            directory = root / "images" / subset
            directory.mkdir(parents=True)
            paths = []
            for i in range(128):
                path = directory / f"sample_{i}.png"
                Image.new("RGB", (2, 2), (i, offset, 0)).save(path)
                paths.append(path.relative_to(root).as_posix())
            # Metadata order and filenames deliberately differ from image IDs.
            pd.DataFrame({"image_id": ids, "rel_path": paths}).iloc[::-1].to_parquet(
                root / "meta" / f"{subset}.parquet", index=False,
            )
            np.save(root / "batches" / f"{subset}.npy", schedule)
        self.write_config([
            {"name": source, "subsets": [subset]}
            for source, subset in zip(self.sources, self.subsets)
        ])

    def write_config(self, data):
        OmegaConf.save(OmegaConf.create({"path": ".", "data": data}), self.config)

    def test_schedule_order_sparse_ids_and_subset_local_labels(self):
        dataset = MegaLocDataset(self.config, PILToTensor())
        self.assertEqual(dataset.subsets, self.subsets)
        self.assertEqual(len(dataset), 2)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        for index, batch in enumerate(loader):
            self.assertEqual(tuple(batch), self.subsets)
            expected = torch.arange(128, dtype=torch.uint8)
            if index == 1:
                expected = expected.flip(0)
            for offset, (images, labels) in enumerate(batch.values()):
                self.assertEqual(images.shape, (128, 3, 2, 2))
                torch.testing.assert_close(images[:, 0, 0, 0], expected)
                self.assertTrue((images[:, 1] == offset).all())
                torch.testing.assert_close(labels, torch.arange(32).repeat_interleave(4))

    def test_subset_selection(self):
        self.write_config([
            {"name": "dataset_a", "subsets": ["msls"]},
        ])
        dataset = MegaLocDataset(self.config, PILToTensor())
        self.assertEqual(tuple(dataset[0]), ("msls",))

    def test_rejects_misaligned_schedules(self):
        path = self.root / "dataset_a" / "batches" / "msls.npy"
        np.save(path, np.load(path)[:1])
        with self.assertRaisesRegex(ValueError, "same number of iterations"):
            MegaLocDataset(self.config, PILToTensor())

    def test_missing_metadata_id_is_an_error(self):
        path = self.root / "dataset_a" / "meta" / "msls.parquet"
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
        self.write_config([{"name": "dataset_a", "subsets": ["msls"]}])
        dataset = MegaLocDataset(self.config, PILToTensor())
        with self.assertRaises(KeyError):
            dataset[0]

    def test_selected_subset_files_must_exist(self):
        (self.root / "dataset_a" / "meta" / "msls.parquet").unlink()
        with self.assertRaises(FileNotFoundError):
            MegaLocDataset(self.config, PILToTensor())


if __name__ == "__main__":
    unittest.main()
