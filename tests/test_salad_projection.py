import unittest

import torch
from torch.nn import functional as F

from models.aggregators.salad import SALAD


class SALADProjectionTests(unittest.TestCase):
    def test_projection_follows_normalized_salad(self):
        torch.manual_seed(7)
        config = dict(num_channels=8, num_clusters=2, cluster_dim=3, token_dim=4, dropout=0)
        legacy = SALAD(**config)
        projected = SALAD(**config, output_dim=6)
        # Original checkpoints contain no projection parameters and still load strictly.
        self.assertFalse(any("projection" in key for key in legacy.state_dict()))
        SALAD(**config).load_state_dict(legacy.state_dict(), strict=True)
        missing, unexpected = projected.load_state_dict(legacy.state_dict(), strict=False)
        self.assertEqual(set(missing), {"projection.weight", "projection.bias"})
        self.assertEqual(unexpected, [])
        features = torch.randn(2, 8, 3, 3), torch.randn(2, 8)
        with torch.no_grad():
            original = legacy(features)
            actual = projected(features)
            expected = F.normalize(projected.projection(original), dim=-1)
        self.assertEqual(original.shape, (2, 10))
        self.assertEqual(actual.shape, (2, 6))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(2))

    def test_megaloc_dimensions(self):
        # Check the full head without allocating its 140M projection weights.
        with torch.device("meta"):
            head = SALAD(num_channels=768, num_clusters=64, cluster_dim=256,
                         token_dim=256, output_dim=8448)
            self.assertEqual(head.projection.weight.shape, (8448, 16640))
            self.assertEqual(head.projection.bias.shape, (8448,))
            descriptors = head((torch.randn(2, 768, 14, 14), torch.randn(2, 768)))
            self.assertEqual(descriptors.shape, (2, 8448))


if __name__ == "__main__":
    unittest.main()
