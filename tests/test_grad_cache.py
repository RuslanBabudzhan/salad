from copy import deepcopy
import unittest

import torch
from torch import nn

from utils.grad_cache import grad_cache_backward


def pair_loss(descriptors, labels):
    similarities = descriptors @ descriptors.T
    targets = (labels[:, None] == labels[None, :]).to(similarities)
    return (similarities - targets).square().mean()


class GradCacheTests(unittest.TestCase):
    def check_gradients(self, dropout, chunk_size, dtype=None):
        torch.manual_seed(7)
        reference = nn.Sequential(nn.Linear(5, 12), nn.Tanh(), nn.Dropout(dropout), nn.Linear(12, 6))
        cached = deepcopy(reference)
        images = torch.randn(11, 5)
        labels = torch.arange(11) // 3
        rng = torch.get_rng_state()
        with torch.autocast("cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None, cache_enabled=False):
            descriptors = torch.cat([reference(x) for x in images.split(chunk_size)]).float()
        expected_rng = torch.get_rng_state()
        expected_loss = pair_loss(descriptors, labels)
        expected_loss.backward()

        torch.set_rng_state(rng)
        calls = []

        def loss_fn(descriptors, labels):
            calls.append(len(descriptors))
            return pair_loss(descriptors, labels)

        with torch.autocast("cpu", dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            actual_loss = grad_cache_backward(cached, images, labels, loss_fn, chunk_size, torch.Tensor.backward)
        self.assertEqual(calls, [11])  # All positives/negatives participate in one loss.
        torch.testing.assert_close(actual_loss, expected_loss)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for actual, expected in zip(cached.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_dropout_replay_and_partial_chunk(self):
        self.check_gradients(dropout=0.4, chunk_size=4)

    def test_chunk_size_one(self):
        self.check_gradients(dropout=0, chunk_size=1)

    def test_bfloat16_autocast(self):
        self.check_gradients(dropout=0.4, chunk_size=4, dtype=torch.bfloat16)

    def test_scaled_backward_is_applied_once(self):
        torch.manual_seed(8)
        model = nn.Linear(5, 6)
        reference = deepcopy(model)
        images, labels = torch.randn(8, 5), torch.arange(8) // 2
        scale = 128
        pair_loss(reference(images), labels).backward()
        grad_cache_backward(model, images, labels, pair_loss, 3, lambda loss: (scale * loss).backward())
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad / scale, expected.grad)


if __name__ == "__main__":
    unittest.main()
