"""Gradient caching for one complete metric-learning sub-batch."""

import torch


def grad_cache_backward(model, images, labels, loss_fn, chunk_size, backward):
    """Cache d(loss)/d(descriptor), then replay chunks through ``backward``.

    Inputs may stay on CPU. The caller owns autocast, gradient scaling and the
    optimizer step. RNG replay preserves dropout; the model must not update
    running statistics (e.g. training BatchNorm) during forward.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []

    def forward(chunk):
        # Cached autocast weights from the graph-free pass must not reach replay.
        with torch.autocast(device.type, enabled=torch.is_autocast_enabled(device.type), cache_enabled=False):
            return model(chunk.to(device, non_blocking=True)).float()

    chunks = images.split(chunk_size)
    states, representations = [], []
    with torch.no_grad():
        for chunk in chunks:
            states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]))
            representations.append(forward(chunk))

    descriptors = torch.cat(representations).requires_grad_()
    del representations
    # Descriptor gradients remain unscaled; Lightning scales each replay once.
    with torch.autocast(device_type=device.type, enabled=False):
        loss = loss_fn(descriptors, labels.to(device, non_blocking=True))
        gradient, = torch.autograd.grad(loss, descriptors)
    value = loss.detach()
    del loss, descriptors

    for chunk, grad, (cpu_state, cuda_states) in zip(chunks, gradient.split(chunk_size), states):
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            for d, state in zip(devices, cuda_states):
                torch.cuda.set_rng_state(state, d)
            replay = forward(chunk)
        with torch.autocast(device.type, enabled=False):
            backward((replay * grad).sum())
    return value
