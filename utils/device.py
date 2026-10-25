import torch


def resolve_device(name='auto', gpu=0):
    """Return a usable torch.device with a clear error for unavailable requests."""
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda', gpu)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available. Use --device cpu or --device auto.')
        return torch.device('cuda', gpu)
    if name == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('MPS was requested but is not available. Use --device cpu or --device auto.')
        return torch.device('mps')
    return torch.device('cpu')
