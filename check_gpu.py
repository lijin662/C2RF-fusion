"""Execute CUDA model forward/backward before starting a long server experiment."""
import argparse
import json
import platform

import torch

from modules.losses import Fusionloss
from utils.fusion import FusionModel


def check(gpu=0, precision='fp32'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check nvidia-smi, driver/container GPU access and the PyTorch wheel.')
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    runtime = tuple(int(value) for value in torch.version.cuda.split('.')[:2])
    if capability[0] >= 10 and runtime < (12, 8):
        raise RuntimeError('Blackwell requires CUDA 12.8+ PyTorch wheels. Run scripts/setup_5090.sh.')
    if precision == 'bf16' and (capability[0] < 8 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Native BF16 is unavailable on this GPU.')
    model = FusionModel().to(device).train()
    loss_fn = Fusionloss().to(device)
    ir, vi = (torch.rand(2, 1, 32, 32, device=device) for _ in range(2))
    # This exercises project convolution/attention kernels, not just CUDA enumeration.
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
        fused = model(ir, vi)
    loss = loss_fn(vi, ir, fused.float())[0]
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                      for p in model.parameters()):
        raise RuntimeError('Non-finite CUDA loss or gradients.')
    result = {'status': 'PASS', 'gpu': torch.cuda.get_device_name(device),
              'capability': list(capability), 'torch': str(torch.__version__),
              'cuda_runtime': torch.version.cuda, 'python': platform.python_version(),
              'compiled_architectures': torch.cuda.get_arch_list(), 'precision': precision,
              'loss': loss.item()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    arguments = parser.parse_args()
    check(arguments.gpu, arguments.precision)
