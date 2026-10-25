"""Fuse aligned pairs with trained weights (no registration network required)."""
import json
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import FusionTestData, imsave
from options import TrainOptions
from utils.device import resolve_device
from utils.fusion import FusionModel, load_checkpoint
from utils.utils import RGB2YCrCb, YCbCr2RGB


def main(opts):
    device = resolve_device(opts.device, opts.gpu)
    model = FusionModel().to(device)
    if opts.fusion_checkpoint:
        checkpoint = load_checkpoint(opts.fusion_checkpoint, device)
        if checkpoint['stage'] != 'fusion':
            raise ValueError('Checkpoint only contains reconstruction training; complete fusion training first.')
        if checkpoint['options']['dataset'] != opts.dataset:
            raise ValueError('Checkpoint dataset differs; use weights trained for %s.' % opts.dataset)
        model.load_state_dict(checkpoint['model'])
        weights = str(Path(opts.fusion_checkpoint).resolve())
    else:
        folder = Path(opts.checkpoint_dir) / opts.dataset
        for name, module in [('Encoder', model.encoder), ('Decoder', model.decoder), ('Fusion_layer', model.fusion)]:
            module.load_state_dict(load_checkpoint(folder / (name + '.pth'), device))
        weights = str(folder.resolve())
    model.eval()
    data = FusionTestData(opts)
    output = Path(opts.output_dir) / opts.dataset / opts.experiment / 'fused'
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for ir, vi, name in tqdm(data, total=len(data)):
            ir, vi = ir.to(device), vi.to(device)
            ir_y, _, _ = RGB2YCrCb(ir)
            vi_y, cb, cr = RGB2YCrCb(vi)
            fused = model(ir_y, vi_y)
            if not torch.isfinite(fused).all():
                raise RuntimeError('Non-finite fusion output for %s' % name)
            rgb = YCbCr2RGB(fused, cb, cr)
            imsave(rgb, str(output / (Path(name).stem + '.png')))
    (output.parent / 'inference.json').write_text(json.dumps(
        {'dataset': opts.dataset, 'pairs': len(data), 'weights': weights,
         'test_dataroot': opts.test_dataroot, 'device': str(device)}, indent=2) + '\n')
    print('Saved %d fused images to %s' % (len(data), output))


if __name__ == '__main__':
    main(TrainOptions().parse())
