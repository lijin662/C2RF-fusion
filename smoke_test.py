"""Exercise real dataset loading and model inference without pretrained weights."""
import json
from pathlib import Path
import time

import torch

from dataset import Fusion_Data, FusionTestData, Generate_Reg_Data, Registration_Data, TestData
from model import C2RF
from options import TrainOptions
from utils.device import resolve_device
from utils.fusion import FusionModel
from utils.utils import RGB2YCrCb


if __name__ == '__main__':
    opts = TrainOptions().parse()
    torch.manual_seed(0)
    device = resolve_device(opts.device, opts.gpu)
    report = {'dataset': opts.dataset, 'device': str(device),
              'weights': 'random initialization; execution check only'}
    if opts.phase == 'train':
        fusion = Fusion_Data(opts)
        registration = Registration_Data(opts)
        registration.set_epoch(0)
        report['fusion_shapes'] = [list(x.shape) for x in fusion[0][:-1]]
        report['registration_shapes'] = [list(x.shape) for x in registration[0]]
        print(json.dumps(report, indent=2))
    else:
        if opts.generate_test_warp:
            generator = Generate_Reg_Data(opts)
            for index in range(len(generator)):
                generator[index]
        data = TestData(opts) if opts.generate_test_warp else FusionTestData(opts)
        if opts.generate_test_warp:
            ir, vi, warp, name = data[0]
            model = C2RF(training_components=False).eval().to(device)
        else:
            ir, vi, name = data[0]
            model = FusionModel().eval().to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            if opts.generate_test_warp:
                outputs = model(warp.to(device), ir.to(device), vi.to(device))
            else:
                outputs = (model(RGB2YCrCb(ir.to(device))[0], RGB2YCrCb(vi.to(device))[0]),)
        if not all(torch.isfinite(value).all() for value in outputs):
            raise RuntimeError('Non-finite model output')
        report.update({'pairs': len(data), 'image': name,
                       'input_shape': list(ir.shape),
                       'output_shapes': [list(value.shape) for value in outputs],
                       'elapsed_seconds': round(time.perf_counter() - start, 3)})
        destination = Path(opts.output_dir) / opts.dataset / opts.experiment / 'smoke'
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    print('PASS')
