"""Reproducible grayscale EN/SD/SF/MI diagnostics; no fused ground truth required."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import dataset_folders, paired_image_names
from options import TrainOptions


def entropy(image):
    counts = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / image.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def mutual_information(left, right):
    counts = np.bincount(left.ravel().astype(np.int64) * 256 + right.ravel(),
                         minlength=65536).reshape(256, 256)
    joint = counts / counts.sum()
    independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float((joint[nonzero] * np.log2(joint[nonzero] / independent[nonzero])).sum())


def read_gray(path):
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.uint8)


def main(opts):
    vi_folder, ir_folder = dataset_folders(opts, 'test')
    names = paired_image_names(vi_folder, ir_folder)
    experiment = Path(opts.output_dir) / opts.dataset / opts.experiment
    folder = Path(opts.fused_dir) if opts.fused_dir else experiment / 'fused'
    rows = []
    for name in names:
        fused = read_gray(folder / (Path(name).stem + '.png'))
        vi, ir = read_gray(Path(vi_folder) / name), read_gray(Path(ir_folder) / name)
        if not (fused.shape == vi.shape == ir.shape):
            raise ValueError('Image size mismatch for %s' % name)
        values = fused.astype(np.float64)
        spatial_frequency = np.sqrt(np.mean(np.diff(values, axis=0) ** 2) +
                                    np.mean(np.diff(values, axis=1) ** 2))
        rows.append({'name': name, 'EN': entropy(fused), 'SD': float(values.std()),
                     'SF': float(spatial_frequency),
                     'MI': mutual_information(fused, vi) + mutual_information(fused, ir)})
    report = {'dataset': opts.dataset, 'pairs': len(rows), 'fused_dir': str(folder),
              'mean': {key: float(np.mean([row[key] for row in rows])) for key in ['EN', 'SD', 'SF', 'MI']},
              'definitions': '8-bit PIL L grayscale; EN/MI in bits, SD/SF on 0..255; MI=MI(F,VI)+MI(F,IR).'}
    folder.mkdir(parents=True, exist_ok=True)
    with (folder.parent / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'EN', 'SD', 'SF', 'MI'])
        writer.writeheader()
        writer.writerows(rows)
    (folder.parent / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(TrainOptions().parse())
