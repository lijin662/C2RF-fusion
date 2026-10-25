"""Aligned image pairs for RoadScene (visible/IR) and OSDataset (optical/SAR)."""
from pathlib import Path
import warnings

from PIL import Image
from natsort import natsorted
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.utils import RGB2YCrCb


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def dataset_folders(opts, split):
    root = opts.train_dataroot if split == 'train' else opts.test_dataroot
    base = Path(root) / opts.dataset
    folders = (base / (opts.vis_dir or 'vi'), base / (opts.ir_dir or 'ir'))
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        raise FileNotFoundError('Missing dataset folders: %s' % ', '.join(missing))
    return tuple(map(str, folders))


def paired_image_names(*folders):
    name_sets = [{path.name for path in Path(folder).iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
                 for folder in folders]
    common = set.intersection(*name_sets)
    if not common:
        raise ValueError('No same-named image pairs found in: %s' % ', '.join(folders))
    if any(names != common for names in name_sets):
        warnings.warn('Ignoring unpaired images; paired=%d in %s' % (len(common), folders))
    return natsorted(common)


def read_pair(vis_folder, ir_folder, name):
    tensors = []
    for folder in (ir_folder, vis_folder):
        with Image.open(Path(folder) / name) as image:
            tensors.append(TF.to_tensor(image.convert('RGB')).unsqueeze(0))
    ir, vi = tensors
    if ir.shape != vi.shape:
        raise ValueError('Image sizes differ for %s: vi=%s ir=%s' % (name, vi.shape, ir.shape))
    return ir, vi


def imsave(image, filename):
    pixels = image.detach().squeeze(0).cpu().clamp(0, 1).mul(255).round().byte()
    if pixels.shape[0] == 1:
        array = pixels[0].numpy()
    else:
        array = pixels.permute(1, 2, 0).numpy()
    Image.fromarray(array).save(filename)


class Fusion_Data(Dataset):
    def __init__(self, opts):
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'train')
        self.names = paired_image_names(self.vis_folder, self.ir_folder)
        self.crop = transforms.RandomCrop(opts.crop_size)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        ir, vi = read_pair(self.vis_folder, self.ir_folder, name)
        ir_y, _, _ = RGB2YCrCb(ir)
        vi_y, cb, cr = RGB2YCrCb(vi)
        paired = torch.cat([ir_y, vi_y, cb, cr], dim=1)
        if min(paired.shape[-2:]) < self.crop.size[0]:
            paired = TF.resize(paired, self.crop.size[0])
        ir_y, vi_y, cb, cr = torch.split(self.crop(paired), 1, dim=1)
        return ir_y, vi_y, cb, cr, name


class FusionTestData(Dataset):
    def __init__(self, opts):
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'test')
        self.names = paired_image_names(self.vis_folder, self.ir_folder)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        ir, vi = read_pair(self.vis_folder, self.ir_folder, name)
        return ir, vi, name
