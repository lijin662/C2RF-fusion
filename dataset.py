import os
import sys
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import torch
from utils.utils import randflow, randflow_fixed, randrot, randfilp
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from natsort import natsorted
import kornia.utils as KU
from scipy import io
from utils.utils import RGB2YCrCb, YCbCr2RGB
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def dataset_folders(opts, split, require_warp=False):
    """Resolve built-in or user-defined paired-modality dataset folders."""
    presets = {
        'RoadScene': ('vi', 'ir', 'vi_warp'),
        'PET-MRI': ('PET', 'MRI', 'PET_warp'),
    }
    vis_name, ir_name, warp_name = presets.get(opts.dataset, ('vi', 'ir', 'vi_warp'))
    vis_name = opts.vis_dir or vis_name
    ir_name = opts.ir_dir or ir_name
    warp_name = opts.warp_dir or warp_name
    root = opts.train_dataroot if split == 'train' else opts.test_dataroot
    dataset_root = os.path.join(root, opts.dataset)
    folders = [os.path.join(dataset_root, vis_name), os.path.join(dataset_root, ir_name)]
    if require_warp:
        folders.append(os.path.join(dataset_root, warp_name))
    missing = [path for path in folders if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(
            'Dataset folders not found: %s. Use --train_dataroot/--test_dataroot and '
            '--vis_dir/--ir_dir/--warp_dir to describe your dataset.' % ', '.join(missing))
    return tuple(folders)


def paired_image_names(*folders):
    name_sets = []
    for folder in folders:
        names = {name for name in os.listdir(folder)
                 if os.path.isfile(os.path.join(folder, name))
                 and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS}
        name_sets.append(names)
    common = set.intersection(*name_sets)
    if not common:
        raise RuntimeError('No same-named image pairs found in: %s' % ', '.join(folders))
    if any(names != common for names in name_sets):
        counts = ', '.join('%s=%d' % (os.path.basename(folder), len(names))
                           for folder, names in zip(folders, name_sets))
        warnings.warn('Ignoring unpaired images (%s); paired=%d' % (counts, len(common)))
    return natsorted(common)


def imsave(img, filename):
    img = img.detach().squeeze().cpu().clamp(0, 1)
    img = (KU.tensor_to_image(img) * 255.).round().astype('uint8')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(filename, img):
        raise OSError('Failed to save image: %s' % filename)

class Fusion_Data(torch.utils.data.Dataset):
    def __init__(self, opts):
        super(Fusion_Data, self).__init__()
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'train')
        self.vis_list = paired_image_names(self.vis_folder, self.ir_folder)
        self.ir_list = self.vis_list
        self.crop = torchvision.transforms.RandomCrop(getattr(opts, 'crop_size', 256))

        # print('path of vi:', self.vis_folder)
        # print('path of ir:', self.ir_folder)
        # print('number of vi:', len(self.vis_list))
        # print('number of ir:', len(self.ir_list))
        # print('crop size:', (self.crop.size[0], self.crop.size[1]))

    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        vis_Y, vi_Cb, vi_Cr = RGB2YCrCb(vis)
        ir_Y, ir_Cb, ir_Cr = RGB2YCrCb(ir)
        vis_ir = torch.cat([vis_Y, ir_Y, vi_Cb, vi_Cr], dim=1)
        crop_size = self.crop.size[0]
        if min(vis_ir.shape[-2:]) < crop_size:
            vis_ir = TF.resize(vis_ir, crop_size)
        patch = self.crop(vis_ir)
        vis_Y, ir_Y, vi_Cb, vi_Cr = torch.split(patch, [1, 1, 1, 1], dim=1)
        return ir_Y, vis_Y,vi_Cb, vi_Cr,self.vis_list[index]

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class Registration_Data(torch.utils.data.Dataset):
    def __init__(self, opts, crop=lambda x: x):
        super(Registration_Data, self).__init__()
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'train')
        self.vis_list = paired_image_names(self.vis_folder, self.ir_folder)
        self.ir_list = self.vis_list
        self.crop = torchvision.transforms.RandomCrop(256)
        self.opts = opts
        # print('path of vi:', self.vis_folder)
        # print('path of ir:', self.ir_folder)
        # print('number of vi:', len(self.vis_list))
        # print('number of ir:', len(self.ir_list))
        # print('crop size:', (self.crop.size[0], self.crop.size[1]))

    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        assert os.path.basename(vis_path) == os.path.basename(
            ir_path), f"Mismatch ir:{os.path.basename(ir_path)} vi:{os.path.basename(vis_path)}."

        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_ir = torch.cat([vis, ir], dim=1)

        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)

        if self.epoch <= 1500:
            flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
            vis_ir_warped_1 = F.grid_sample(vis_ir, flow_1, align_corners=False, mode='bilinear')
            patch = torch.cat([vis_ir,vis_ir_warped_1, disp_1.permute(0, 3, 1, 2), ], dim=1)
            patch = self.crop(patch)
            vis, ir, vis_warped_1, ir_warped_1, disp_1 = torch.split(patch, [3, 3, 3, 3, 2], dim=1)
            h, w = vis_ir.shape[2], vis_ir.shape[3]
            scale = (torch.FloatTensor([w, h]).unsqueeze(0).unsqueeze(0) - 1) / (self.crop.size[0] * 1.0 - 1)
            disp_1 = disp_1.permute(0, 2, 3, 1) * scale
        else:
            flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
            flow_2, disp_2, _ = randflow(vis_ir, 3, 0.03, 0.3)
            flow_3, disp_3, _ = randflow(vis_ir, 1, 0.01, 0.1)
            vis_ir_warped_1 = F.grid_sample(vis_ir, flow_1, align_corners=False, mode='bilinear')
            vis_ir_warped_2 = F.grid_sample(vis_ir, flow_2, align_corners=False, mode='bilinear')
            vis_ir_warped_3 = F.grid_sample(vis_ir, flow_3, align_corners=False, mode='bilinear')

            patch = torch.cat([vis_ir,vis_ir_warped_1, vis_ir_warped_2, vis_ir_warped_3,
                            disp_1.permute(0, 3, 1, 2), disp_2.permute(0, 3, 1, 2), disp_3.permute(0, 3, 1, 2)
                            ], dim=1)
            patch = self.crop(patch)

            vis, ir, vis_warped_1, ir_warped_1, vis_warped_2, ir_warped_2, vis_warped_3, ir_warped_3, \
            disp_1, disp_2, disp_3 = torch.split(patch, [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2], dim=1)
            h, w = vis_ir.shape[2], vis_ir.shape[3]
            scale = (torch.FloatTensor([w, h]).unsqueeze(0).unsqueeze(0) - 1) / (self.crop.size[0] * 1.0 - 1)
            disp_1 = disp_1.permute(0, 2, 3, 1) * scale
            disp_2 = disp_2.permute(0, 2, 3, 1) * scale
            disp_3 = disp_3.permute(0, 2, 3, 1) * scale
        if self.epoch <= 1500:
            return vis, ir, vis_warped_1, ir_warped_1, disp_1
        else:
            return vis, ir, vis_warped_1, ir_warped_1, vis_warped_2, ir_warped_2, vis_warped_3, ir_warped_3, disp_1, disp_2, disp_3

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        # resize_transform = transforms.Resize((300, 460))
        # im_ts = resize_transform(im_ts)
        return im_ts

class Registration_Data_v2(torch.utils.data.Dataset):
    def __init__(self, opts, crop=lambda x: x):
        super(Registration_Data_v2, self).__init__()
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'train')
        self.vis_list = paired_image_names(self.vis_folder, self.ir_folder)
        self.ir_list = self.vis_list
        self.crop = torchvision.transforms.RandomCrop(256)
        self.opts = opts
        # print('path of vi:', self.vis_folder)
        # print('path of ir:', self.ir_folder)
        # print('number of vi:', len(self.vis_list))
        # print('number of ir:', len(self.ir_list))
        # print('crop size:', (self.crop.size[0], self.crop.size[1]))

    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        assert os.path.basename(vis_path) == os.path.basename(
            ir_path), f"Mismatch ir:{os.path.basename(ir_path)} vi:{os.path.basename(vis_path)}."

        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_ir = torch.cat([vis, ir], dim=1)

        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)

        if self.epoch <= 1500:
            flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
            vis_ir_warped_1 = F.grid_sample(vis_ir, flow_1, align_corners=False, mode='bilinear')
            patch = torch.cat([vis_ir,vis_ir_warped_1, disp_1.permute(0, 3, 1, 2), ], dim=1)
            patch = self.crop(patch)
            vis, ir, vis_warped_1, ir_warped_1, disp_1 = torch.split(patch, [3, 3, 3, 3, 2], dim=1)
            h, w = vis_ir.shape[2], vis_ir.shape[3]
            scale = (torch.FloatTensor([w, h]).unsqueeze(0).unsqueeze(0) - 1) / (self.crop.size[0] * 1.0 - 1)
            disp_1 = disp_1.permute(0, 2, 3, 1) * scale
        else:
            if self.epoch <= 1700:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_3, disp_3, _ = randflow(vis_ir, 4, 0.04, 0.5)
            elif 1700< self.epoch <= 1900:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 4, 0.04, 0.5)
                flow_3, disp_3, _ = randflow(vis_ir, 3, 0.04, 0.5)
            elif 1900< self.epoch <= 2100:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 3, 0.04, 0.5)
                flow_3, disp_3, _ = randflow(vis_ir, 3, 0.04, 0.5)
            elif 2100< self.epoch <= 2300:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 2.6, 0.04, 0.3)
                flow_3, disp_3, _ = randflow(vis_ir, 3, 0.04, 0.5)
            elif 2300< self.epoch <= 2500:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 3, 0.03, 0.5)
                flow_3, disp_3, _ = randflow(vis_ir, 2.4, 0.03, 0.5)
            elif 2500< self.epoch <= 2800:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 2.4, 0.03, 0.5)
                flow_3, disp_3, _ = randflow(vis_ir, 2, 0.025, 0.3)
            elif 2800< self.epoch <= 3100:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 2, 0.02, 0.2)
                flow_3, disp_3, _ = randflow(vis_ir, 1, 0.015, 0.1)
            elif 3100< self.epoch:
                flow_1, disp_1, _ = randflow(vis_ir, 5, 0.05, 0.5)
                flow_2, disp_2, _ = randflow(vis_ir, 1, 0.02, 0.2)
                flow_3, disp_3, _ = randflow(vis_ir, 1, 0.015, 0.1)

            vis_ir_warped_1 = F.grid_sample(vis_ir, flow_1, align_corners=False, mode='bilinear')
            vis_ir_warped_2 = F.grid_sample(vis_ir, flow_2, align_corners=False, mode='bilinear')
            vis_ir_warped_3 = F.grid_sample(vis_ir, flow_3, align_corners=False, mode='bilinear')

            patch = torch.cat([vis_ir,vis_ir_warped_1, vis_ir_warped_2, vis_ir_warped_3,
                            disp_1.permute(0, 3, 1, 2), disp_2.permute(0, 3, 1, 2), disp_3.permute(0, 3, 1, 2)
                            ], dim=1)
            patch = self.crop(patch)

            vis, ir, vis_warped_1, ir_warped_1, vis_warped_2, ir_warped_2, vis_warped_3, ir_warped_3, \
            disp_1, disp_2, disp_3 = torch.split(patch, [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2], dim=1)
            h, w = vis_ir.shape[2], vis_ir.shape[3]
            scale = (torch.FloatTensor([w, h]).unsqueeze(0).unsqueeze(0) - 1) / (self.crop.size[0] * 1.0 - 1)
            disp_1 = disp_1.permute(0, 2, 3, 1) * scale
            disp_2 = disp_2.permute(0, 2, 3, 1) * scale
            disp_3 = disp_3.permute(0, 2, 3, 1) * scale
        if self.epoch <= 1500:
            return vis, ir, vis_warped_1, ir_warped_1, disp_1
        else:
            return vis, ir, vis_warped_1, ir_warped_1, vis_warped_2, ir_warped_2, vis_warped_3, ir_warped_3, disp_1, disp_2, disp_3

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        # resize_transform = transforms.Resize((300, 460))
        # im_ts = resize_transform(im_ts)
        return im_ts


class TestData(torch.utils.data.Dataset):
    def __init__(self, opts):
        super(TestData, self).__init__()
        self.vis_folder, self.ir_folder, self.vis_warp_folder = dataset_folders(
            opts, 'test', require_warp=True)
        self.ir_list = paired_image_names(
            self.vis_folder, self.ir_folder, self.vis_warp_folder)

        # print('path of vi:', self.vis_folder)
        # print('path of ir:', self.ir_folder)

    def __getitem__(self, index):
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        vi_warp_path = os.path.join(self.vis_warp_folder, image_name)
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        vi_warp = self.imread(path=vi_warp_path)
        return ir, vis, vi_warp, image_name

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        im_ts = KU.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.).float()
        im_ts = im_ts.unsqueeze(0)
        return im_ts


class FusionTestData(torch.utils.data.Dataset):
    """Aligned optical/IR or optical/SAR pairs; no registration weights/warps needed."""
    def __init__(self, opts):
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'test')
        self.names = paired_image_names(self.vis_folder, self.ir_folder)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        vi = Fusion_Data.imread(os.path.join(self.vis_folder, name))
        ir = Fusion_Data.imread(os.path.join(self.ir_folder, name))
        if vi.shape != ir.shape:
            raise ValueError('Image sizes differ for %s: vi=%s ir=%s' % (name, vi.shape, ir.shape))
        return ir, vi, name

class Generate_Reg_Data(torch.utils.data.Dataset):
    def __init__(self, opts, crop=False):
        super(Generate_Reg_Data, self).__init__()
        self.vis_folder, self.ir_folder = dataset_folders(opts, 'test')
        dataset_root = os.path.join(opts.test_dataroot, opts.dataset)
        self.flow_dir = os.path.join(dataset_root, 'flow')
        self.warp_dir = os.path.join(dataset_root, opts.warp_dir or
                                     ('PET_warp' if opts.dataset == 'PET-MRI' else 'vi_warp'))

        if not os.path.exists(self.flow_dir):
            os.makedirs(self.flow_dir)
        if not os.path.exists(self.warp_dir):
            os.makedirs(self.warp_dir)

        # self.crop = torchvision.transforms.RandomCrop(256)
        self.opts = opts
        self.vis_list = paired_image_names(self.vis_folder, self.ir_folder)
        self.ir_list = self.vis_list

        # print('path of vi:', self.vis_folder)
        # print('path of ir:', self.ir_folder)
        # print('number of vi:', len(self.vis_list))
        # print('number of ir:', len(self.ir_list))
        # # print('crop size:', (self.crop.size[0], self.crop.size[1]))

    def __getitem__(self, index):
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_ir = torch.cat([vis, ir], dim=1)

        flow, disp, _ = randflow(vis_ir, 5, 0.05, 0.5)
        vis_ir_warped = F.grid_sample(vis_ir, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vis_ir, vis_ir_warped, disp.permute(0, 3, 1, 2)], dim=1)
        vis, ir, vis_warped, ir_warped, disp = torch.split(patch, [3, 3, 3, 3, 2], dim=1)

        imsave(vis_warped.squeeze(0), os.path.join(self.warp_dir, self.vis_list[index]))
        stem = os.path.splitext(self.vis_list[index])[0]
        io.savemat(os.path.join(self.flow_dir, stem + '.mat'), {'flow': flow.detach().numpy()})

        return ir, vis, ir_warped, vis_warped, disp

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        # resize_transform = transforms.Resize((300, 460))
        # im_ts = resize_transform(im_ts)
        return im_ts
