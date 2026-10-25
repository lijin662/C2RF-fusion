"""Validate paired C2RF dataset folders without loading a neural network."""
from dataset import dataset_folders, paired_image_names
from options import TrainOptions


if __name__ == '__main__':
    opts = TrainOptions().parse()
    split = 'train' if opts.phase == 'train' else 'test'
    require_warp = split == 'test' and not opts.generate_test_warp
    folders = dataset_folders(opts, split, require_warp=require_warp)
    names = paired_image_names(*folders)
    print('Dataset: %s' % opts.dataset)
    print('Split: %s' % split)
    for folder in folders:
        print('Folder: %s' % folder)
    print('Paired images: %d' % len(names))
    print('First image: %s' % names[0])
