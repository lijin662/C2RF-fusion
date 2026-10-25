"""Options shared by the two-dataset fusion workflow."""
import argparse


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        add = self.parser.add_argument
        add('--dataset', default='RoadScene', choices=['RoadScene', 'OSDataset', 'OSDataset512'])
        add('--train_dataroot', default='./dataset/train')
        add('--test_dataroot', default='./dataset/test')
        add('--vis_dir', default=None, help='visible/optical image subdirectory')
        add('--ir_dir', default=None, help='infrared/SAR image subdirectory')
        add('--phase', default='train', choices=['train', 'test'])
        add('--batch_size', type=int, default=2)
        add('--nThreads', type=int, default=4)
        add('--crop_size', type=int, default=256)
        add('--n_ep_Fu', type=int, default=150)
        add('--warmup_epochs', type=int, default=50)
        add('--lr', type=float, default=1e-4)
        add('--seed', type=int, default=42)
        add('--max_steps', type=int, default=0, help='batches per epoch; 0 means all')
        add('--display_freq', type=int, default=50)
        add('--model_save_freq', type=int, default=50)
        add('--resume', default=None)
        add('--gpu', type=int, default=0)
        add('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
        add('--precision', default='fp32', choices=['fp32', 'bf16'],
            help='bf16 uses CUDA autocast for model forwards; losses remain float32')
        add('--checkpoint_dir', default='./checkpoint', help='directory of per-dataset fusion component weights')
        add('--fusion_checkpoint', default=None)
        add('--output_dir', default='./result')
        add('--experiment', default='baseline')
        add('--fused_dir', default=None)

    def parse(self):
        return self.parser.parse_args()
