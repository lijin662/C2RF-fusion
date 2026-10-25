"""Reconstruction pretraining followed by fusion training on aligned pairs."""
import csv
import json
import random
from pathlib import Path

import kornia
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Fusion_Data
from modules.losses import Fusionloss, cc
from options import TrainOptions
from utils.device import resolve_device
from utils.fusion import FusionModel, load_checkpoint, seed_everything


def main(opts):
    if not 0 < opts.warmup_epochs < opts.n_ep_Fu:
        raise ValueError('Require 0 < --warmup_epochs < --n_ep_Fu so both stages run.')
    if opts.batch_size < 2 or opts.crop_size < 32 or opts.max_steps < 0:
        raise ValueError('Require batch_size >= 2 (AFF global BatchNorm), crop_size >= 32 and max_steps >= 0.')
    if opts.display_freq < 1 or opts.model_save_freq < 1:
        raise ValueError('display_freq and model_save_freq must be positive.')
    seed_everything(opts.seed)
    device = resolve_device(opts.device, opts.gpu)
    destination = Path(opts.output_dir) / opts.dataset / opts.experiment
    if (destination / 'latest.pt').exists() and not opts.resume:
        raise FileExistsError('Use --resume or a new --experiment: %s' % destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = Fusion_Data(opts)
    if len(data) < opts.batch_size:
        raise ValueError('Need at least batch_size paired training images.')
    loader = DataLoader(data, batch_size=opts.batch_size, shuffle=True,
                        num_workers=opts.nThreads, pin_memory=device.type == 'cuda', drop_last=True)
    model = FusionModel().to(device)
    optimizers = {key: torch.optim.Adam(module.parameters(), lr=opts.lr)
                  for key, module in [('encoder', model.encoder), ('decoder', model.decoder), ('fusion', model.fusion)]}
    schedulers = {}
    for key, optimizer in optimizers.items():
        length = opts.warmup_epochs if key != 'fusion' else opts.n_ep_Fu - opts.warmup_epochs
        milestones = sorted(set([max(1, length // 3), max(1, 2 * length // 3)]))
        schedulers[key] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=0.2 if key != 'fusion' else 0.9)
    start_epoch = 0
    if opts.resume:
        saved = load_checkpoint(opts.resume, device)
        for key in ['dataset', 'warmup_epochs', 'n_ep_Fu', 'crop_size', 'lr', 'batch_size', 'seed']:
            if saved['options'][key] != getattr(opts, key):
                raise ValueError('Resume requires original --%s=%s' % (key, saved['options'][key]))
        model.load_state_dict(saved['model'])
        for key in optimizers:
            optimizers[key].load_state_dict(saved['optimizers'][key])
            schedulers[key].load_state_dict(saved['schedulers'][key])
        start_epoch = saved['epoch']
        rng = saved['rng']
        torch.set_rng_state(rng['torch'].cpu())
        random.setstate(rng['python'])
        np.random.set_state(('MT19937', np.array(rng['numpy_keys'], dtype=np.uint32),
                             rng['numpy_pos'], rng['numpy_gauss'], rng['numpy_cached']))
        if device.type == 'cuda' and 'cuda' in rng:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    (destination / 'config.json').write_text(json.dumps(vars(opts), indent=2) + '\n')
    reconstruction_ssim = kornia.losses.SSIMLoss(11, reduction='mean').to(device)
    fusion_loss = Fusionloss().to(device)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    print('Device: %s | dataset: %s | pairs: %d | output: %s' %
          (device, opts.dataset, len(data), destination), flush=True)

    for epoch in range(start_epoch, opts.n_ep_Fu):
        stage = 'reconstruction' if epoch < opts.warmup_epochs else 'fusion'
        model.train()
        if stage == 'fusion':
            model.freeze_autoencoder()
        active = ['encoder', 'decoder'] if stage == 'reconstruction' else ['fusion']
        total, count = 0., 0
        for step, (ir, vi, _, _, _) in enumerate(loader):
            ir, vi = ir.squeeze(1).to(device), vi.squeeze(1).to(device)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            if stage == 'reconstruction':
                _, i2, ib, id_, _, v2, vb, vd = model.encoder(ir, vi)
                recon_ir, recon_vi = model.decoder(ib, id_, i2), model.decoder(vb, vd, v2)
                decomposition = cc(vd, id_) - cc(vb, ib) + torch.tanh(l1(ib, vb))
                loss = 5 * (5 * reconstruction_ssim(vi, recon_vi) + mse(vi, recon_vi))
                loss = loss + 5 * (5 * reconstruction_ssim(ir, recon_ir) + mse(ir, recon_ir))
                for source, reconstructed in [(vi, recon_vi), (ir, recon_ir)]:
                    loss = loss + 10 * l1(kornia.filters.spatial_gradient(source),
                                         kornia.filters.spatial_gradient(reconstructed))
                loss = loss + 0.01 * decomposition
            else:
                loss, _, _, _ = fusion_loss(vi, ir, model(ir, vi))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss at epoch %d, batch %d' % (epoch + 1, step + 1))
            loss.backward()
            if stage == 'reconstruction':
                nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.01)
                nn.utils.clip_grad_norm_(model.decoder.parameters(), 0.01)
            for key in active:
                optimizers[key].step()
            total += loss.item()
            count += 1
            if step == 0 or (step + 1) % opts.display_freq == 0:
                print('Epoch %d/%d | %s | batch %d/%d | loss %.6f' %
                      (epoch + 1, opts.n_ep_Fu, stage, step + 1, len(loader), loss.item()), flush=True)
            if opts.max_steps and count >= opts.max_steps:
                break
        for key in active:
            schedulers[key].step()
        np_state = np.random.get_state()
        rng = {'torch': torch.get_rng_state(), 'python': random.getstate(),
               'numpy_keys': np_state[1].tolist(), 'numpy_pos': np_state[2],
               'numpy_gauss': np_state[3], 'numpy_cached': np_state[4]}
        if device.type == 'cuda':
            rng['cuda'] = torch.cuda.get_rng_state_all()
        checkpoint = {'epoch': epoch + 1, 'stage': stage, 'options': vars(opts),
                      'model': model.state_dict(), 'rng': rng,
                      'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                      'schedulers': {k: v.state_dict() for k, v in schedulers.items()}}
        temporary = destination / 'latest.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(destination / 'latest.pt')
        if (epoch + 1) % opts.model_save_freq == 0 or epoch + 1 == opts.n_ep_Fu:
            torch.save(checkpoint, destination / ('epoch_%04d.pt' % (epoch + 1)))
        if stage == 'fusion':
            for name, module in [('Encoder', model.encoder), ('Decoder', model.decoder), ('Fusion_layer', model.fusion)]:
                torch.save(module.state_dict(), destination / (name + '.pth'))
        logfile = destination / 'loss.csv'
        empty = not logfile.exists()
        with logfile.open('a', newline='') as handle:
            writer = csv.writer(handle)
            if empty:
                writer.writerow(['epoch', 'stage', 'loss', 'batches'])
            writer.writerow([epoch + 1, stage, total / count, count])
    print('Training finished: %s' % (destination / 'latest.pt'), flush=True)


if __name__ == '__main__':
    main(TrainOptions().parse())
