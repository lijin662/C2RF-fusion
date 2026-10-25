"""Shared fusion model and reproducibility utilities."""
import random
import numpy as np
import torch
from torch import nn
from modules.FusionNet import AE_Encoder, AE_Decoder, Fusion_layer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AE_Encoder()
        self.decoder = AE_Decoder()
        self.fusion = Fusion_layer()

    def forward(self, ir, vi):
        _, i2, ib, id_, _, v2, vb, vd = self.encoder(ir, vi)
        return self.decoder(*self.fusion(ib, vb, id_, vd, i2, v2))

    def freeze_autoencoder(self):
        self.encoder.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)
