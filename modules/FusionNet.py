import torch
from torch import nn
from .layers_fusion import (Inception_ResidualBlock, ChannelAttention, SpatialAttention,
                            ConvBnLeakyRelu2d, AFF, CrossAttention)


class AE_Encoder(nn.Module):
    def __init__(self, channel=64):
        super(AE_Encoder, self).__init__()
        self.inception_res_1 = Inception_ResidualBlock(1, channel // 2)
        self.inception_res_2 = Inception_ResidualBlock(channel // 2, channel)
        self.inception_res_3 = Inception_ResidualBlock(channel, channel)

        # common
        self.CATT_ir = ChannelAttention(channel)
        self.SATT_ir = SpatialAttention()
        self.CATT_vi = ChannelAttention(channel)
        self.SATT_vi = SpatialAttention()

        # unique
        self.conv0_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # output
        self.conv_fB_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fB_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)

    def forward(self, ir, vi):
        f0_ir = self.inception_res_1(ir)
        f1_ir = self.inception_res_2(f0_ir)
        f2_ir = self.inception_res_3(f1_ir)

        f0_vi = self.inception_res_1(vi)
        f1_vi = self.inception_res_2(f0_vi)
        f2_vi = self.inception_res_3(f1_vi)

        fB_ir_CATT = f2_ir * self.CATT_ir(f2_ir)
        fB_ir_SATT = f2_ir * self.SATT_ir(f2_ir)
        fB_ir = self.conv_fB_ir(torch.cat([fB_ir_CATT, fB_ir_SATT], 1))

        fB_vi_CATT = f2_vi * self.CATT_vi(f2_vi)
        fB_vi_SATT = f2_vi * self.SATT_vi(f2_vi)
        fB_vi = self.conv_fB_vi(torch.cat([fB_vi_CATT, fB_vi_SATT], 1))

        fD_ir = self.conv0_D1(self.conv1_D1(f2_ir))
        fD_vi = self.conv0_D2(self.conv1_D2(f2_vi))

        return f1_ir, f2_ir, fB_ir, fD_ir, f1_vi, f2_vi, fB_vi, fD_vi


class AE_Decoder(nn.Module):
    def __init__(self, channel=64):
        super(AE_Decoder, self).__init__()
        self.cov5 = nn.Sequential(nn.Conv2d(channel * 3, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cov6 = nn.Sequential(nn.Conv2d(channel * 1, channel, 3, padding=1), nn.BatchNorm2d(channel), nn.PReLU())
        self.cov7 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channel * 1, channel // 2, 3, padding=0),
                                  nn.Conv2d(channel // 2, 1, 3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, feature_B, feature_D, feature_cross):
        Output1 = self.cov5(torch.cat([feature_B, feature_D, feature_cross], 1))
        Output2 = self.cov6(Output1)
        Output3 = self.cov7(Output2)
        return Output3


class Fusion_layer(nn.Module):
    def __init__(self, channel=64):
        super(Fusion_layer, self).__init__()
        self.conv0_ir_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_vi_B = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)

        self.conv0_ir_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_vi_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_vi_D = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_ir_vi_D = AFF()

        self.Cross_ir_vi = CrossAttention(channel)

    def forward(self, ir_B, vi_B, ir_D, vi_D, ir_2, vi_2):
        # common
        conv0_ir_B = self.conv0_ir_B(ir_B) + ir_B
        conv0_vi_B = self.conv0_vi_B(vi_B) + vi_B
        conv1_ir_B = self.conv1_ir_B(conv0_ir_B)
        conv1_vi_B = self.conv1_vi_B(conv0_vi_B)
        ir_vi_B = self.conv1_ir_vi_B(conv1_ir_B + conv1_vi_B)

        # unique
        conv0_ir_D = self.conv0_ir_D(ir_D) + ir_D
        conv0_vi_D = self.conv0_vi_D(vi_D) + vi_D
        conv1_ir_D = self.conv1_ir_D(conv0_ir_D)
        conv1_vi_D = self.conv1_vi_D(conv0_vi_D)
        ir_vi_D = self.conv1_ir_vi_D(conv1_ir_D, conv1_vi_D)

        cross_att_ir, cross_att_vi = self.Cross_ir_vi(ir_2, vi_2)
        ir_crossATT = ir_2 * cross_att_ir
        vi_crossATT = vi_2 * cross_att_vi
        ir_vi_cross = ir_crossATT + vi_crossATT

        return ir_vi_B, ir_vi_D, ir_vi_cross
