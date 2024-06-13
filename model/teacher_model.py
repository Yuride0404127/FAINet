import torch
import torch.nn as nn
import torch.nn.functional as F

# from backbone.TransXNet.models.transxnet import transxnet_b, transxnet_s, transxnet_t
from backbone.Shunted_Transformer.SSA import shunted_t, shunted_b, shunted_s
# from backbone.P2T.p2t import p2t_large, p2t_base, p2t_tiny, p2t_small
# from KD_model_3.fft import myfft
from backbone.Segformer.backbone import *
from pytorch_wavelets import *

from einops import rearrange

stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                # nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)

class ACA_v3(nn.Module):
    def __init__(self, in_channel, out_channel, hw):
        super(ACA_v3, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv_inter = DSConv3x3(in_channel, out_channel)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv_compress = DSConv3x3(3 * out_channel, out_channel)

        self.Channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            DSConv3x3(out_channel, out_channel // hw),
            nn.ReLU(),
            DSConv3x3(out_channel // hw, out_channel),
            nn.Sigmoid()
        )
        self.conv_cat = DSConv3x3(2 * out_channel, out_channel)

    def forward(self, x_up, x_midd, x_down):
        if (self.in_channel == self.out_channel):
            x_midd = self.conv_inter(x_midd)
        else:
            x_midd = self.pool2(self.conv_inter(x_midd))

        x_cat = torch.cat([x_up, x_midd, x_down], dim=1)

        x_compress = self.conv_compress(x_cat)
        # print("x_compress", x_compress.shape)
        x_channel_weight = self.Channel_weight(x_compress)
        x_up_weighted = x_up * x_channel_weight
        x_down_weighted = x_down * x_channel_weight
        x_add = x_down_weighted + x_up_weighted
        x_cat = torch.cat([x_add, x_midd], dim=1)
        out = self.conv_cat(x_cat)
        return out

class FDW_v2(nn.Module):
    def __init__(self, in_channel):
        super(FDW_v2, self).__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.conv_fl_up = DSConv3x3(in_channel, in_channel)
        self.conv_fl_down = DSConv3x3(in_channel, in_channel)

        self.conv_cat_fl = DSConv3x3(2 * in_channel, in_channel)


        self.conv_out = DSConv3x3(in_channel, in_channel)

        self.conv_up = DSConv3x3(in_channel, in_channel)
        self.conv_down = DSConv3x3(in_channel, in_channel)

        self.conv_FFT_up = DSConv3x3(in_channel, in_channel)
        self.conv_FFT_down = DSConv3x3(in_channel, in_channel)


    def forward(self, x_up, x_down):
        FL_up, FH_up = self.DWT(x_up)
        FL_down, FH_down = self.DWT(x_down)

        FL_up = self.conv_fl_up(FL_up)
        FL_down = self.conv_fl_down(FL_down)

        FL_cat = torch.cat([FL_up, FL_down], dim=1)
        FL_cat = self.conv_cat_fl(FL_cat)

        FFT_out_up = self.IWT((FL_cat, FH_up))
        FFT_out_down = self.IWT((FL_cat, FH_down))
        # print("FFT_out", FFT_out.shape)
        Muti_up = self.conv_FFT_up(FFT_out_up) * x_up
        Muti_down = self.conv_FFT_down(FFT_out_down) * x_down

        out = self.conv_out(Muti_up + Muti_down)

        return out

class KD_model_3_teacher(nn.Module):
    def __init__(self):
        super(KD_model_3_teacher, self).__init__()
        # Backbone model
        self.rgb = shunted_b(pretrained=True)
        self.depth = shunted_b(pretrained=True)


        self.Head1 = SalHead(stage1_channel)
        self.Head2 = SalHead(stage2_channel)
        self.Head3 = SalHead(stage3_channel)
        self.Head4 = SalHead(stage4_channel)

        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.aca1 = ACA_v3(stage1_channel, stage1_channel, 64)
        self.aca2 = ACA_v3(stage1_channel, stage2_channel, 32)
        self.aca3 = ACA_v3(stage2_channel, stage3_channel, 16)
        self.aca4 = ACA_v3(stage3_channel, stage4_channel, 8)

        self.fdw1 = FDW_v2(stage1_channel)
        self.fdw2 = FDW_v2(stage2_channel)
        self.fdw3 = FDW_v2(stage3_channel)
        self.fdw4 = FDW_v2(stage4_channel)

        self.fdw2_conv = DSConv3x3(stage2_channel, stage1_channel)
        self.fdw3_conv = DSConv3x3(stage3_channel, stage2_channel)
        self.fdw4_conv = DSConv3x3(stage4_channel, stage3_channel)

        self.stage_1_cat_conv = DSConv3x3(2 * stage1_channel, stage1_channel)


    def forward(self, x_rgb, x_depth):
        rgb_list = self.rgb(x_rgb)
        rgb_1 = rgb_list[0]
        rgb_2 = rgb_list[1]
        rgb_3 = rgb_list[2]
        rgb_4 = rgb_list[3]

        x_depth = torch.cat([x_depth, x_depth, x_depth], dim=1)
        depth_list = self.depth(x_depth)
        depth_1 = depth_list[0]
        depth_2 = depth_list[1]
        depth_3 = depth_list[2]
        depth_4 = depth_list[3]

        stage_1_cat = self.stage_1_cat_conv(torch.cat([rgb_1, depth_1], dim=1))

        aca1_out = self.aca1(rgb_1, stage_1_cat, depth_1)
        aca2_out = self.aca2(rgb_2, aca1_out, depth_2)
        aca3_out = self.aca3(rgb_3, aca2_out, depth_3)
        aca4_out = self.aca4(rgb_4, aca3_out, depth_4)

        fdw4_out = self.fdw4(aca4_out, aca4_out)
        fdw3_out = self.fdw3(aca3_out, self.upsample2(self.fdw4_conv(fdw4_out)))
        fdw2_out = self.fdw2(aca2_out, self.upsample2(self.fdw3_conv(fdw3_out)))
        fdw1_out = self.fdw1(aca1_out, self.upsample2(self.fdw2_conv(fdw2_out)))


        out_1 = self.upsample4 (self.Head1(fdw1_out))
        out_2 = self.upsample8 (self.Head2(fdw2_out))
        out_3 = self.upsample16(self.Head3(fdw3_out))
        out_4 = self.upsample32(self.Head4(fdw4_out))

        return out_1, out_2, out_3, out_4, aca1_out, aca2_out, aca3_out, aca4_out,\
               fdw1_out, fdw2_out, fdw3_out, fdw4_out
        # return out_1, out_2, out_3, out_4


if __name__ == '__main__':
    input_rgb = torch.randn(2, 3, 256, 256)
    input_depth = torch.randn(2, 1, 256, 256)
    net = KD_model_3_teacher()

    # input_rgb = torch.randn(2, 512, 64, 64)
    # input_depth = torch.randn(2, 512, 64, 64)
    # net = FDW(512)
    # out = net(input_rgb, input_depth)
    # # print("out", out.shape)
    # print("out1", out[0].shape)
    # print("out2", out[1].shape)
    # print("out3", out[2].shape)
    # print("out4", out[3].shape)
    # print("out5", out[4].shape)
    # print("out6", out[5].shape)
    # print("out7", out[6].shape)
    # print("out8", out[7].shape)
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 1, 256, 256)
    model = KD_model_3_teacher()
    from FLOP import CalParams

    CalParams(model, a, b)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

# if __name__ == '__main__':
#     input_rgb = torch.randn(6, 128, 32, 32)
#     input_depth = torch.randn(6, 64, 64, 64)
#     input_depth_png = torch.randn(6, 128, 32, 32)
#     net = ACA_v3(64, 128, 32)
#
#     out = net(input_rgb, input_depth, input_depth_png)
#     print("out", out.shape)