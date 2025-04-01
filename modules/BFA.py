import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit_modified(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit_modified, self).__init__()
        self.groups = groups

        self.input_shape = 32  # change!!!!!it!!!!!!manually!!!!!!
        self.in_channels = in_channels

        self.locMap = nn.Parameter(torch.rand(self.input_shape, self.input_shape // 2 + 1))

        self.lambda_base = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.conv_layer_down55 = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                 out_channels=out_channels * 2,
                                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups,
                                                 bias=False, padding_mode='reflect')
        self.conv_layer_down55_shift = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                       out_channels=out_channels * 2,
                                                       kernel_size=3, stride=1, padding=2, dilation=2,
                                                       groups=self.groups, bias=False, padding_mode='reflect')

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.img_freq = None
        self.distill = None

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        locMap = self.locMap.expand_as(ffted[:, :1, :, :])  # B 1 H' W'
        ffted_copy = ffted.clone()

        cat_img_mask_freq = torch.cat((ffted[:, :self.in_channels, :, :],
                                       ffted[:, self.in_channels:, :, :],
                                       locMap), dim=1)

        ffted = self.conv_layer_down55(cat_img_mask_freq)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        ffted = self.relu(ffted)

        locMap_shift = torch.fft.fftshift(locMap, dim=-2)  ## ONLY IF NOT SHIFT BACK

        # REPEAT CONV
        cat_img_mask_freq1 = torch.cat((ffted[:, :self.in_channels, :, :],
                                        ffted[:, self.in_channels:, :, :],
                                        locMap_shift), dim=1)

        ffted = self.conv_layer_down55_shift(cat_img_mask_freq1)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        lambda_base = torch.sigmoid(self.lambda_base)

        ffted = ffted_copy * lambda_base + ffted * (1 - lambda_base)

        # irfft
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        epsilon = 0.5
        output = output - torch.mean(output) + torch.mean(x)
        output = torch.clip(output, float(x.min() - epsilon), float(x.max() + epsilon))

        self.distill = output  # for self perc
        return output



def build_act_layer(act_type):
    #Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * 4)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


