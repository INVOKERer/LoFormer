# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from basicsr.models.archs.attn_util import *
from basicsr.models.archs.norm_util import *
def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    idx = list(range(sampling_locations.shape[-2]))
    C = (kernel_w * kernel_h - 1)//2
    idx = [i for i in idx if i != C and (i-C) % (C*2+1) != 0]
    sampling_locations = sampling_locations[:,:,:,idx, :]
    return sampling_locations
def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid

def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale, remove_center):
    # for debug and test only,
    # need to use cuda version instead

    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_w != kernel_h):
        raise ValueError('remove_center is only compatible with square odd kernel size.')

    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    # print(input.shape)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*(kernel_h*kernel_w-remove_center)).to(input.device)
    # print(ref.shape)
    sampling_locations = (ref[:, :H_out, :W_out, :, :] + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    # print(sampling_locations.shape, ref.shape, H_out, W_out, kernel_w, kernel_h, input.shape)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    # print(sampling_grids.shape, N_, H_out*W_out, group, P_)
    # print(sampling_grids)
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    # print(mask.shape, (N_, H_out, W_out, group*P_))
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group*group_channels, H_out*W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()

def misc_core_pytorch(
        input, offset, mask, kernel, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale, remove_center):
    # for debug and test only,
    # need to use cuda version instead

    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_w != kernel_h):
        raise ValueError('remove_center is only compatible with square odd kernel size.')

    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w]) # , 'replicate'
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    # print(input.shape)
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*(kernel_h*kernel_w-remove_center)).to(input.device)
    # print(ref.shape, H_out, W_out)
    sampling_locations = (ref[:, :H_out, :W_out, :, :] + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    # print(offset.shape, sampling_locations.shape, ref.shape, H_out, W_out, kernel_w, kernel_h, input.shape)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    # print(sampling_grids.shape, N_, H_out*W_out, group, P_)
    # print(sampling_grids)
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)
    # print(sampling_input_.shape)
    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    # print(mask.shape, (N_, H_out, W_out, group*P_))
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    kernel = kernel.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    output = (sampling_input_ * mask * kernel).sum(-1).view(N_, group*group_channels, H_out*W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1) # .contiguous()

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, out_channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        # print(offset.shape)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x.permute(0, 3, 1, 2)
class MiSCFilter_mxt3(nn.Module):
    def __init__(
            self,
            channels=3,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        # self.dw_conv = nn.Sequential(
        #     nn.Conv2d(
        #         channels,
        #         channels,
        #         kernel_size=dw_kernel_size,
        #         stride=1,
        #         padding=(dw_kernel_size - 1) // 2,
        #         groups=channels),
        #     build_norm_layer(
        #         channels,
        #         norm_layer,
        #         'channels_first',
        #         'channels_last'),
        #     build_act_layer(act_layer))
        # self.offset = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center) * 2)
        # self.mask = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center))
        # self.input_proj = nn.Linear(channels, channels)
        # self.output_proj = nn.Linear(channels, out_channels)
        # self._reset_parameters()
        #
        # if center_feature_scale:
        #     self.center_feature_scale_proj_weight = nn.Parameter(
        #         torch.zeros((group, channels), dtype=torch.float))
        #     self.center_feature_scale_proj_bias = nn.Parameter(
        #         torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
        #     self.center_feature_scale_module = CenterFeatureScaleModule()



    def forward(self, img1, offset, weight):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        input = img1.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        mask = weight.permute(0, 2, 3, 1)
        offset = offset.permute(0, 2, 3, 1)
        # kernel = torch.einsum('bczij,bzdij->bcdij', kernel_h.unsqueeze(2), kernel_v.unsqueeze(1))
        # kernel = kernel.view(N, -1, H, W).permute(0, 2, 3, 1)
        # print(mask.shape, offset.shape)
        x = dcnv3_core_pytorch(
            input, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)


        return x.permute(0, 3, 1, 2)
class MiSCFilter_mxt(nn.Module):
    def __init__(
            self,
            channels=3,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        # self.dw_conv = nn.Sequential(
        #     nn.Conv2d(
        #         channels,
        #         channels,
        #         kernel_size=dw_kernel_size,
        #         stride=1,
        #         padding=(dw_kernel_size - 1) // 2,
        #         groups=channels),
        #     build_norm_layer(
        #         channels,
        #         norm_layer,
        #         'channels_first',
        #         'channels_last'),
        #     build_act_layer(act_layer))
        # self.offset = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center) * 2)
        # self.mask = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center))
        # self.input_proj = nn.Linear(channels, channels)
        # self.output_proj = nn.Linear(channels, out_channels)
        # self._reset_parameters()
        #
        # if center_feature_scale:
        #     self.center_feature_scale_proj_weight = nn.Parameter(
        #         torch.zeros((group, channels), dtype=torch.float))
        #     self.center_feature_scale_proj_bias = nn.Parameter(
        #         torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
        #     self.center_feature_scale_module = CenterFeatureScaleModule()



    def forward(self, img1, kernel_v, kernel_h, offset, weight):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        input = img1.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        mask = weight.permute(0, 2, 3, 1)
        offset = offset.permute(0, 2, 3, 1)
        kernel = torch.einsum('bczij,bzdij->bcdij', kernel_h.unsqueeze(2), kernel_v.unsqueeze(1))
        kernel = kernel.view(N, -1, H, W).permute(0, 2, 3, 1)
        x = misc_core_pytorch(
            input, offset, mask, kernel,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)


        return x.permute(0, 3, 1, 2)
class MiSCFilter_mxt2(nn.Module):
    def __init__(
            self,
            channels=3,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        # self.dw_conv = nn.Sequential(
        #     nn.Conv2d(
        #         channels,
        #         channels,
        #         kernel_size=dw_kernel_size,
        #         stride=1,
        #         padding=(dw_kernel_size - 1) // 2,
        #         groups=channels),
        #     build_norm_layer(
        #         channels,
        #         norm_layer,
        #         'channels_first',
        #         'channels_last'),
        #     build_act_layer(act_layer))
        # self.offset = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center) * 2)
        # self.mask = nn.Linear(
        #     channels,
        #     group * (kernel_size * kernel_size - remove_center))
        # self.input_proj = nn.Linear(channels, channels)
        # self.output_proj = nn.Linear(channels, out_channels)
        # self._reset_parameters()
        #
        # if center_feature_scale:
        #     self.center_feature_scale_proj_weight = nn.Parameter(
        #         torch.zeros((group, channels), dtype=torch.float))
        #     self.center_feature_scale_proj_bias = nn.Parameter(
        #         torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
        #     self.center_feature_scale_module = CenterFeatureScaleModule()



    def forward(self, img1, kernel, offset, weight):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = img1.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        mask = weight.permute(0, 2, 3, 1)
        offset = offset.permute(0, 2, 3, 1)
        # kernel = torch.einsum('bczij,bzdij->bcdij', kernel_h.unsqueeze(2), kernel_v.unsqueeze(1))
        kernel = kernel.permute(0, 2, 3, 1)
        x = misc_core_pytorch(
            input, offset, mask, kernel,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)


        return x.permute(0, 3, 1, 2)
class D2Conv_pytorch(nn.Module):
    def __init__(
            self,
            channels=3,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)


    def forward(self, input, mask, offset):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = input # self.input_proj(input)

        mask = mask.reshape(N, 1, 1, self.group, -1)
        mask = repeat(mask, 'b h w g n -> b (h h2) (w w2) g n', h2=H, w2=W)
        # mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        # print(mask.shape, x.shape, offset.shape)
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        return x.permute(0, 3, 1, 2)
class DCNv3_ker_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=7,
            motion_blur_kernel_size=19,
            dw_kernel_size=3,
            stride=1,
            pad=3,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.deform_kersize = (kernel_size * kernel_size - remove_center)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Conv2d(channels, group, 1)
        # for p in self.offset.parameters(): p.requires_grad = False
        h, w = motion_blur_kernel_size, motion_blur_kernel_size
        x_coords = torch.linspace(-w // 2 + 1, w // 2, steps=w)  # .view(1, w).expand(h, w)
        y_coords = torch.linspace(-h // 2 + 1, h // 2, steps=h)  # .view(h, 1).expand(h, w)
        # print(x_coords, y_coords)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        # self.offseth = nn.Parameter(grid_x, requires_grad=False)
        # self.offsetw = nn.Parameter(grid_y, requires_grad=False)
        self.offsetw = nn.Parameter(grid_x, requires_grad=False)
        self.offseth = nn.Parameter(grid_y, requires_grad=False)
        # self.kernel_deblur = kernel_deblurV2(in_ch=group, out_ch=out_channels)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        # self.input_proj2 = nn.Conv2d(channels, group, 1)
        self.output_proj = nn.Linear(channels, out_channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 1./self.channels)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        # xavier_uniform_(self.input_proj2.weight.data)
        # constant_(self.input_proj2.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, kernel):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        # x2 = self.input_proj2(input)
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)

        kernel = self.offset(kernel)
        # x2 = self.kernel_deblur(x2, kernel)
        # print(kernel.shape)
        b, c, kh, kw = kernel.shape
        ker = kernel.view(b, c, -1)
        _, coord = torch.topk(ker, k=self.deform_kersize, dim=-1)
        offseth = repeat(self.offseth.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
        offsetw = repeat(self.offsetw.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
        offseth = offseth.contiguous().view(b, c, -1)
        offsetw = offsetw.contiguous().view(b, c, -1)
        # print(ker.shape, offseth.shape)
        offseth = torch.gather(offseth, dim=-1, index=coord)  # offseth[coord]
        offsetw = torch.gather(offsetw, dim=-1, index=coord)  # offsetw[coord]
        offset = torch.cat([offseth.unsqueeze(-1), offsetw.unsqueeze(-1)], dim=-1)
        offset = offset.view(b, 1, 1, -1)
        # print(offset.shape)
        # offset = self.offset(offset)
        offset = repeat(offset, 'b h w c -> b (rh h) (rw w) c', rh=H, rw=W)
        # print(offset.shape)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        # print(mask.shape)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        # print(x.shape, mask.shape)
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)
        x = x.permute(0, 3, 1, 2) # + x2
        return x, kernel
class DCNv3_kerxxx_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=7,
            dw_kernel_size=3,
            stride=1,
            pad=3,
            dilation=1,
            group=64,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.deform_kersize = (kernel_size * kernel_size - remove_center)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Conv2d(channels, group , True)

        h, w = kernel_size, kernel_size
        x_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=w)  # .view(1, w).expand(h, w)
        y_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=h)  # .view(h, 1).expand(h, w)
        # print(x_coords, y_coords)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        self.offseth = nn.Parameter(grid_x, requires_grad=False)
        self.offsetw = nn.Parameter(grid_y, requires_grad=False)



        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, out_channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, kernel):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)

        kernel = self.offset(kernel)
        # print(kernel.shape)
        b, c, kh, kw = kernel.shape
        ker = kernel.view(b, c, -1)
        _, coord = torch.topk(ker, k=self.deform_kersize, dim=-1)
        offseth = repeat(self.offseth.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
        offsetw = repeat(self.offsetw.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
        offseth = offseth.contiguous().view(b, c, -1)
        offsetw = offsetw.contiguous().view(b, c, -1)
        offseth = torch.gather(offseth, dim=-1, index=coord)  # offseth[coord]
        offsetw = torch.gather(offsetw, dim=-1, index=coord)  # offsetw[coord]
        offset = torch.cat([offseth.unsqueeze(-1), offsetw.unsqueeze(-1)], dim=-1)
        offset = offset.view(b, 1, 1, -1)
        # print(offset.shape)
        # offset = self.offset(offset)
        offset = repeat(offset, 'b h w c -> b (rh h) (rw w) c', rh=H, rw=W)
        # print(offset.shape)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        # print(mask.shape)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)
        # print(x.shape, mask.shape)
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x.permute(0, 3, 1, 2)
class DCNv3_pytorch_calayer(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.norm_input = LayerNorm2d(channels)
        self.dw_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((dw_kernel_size, dw_kernel_size)),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=0,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, out_channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, C, H, W)
        :return output                     (N, C, H, W)
        """
        input = self.norm_input(input)
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        Ho, Wo = x1.shape[1:-1]
        # print(x1.shape)
        mask = self.mask(x1).reshape(N, Ho, Wo, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, Ho, Wo, -1)
        # print(offset)
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x.permute(0, 3, 1, 2)
class DCNv3_pytorch_aaai(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
            out_channels=None
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        if out_channels is None:
            out_channels = channels
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.dw_conv_phase = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, out_channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)

        x1_phase = torch.fft.rfft2(x1)
        x1_phase = torch.angle(x1_phase)
        x1_phase = torch.fft.irfft2(torch.exp(1j*x1_phase))
        x1_offset = self.dw_conv_phase(x1_phase)
        offset = self.offset(x1_offset)

        x1_mask = self.dw_conv(x1)
        mask = self.mask(x1_mask).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x.permute(0, 3, 1, 2)
class DCNv3_pytorch_dual(nn.Module):
    def __init__(
            self,
            channels=64,
            channels_ref=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(nn.Conv2d(
                channels_ref,
                channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.ref_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, ref):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.permute(0, 2, 3, 1)
        N, H, W, _ = input.shape
        # print(input.shape)
        x = self.input_proj(input)
        x_proj = x

        # x1 = input.permute(0, 3, 1, 2)
        ref = self.dw_conv(ref)

        ref = ref.permute(0, 2, 3, 1)
        offset = self.offset(self.ref_proj(ref))
        mask = self.mask(ref).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                ref, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x.permute(0, 3, 1, 2)
class Offset_Gen(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        n = window_size ** 2
        self.n = n * num_heads
        self.out_dim = out_dim
        self.window_size = window_size
        self.qk_norm = False
        self.qk = nn.Conv2d(dim, n * num_heads + out_dim, kernel_size=1, bias=True)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1, num_heads, 1, 1) / math.sqrt(dim))

        # self.project_out = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)

    def get_attn(self, qk):
        H, W = qk.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qk = check_image_size(qk, self.window_size)

        q, k = torch.split(qk, [self.n, self.out_dim], dim=1)
        # print(q.shape, k.shape)
        q = rearrange(q, 'b (head c) (h h1) (w w1) -> b h1 w1 head c (h w)', head=self.num_heads, h=self.window_size, w=self.window_size) # c = n
        k = rearrange(k, 'b (head c) (h h1) (w w1) -> b h1 w1 head c (h w)', head=self.num_heads, h=self.window_size, w=self.window_size)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

        out = (q @ k.transpose(-2, -1)) * self.temperature
        # print(out.shape)
        out = rearrange(out, 'b h1 w1 head (h w) c -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h=self.window_size, w=self.window_size)

        return out[:, :, :H, :W]

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        qk = self.qk(x)
        out = self.get_attn(qk)
        # out = self.project_out(out)
        return out.permute(0, 2, 3, 1)


if __name__ == '__main__':
    # kernel_size = 19
    # h, w = kernel_size, kernel_size
    # x_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=w) # .view(1, w).expand(h, w)
    # y_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=h) # .view(h, 1).expand(h, w)
    # print(x_coords, y_coords)
    # grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
    # print(grid_x, grid_y)
    # kernel = torch.randn([2, 3, kernel_size, kernel_size])
    # b, c, kh, kw = kernel.shape
    # ker = kernel.view(b, c, -1)
    # _, coord = torch.topk(ker, k=49, dim=-1)
    # offseth = repeat(grid_x.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
    # offsetw = repeat(grid_y.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=c)
    # print(offseth.shape)
    # offseth = offseth.contiguous().view(b, c, -1)
    # offsetw = offsetw.contiguous().view(b, c, -1)
    # print(coord, coord.shape)
    # offseth = torch.gather(offseth, dim=-1, index=coord)  # offseth[coord]
    # offsetw = torch.gather(offsetw, dim=-1, index=coord)  # offsetw[coord]
    # print(offseth.shape, offsetw)

    model = DCNv3_ker_pytorch(32, group=32).cuda()
    kernel = torch.randn([1, 32, 19, 19]).cuda()
    x = torch.randn((1, 32, 128, 128)).cuda()
    y = model(x, kernel)
    print(y.shape)
