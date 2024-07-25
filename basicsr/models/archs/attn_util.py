# import kornia
import einops
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn as nn
import torch
import os
import cv2
import math
from torch.nn.init import xavier_uniform_, constant_
from basicsr.models.archs.dct_util import *
from basicsr.models.archs.norm_util import *
from basicsr.models.archs.win_util import *
from basicsr.models.archs.gcn_lib.torch_vertex import *
from torchvision.ops import DeformConv2d
from basicsr.models.archs.utils_deblur import *
# from basicsr.models.archs.baseblock import *
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
# import matplotlib.gridspec as gridspec
import scipy
# def check_image_size(x, padder_size, mode='constant'):
#     _, _, h, w = x.size()
#     mod_pad_h = (padder_size - h % padder_size) % padder_size
#     mod_pad_w = (padder_size - w % padder_size) % padder_size
#     x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
#     return x
def get_attn_channel(qkv, temperature, num_heads, normalize=True, norm_dim=-1, x_adjust=None): # temp_adj=None):
    _, _, H, W = qkv.shape
    q, k, v = qkv.chunk(3, dim=1)

    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=num_heads)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=num_heads)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=num_heads)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=norm_dim)
        k = torch.nn.functional.normalize(k, dim=norm_dim)

    if x_adjust is not None:
        # if temp_adj == 'mean':
        #     # v = qkv.chunk(3, dim=1)[-1]
        #     x_adjust = torch.mean(torch.abs(v), dim=[-2, -1], keepdim=True)
        # elif temp_adj == '00':
        #     x_adjust = torch.mean(torch.abs(v[:, :, :, 0]), dim=2, keepdim=True)
        #     x_adjust = x_adjust.unsqueeze(3)
        # else:
        #     v_abs = torch.abs(v)
        #     v_abs = rearrange(v_abs, 'b h c d -> b h (c d)')
        #     x_adjust, _ = torch.max(v_abs, dim=-1, keepdim=True)
        #     x_adjust = x_adjust.unsqueeze(3)

        attn = (q @ k.transpose(-2, -1)) * temperature / (x_adjust + 1e-6)
    else:
        attn = (q @ k.transpose(-2, -1))
        attn = attn * temperature
    attn = attn.softmax(dim=-1)

    out = (attn @ v)

    return rearrange(out, 'b head c (h w) -> b (head c) h w', head=num_heads, h=H,
                    w=W)
def get_attn_spatial(qkv, temperature, num_heads, normalize=True, norm_dim=-2):
    # if window_size_attn != window_size:
    _, _, H, W = qkv.shape
    # qkv, batch_list = window_partitionx(qkv, window_size)
    q, k, v = qkv.chunk(3, dim=1)

    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=num_heads)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=num_heads)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=num_heads)

    if normalize:
        q = torch.nn.functional.normalize(q, dim=norm_dim)
        k = torch.nn.functional.normalize(k, dim=norm_dim)

    attn = (q.transpose(-2, -1) @ k) * temperature

    attn = attn.softmax(dim=-1)

    out = (attn @ v.transpose(-2, -1)) # .contiguous())
    # print(attn.shape, out.shape)
    out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=num_heads,
                    h=H, w=W)
    # if window_size_attn != window_size:
    # out = window_reversex(out, window_size, window_size, window_size, batch_list)
    return out
def get_attn_spatial_grid(qkv, temperature, num_heads, num_k=8, normalize=True, norm_dim=-2, window_size=None):
    _, _, h, w = qkv.shape
    if window_size is not None and (h != window_size or w != window_size):
        qkv, batch_list = window_partitionx(qkv, window_size)
    else:
        qkv = check_image_size(qkv, num_k)
    # print(qkv.shape)
    q, k, v = qkv.chunk(3, dim=1)
    _, _, H, W = q.shape
    q = rearrange(q, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    k = rearrange(k, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    v = rearrange(v, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    # print(q.shape)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=norm_dim)
        k = torch.nn.functional.normalize(k, dim=norm_dim)

    attn = (q.transpose(-2, -1) @ k) * temperature

    attn = attn.softmax(dim=-1)
    # print(attn.shape)
    out = (attn @ v.transpose(-2, -1))
    out = rearrange(out.transpose(-2, -1), 'b head (h w) c (k1 k2) -> b (head c) (k1 h) (k2 w)', head=num_heads,
                    h=H//num_k, w=W//num_k, k1=num_k, k2=num_k)
    if window_size is not None and (h != window_size or w != window_size):
        out = window_reversex(out, window_size, h, w, batch_list)
    else:
        out = out[:, :, :h, :w].contiguous()
    return out
def get_attn_channel_grid(qkv, temperature, num_heads, num_k=8, normalize=True, norm_dim=-1, window_size=None):
    _, _, h, w = qkv.shape
    if window_size is not None and (h != window_size or w != window_size):
        qkv, batch_list = window_partitionx(qkv, window_size)
    else:
        qkv = check_image_size(qkv, num_k)
    # print(qkv.shape)
    q, k, v = qkv.chunk(3, dim=1)
    _, _, H, W = q.shape
    q = rearrange(q, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    k = rearrange(k, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    v = rearrange(v, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=num_heads, k1=num_k, k2=num_k)
    # print(q.shape)
    if normalize:
        q = torch.nn.functional.normalize(q, dim=norm_dim)
        k = torch.nn.functional.normalize(k, dim=norm_dim)

    attn = (q @ k.transpose(-2, -1)) * temperature

    attn = attn.softmax(dim=-1)
    # print(attn.shape)
    out = (attn @ v)
    out = rearrange(out, 'b head (h w) c (k1 k2) -> b (head c) (k1 h) (k2 w)', head=num_heads,
                    h=H//num_k, w=W//num_k, k1=num_k, k2=num_k)
    if window_size is not None and (h != window_size or w != window_size):
        out = window_reversex(out, window_size, h, w, batch_list)
    else:
        out = out[:, :, :h, :w].contiguous()
    return out


##########################################################################
## SE
class ShiftAttn(nn.Module):
    def __init__(self, dim, bias=True, window_size=8, sigmoid=True):
        super().__init__()

        self.window_size = window_size
        self.sigmoid = sigmoid
        # Simplified Channel Attention
        # print(dim)
        ratio = 0.25
        hid_dim = int(dim * ratio)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 5)),
            LayerNorm2d(dim),
            nn.Conv2d(in_channels=dim, out_channels=hid_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            nn.Conv2d(in_channels=hid_dim, out_channels=hid_dim, kernel_size=3, padding=(1, 0), stride=1,
                      groups=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels=hid_dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            # nn.BatchNorm2d(dim)
        )
        self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, H, W = x.shape
        # if self.window_size is not None and (H != self.window_size or W != self.window_size):
        #     x, batch_list = window_partitionx(x, self.window_size)
        x_fft = torch.fft.rfft(x, dim=-1)
        # angle = torch.angle(x_fft)
        x_phase = torch.sigmoid(self.sca(x)) * self.gama
        w_range = torch.arange(0, x_fft.shape[-1], device=x.device)
        w_range = w_range.view(1, 1, 1, -1)
        x_phase = torch.einsum('bchi,bciw->bchw', x_phase, w_range)
        x_fft = x_fft * torch.exp(1j*x_phase)
        # if self.window_size is not None and (H != self.window_size or W != self.window_size):
        #     x = window_reversex(x, self.window_size, H, W, batch_list)
        return torch.fft.irfft(x_fft, dim=-1)

class WSCA(nn.Module):
    def __init__(self, dim, bias=True, window_size=8, sigmoid=True):
        super().__init__()

        self.window_size = window_size
        self.sigmoid = sigmoid
        # Simplified Channel Attention
        # print(dim)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dim//4, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            # nn.BatchNorm2d(dim)
        )
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        if self.sigmoid:
            # print(x.shape, torch.sigmoid(self.sca(x)).shape)
            x = x * torch.sigmoid(self.sca(x))
        else:
            x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        return x
class SSA(nn.Module):
    def __init__(self, dim, bias=True, window_size=8):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, bias=bias, padding=1)

    def forward(self, x):
        _, _, H, W = x.shape
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = self.conv(torch.cat([x_max, x_mean], dim=1))
        
        x = x * torch.sigmoid(x_attn)

        return x


class WSSCA(nn.Module):
    def __init__(self, bias=True, window_size=8, attn='channel'):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, bias=bias, padding=1)
        if attn == 'channel':
            self.dct_bench = ChannelAttention(window_size ** 2, 1, bias=bias, window_size=window_size)
        elif attn == 'spatial':
            self.dct_bench = SpatialAttention(window_size ** 2, 1, bias=bias, window_size=window_size)
        else:
            self.dct_bench = WSCA(window_size**2, bias, window_size, sigmoid=True)
        self.dct_ = DCT2d_fold_branch(window_size=8, pad_size=7, stride=1, pad_mode='reflect')
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        _, _, H, W = x.shape
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = self.conv(torch.cat([x_max, x_mean], dim=1))
        x_attn = torch.sigmoid(x_attn)
        x_attn_dct = self.dct_(x_attn, True)
        x_attn = self.dct_bench(x_attn_dct) + x_attn_dct
        x_attn = self.dct_(x_attn, False)
        x = x * x_attn  # torch.sigmoid(x_attn)
        return x
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.frelu = fft_bench_complex_mlp_onlyrelu()
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class AAAIAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.norm = LayerNorm(dim)
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.project_out.weight.data, 0.)
        constant_(self.project_out.bias.data, 0.)
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(self.norm(x)))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out) + x
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class ContentAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, window_size_dct, num_heads, dct=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kmeans=True, cs='spatial'):

        super().__init__()
        self.dim = dim
        self.cs = cs
        self.percent = 0.33
        self.dct = dct
        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()
        self.window_size_dct = window_size_dct
        if self.window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        self.window_size = window_size  # Wh, Ww
        self.ws = window_size ** 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # H, W = x.shape[-2:]
        _, _, Hx, Wx = x.shape
        if self.dct:
            x = self.dct2d(x)
        if self.window_size_dct is not None and (Hx > self.window_size_dct or Wx > self.window_size_dct):
            x, batch_list = self.winp(x)
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)  # 3, B_, self.num_heads,N,D

        if True:
            q_pre = qkv[0].reshape(B_ * self.num_heads, N, C // self.num_heads).permute(0, 2, 1)
            # qkv_pre[:,0].reshape(b*self.num_heads,qkvhd//3//self.num_heads,hh*ww)
            ntimes = int(math.log(N // self.ws, 2))
            q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_ * self.num_heads, N)
            for i in range(ntimes):
                bh, d, n = q_pre.shape
                q_pre_new = q_pre.reshape(bh, d, 2, n // 2)
                q_avg = q_pre_new.mean(dim=-1)  # .reshape(b*self.num_heads,qkvhd//3//self.num_heads,)
                q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                iters = 2
                for i in range(iters):
                    q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg)
                    soft_assign = torch.nn.functional.softmax(q_scores * 100, dim=-1).detach()
                    q_avg = q_pre.bmm(soft_assign)
                    q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg).reshape(bh, n,
                                                                                                            2)  # .unsqueeze(2)
                q_idx = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)
                _, q_idx = torch.sort(q_idx, dim=-1)
                q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)
                q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
                q_pre = q_pre.gather(dim=-1, index=q_idx).reshape(bh, d, 2, n // 2).permute(0, 2, 1, 3).reshape(bh * 2,
                                                                                                                d,
                                                                                                                n // 2)

            q_idx = q_idx_last.view(B_, self.num_heads, N)
            _, q_idx_rev = torch.sort(q_idx, dim=-1)
            q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
            qkv_pre = qkv.gather(dim=-2, index=q_idx)

            q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=self.ws)

            k = k.view(B_ * ((N // self.ws)) // 2, 2, self.num_heads, self.ws, -1)
            k_over1 = k[:, 1, :, :int(self.ws * self.percent)].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over2 = k[:, 0, :, -int(self.ws * self.percent):].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over = torch.cat([k_over1, k_over2], 1)
            k = torch.cat([k, k_over], 3).contiguous().view(B_ * ((N // self.ws)), self.num_heads,
                                                            self.ws + int(self.ws * self.percent), -1)

            v = v.view(B_ * ((N // self.ws)) // 2, 2, self.num_heads, self.ws, -1)
            v_over1 = v[:, 1, :, :int(self.ws * self.percent)].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over2 = v[:, 0, :, -int(self.ws * self.percent):].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over = torch.cat([v_over1, v_over2], 1)
            v = torch.cat([v, v_over], 3).contiguous().view(B_ * ((N // self.ws)), self.num_heads,
                                                            self.ws + int(self.ws * self.percent), -1)
        if 'spatial' in self.cs:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        else:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        if 'spatial' in self.cs:
            out = attn @ v
        else:
            out = attn @ v.transpose(-2, -1)
            out = out.transpose(-2, -1)

        if True:
            out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h=self.num_heads, b=B_)
            v = rearrange(v[:, :, :self.ws, :], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
            W = int(math.sqrt(N))

            out = out.reshape(B_, self.num_heads, C // self.num_heads, -1)
            q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
            x = out.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N) # .permute(0, 2, 1)
            v = v.gather(dim=-1, index=q_idx_rev).reshape(B_, C, W, W)
            v = self.get_v(v)
            v = v.reshape(B_, C, N) # .permute(0, 2, 1)
            x = x + v

        x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        if self.dct:
            x = self.idct2d(x)
        if self.window_size_dct is not None and (Hx > self.window_size_dct or Wx > self.window_size_dct):
            x = self.winr(x, Hx, Wx, batch_list)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ContentChannelAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, window_size_dct, num_heads, dct=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 kmeans=True, cs='channel', norm_dim=-1, qk_norm=True, temp_div=False, temp_adj=None, x_v='add'):

        super().__init__()
        self.dim = dim
        self.cs = cs
        self.percent = 0.33
        self.dct = dct
        self.x_v = x_v
        self.norm_dim = norm_dim
        self.qk_norm = qk_norm
        self.temp_adj = temp_adj
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.window_size_dct = window_size_dct
        if self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        if self.cs == 'channel':
            self.ws = dim // 4  # window_size # ** 2
            self.get_v = nn.Linear(dim, dim)
        else:
            self.ws = window_size # ** 2
            self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans
        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim // num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_adjust(self, v):
        x_adjust = None
        # b = v.shape[0]
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        return x_adjust
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # H, W = x.shape[-2:]
        _, _, Hx, Wx = x.shape
        if self.dct:
            x = self.dct2d(x)
        # if self.cs == 'channel' and self.window_size is not None and (Hx > self.window_size or Wx > self.window_size):
        #     x, batch_list = self.winp(x)
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')

        qkv = self.qkv(x)
        if self.cs == 'channel':
            qkv = qkv.permute(0, 2, 1)
            B_, N_, C = qkv.shape

            N = N_ // 3
            qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B_, self.num_heads,N,D
        else:
            B_, N, C = x.shape
            self.ws = (h // self.window_size) ** 2
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        if True:
            q_pre = qkv[0].reshape(B_ * self.num_heads, N, C // self.num_heads).permute(0, 2, 1)
            # qkv_pre[:,0].reshape(b*self.num_heads,qkvhd//3//self.num_heads,hh*ww)
            ntimes = 1 # int(math.log(self.ws, 2))
            q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_ * self.num_heads, N)
            for i in range(ntimes):
                bh, d, n = q_pre.shape
                q_pre_new = q_pre.reshape(bh, d, 2, n // 2)
                q_avg = q_pre_new.mean(dim=-1)  # .reshape(b*self.num_heads,qkvhd//3//self.num_heads,)
                q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                iters = 2
                for i in range(iters):
                    q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg)
                    soft_assign = torch.nn.functional.softmax(q_scores * 100, dim=-1).detach()
                    q_avg = q_pre.bmm(soft_assign)
                    q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                q_scores = torch.nn.functional.normalize(q_pre.permute(0, 2, 1), dim=-1).bmm(q_avg).reshape(bh, n, 2)  # .unsqueeze(2)
                q_idx = (q_scores[:, :, 0] + 1) / (q_scores[:, :, 1] + 1)
                _, q_idx = torch.sort(q_idx, dim=-1)
                q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh * 2, n // 2)
                q_idx = q_idx.unsqueeze(1).expand(q_pre.size())
                q_pre = q_pre.gather(dim=-1, index=q_idx).reshape(bh, d, 2, n // 2).permute(0, 2, 1, 3).reshape(bh * 2, d, n // 2)

            q_idx = q_idx_last.view(B_, self.num_heads, N)
            _, q_idx_rev = torch.sort(q_idx, dim=-1)
            q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size())
            qkv_pre = qkv.gather(dim=-2, index=q_idx)

            q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=self.ws)

            k = k.view(B_ * ((N // self.ws)) // 2, 2, self.num_heads, self.ws, -1)
            k_over1 = k[:, 1, :, :int(self.ws * self.percent)].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over2 = k[:, 0, :, -int(self.ws * self.percent):].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            k_over = torch.cat([k_over1, k_over2], 1)
            k = torch.cat([k, k_over], 3).contiguous().view(B_ * ((N // self.ws)), self.num_heads,
                                                            self.ws + int(self.ws * self.percent), -1)

            v = v.view(B_ * ((N // self.ws)) // 2, 2, self.num_heads, self.ws, -1)
            v_over1 = v[:, 1, :, :int(self.ws * self.percent)].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over2 = v[:, 0, :, -int(self.ws * self.percent):].unsqueeze(1)  # .expand(-1,2,-1,-1,-1)
            v_over = torch.cat([v_over1, v_over2], 1)
            v = torch.cat([v, v_over], 3).contiguous().view(B_ * ((N // self.ws)), self.num_heads,
                                                            self.ws + int(self.ws * self.percent), -1)
        # if 'spatial' in self.cs:
        #     attn = (q @ k.transpose(-2, -1)) * self.scale
        # else:
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        x_adjust = self.get_adjust(v)
        if x_adjust is not None:
            if 'sub' in self.temp_adj:
                attn = attn - x_adjust
            else:
                attn = attn / (x_adjust + 1e-6)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # if 'spatial' in self.cs:
        # print(attn.shape, v.shape)

        out = attn @ v

        # else:
        #     out = attn @ v.transpose(-2, -1)
        #     out = out.transpose(-2, -1)

        if True:
            out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h=self.num_heads, b=B_)
            # print(out.shape)
            v = rearrange(v[:, :, :self.ws, :], '(b nw) h ws d -> b h d (nw ws)', h=self.num_heads, b=B_)
            # W = int(math.sqrt(N))

            out = out.reshape(B_, self.num_heads, C // self.num_heads, -1)
            q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size())
            # print(out.shape, v.shape, W)
            x = out.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N) # .permute(0, 2, 1)
            v = v.gather(dim=-1, index=q_idx_rev).reshape(B_, C, N)
            if self.cs == 'channel':
                v = self.get_v(v)
            else:
                v = self.get_v(v.reshape(B_, C, h, w))
                v = v.reshape(B_, C, N)
            # v = v.reshape(B_, C, N) # .permute(0, 2, 1)
            if self.x_v == 'mul':
                x = x * v
            else:
                x = x + v
        if self.cs == 'channel':
            x = x.permute(0, 2, 1)
        x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        # if self.cs == 'channel' and self.window_size is not None and (Hx > self.window_size or Wx > self.window_size):
        #     x = self.winr(x, Hx, Wx, batch_list)
        if self.dct:
            x = self.idct2d(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=False,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim
        # print(self.temp_adj)
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj == 'max':
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        if qk_LN:
            self.norm = LayerNorm(dim, True)
            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
            if v_proj:
                self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            else:
                self.v = nn.Identity()
                self.v_dwconv = nn.Identity()
        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 3, bias=bias)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x(64, 64)
            self.idct2d = IDCT2x(64, 64)
        self.num_heads = num_heads
        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            # print(num_heads)
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)

        if self.qk_LN:
            qk = self.qk_dwconv(self.qk(self.norm(x)))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        # if self.qk_LN:
        #     q, k, v = torch.chunk(qkv, 3, 1)
        #     qk = self.norm(torch.cat([q, k], dim=1))
        #     qkv = torch.cat([qk, v], dim=1)
        #     normalize = False
        # else:
        normalize = self.qk_norm
        _, _, H, W = qkv.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        if self.temp_adj and len(self.temp_adj) > 0:
            x_adjust = rearrange(torch.abs(qkv), 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            x_adjust = self.pool(x_adjust)
        else:
            x_adjust = None
        # print(x_adjust.shape)
        out = get_attn_channel(qkv, self.temperature, num_heads=self.num_heads,
                               normalize=normalize, norm_dim=self.norm_dim, x_adjust=x_adjust)# temp_adj=self.temp_adj)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out

class MDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 kernel_size_offset=3, offset_groups=1, kernel_size_mask=3):
        super().__init__()
        self.deformconv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        out_channels_offset = 2 * offset_groups * kernel_size_offset * kernel_size_offset
        self.conv_offset = nn.Conv2d(in_channels, out_channels_offset,
                                     kernel_size=kernel_size_offset, stride=stride, padding=padding, groups=offset_groups)
        init_offset = torch.Tensor(np.zeros([out_channels_offset, in_channels, kernel_size_offset, kernel_size_offset]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0

        out_channels_mask = offset_groups * kernel_size_offset * kernel_size_offset
        self.conv_mask = nn.Conv2d(in_channels, out_channels_mask,
                                   kernel_size=kernel_size_mask, groups=offset_groups, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([out_channels_mask, in_channels, kernel_size_mask, kernel_size_mask]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))  # 保证在0到1之间
        out = self.deformconv(x, offset, mask)
        return out
class WDCT_SE(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', ratio=4):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        n = window_size**2
        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.mlp1 = nn.Linear(dim, dim, bias=bias)
        # self.mlp2 = nn.Linear(n, n, bias=bias)
        hid_dim = dim // ratio
        self.norm1 = nn.LayerNorm(dim)
        self.weight1 = nn.Parameter(torch.randn(dim, dim, n), requires_grad=True)

        # self.weight2 = nn.Parameter(torch.randn(hid_dim, dim, n), requires_grad=True)
        init.kaiming_uniform_(self.weight1, a=math.sqrt(16))
        # init.kaiming_uniform_(self.weight2, a=math.sqrt(16))
        self.act = nn.GELU()
        # self.mlp3 = nn.Linear(dim, dim, bias=bias)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        x = check_image_size(x, self.window_size)
        Hx, Wx = x.shape[-2:]
        x = rearrange(x, 'b c (h1 h) (w1 w) -> b h1 w1 c h w',
                      h=self.window_size, w=self.window_size)
        x_freq_ = self.dct(x)
        x_freq = rearrange(x_freq_, 'b h1 w1 c h w-> b (c h w) h1 w1')
        x_freq = self.avg(x_freq)
        x_freq = rearrange(x_freq, 'b (c h w) h1 w1-> b (c h1 w1) (h w)',
                           h=self.window_size, w=self.window_size)
        # x_freq = self.mlp1(x_freq.transpose(-1,-2)).transpose(-1,-2)
        # x_freq = self.mlp2(x_freq)
        x_freq = self.norm1(x_freq.transpose(-1, -2)).transpose(-1, -2)
        x_freq = torch.einsum('bcn,chn->bhn', x_freq, self.weight1)
        # x_freq = self.act(x_freq)
        # x_freq = torch.einsum('bcn,chn->bhn', x_freq, self.weight2)
        # x_freq = self.mlp3(x_freq.transpose(-1, -2)).transpose(-1, -2)
        x_freq = rearrange(x_freq, 'b (c h1 w1) (h w)-> b h1 w1 c h w', h1=1, w1=1,
                           h=self.window_size, w=self.window_size)
        out = x_freq_ * torch.sigmoid(x_freq)
        out = self.idct(out)
        out = rearrange(out, 'b h1 w1 c h w -> b c (h1 h) (w1 w)')
        return out[:, :, :H, :W]

    def forward(self, x):
        out = self.get_attn(x)
        return out

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_std, x_mean = torch.std_mean(x, dim=1, keepdim=True)
        uncertainty = x_std / (torch.abs(x_mean)+1e-6)
        # print(x.shape, uncertainty.shape)
        uncertainty = self.conv(uncertainty * x *self.beta)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_Freq(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_std, x_mean = torch.std_mean(torch.abs(x_fft), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty*self.beta)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_Phase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, dim // 2, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim // 2, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_Phase = torch.fft.irfft2(x_fft / (torch.abs(x_fft)+1e-6))
        x_std, x_mean = torch.std_mean(x_Phase, dim=1, keepdim=True)
        uncertainty = x_std / (torch.abs(x_mean)+1e-6)
        # uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        # uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty)*self.beta
        # uncertainty = torch.sigmoid(uncertainty)
        return uncertainty * x

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_DCTFreq(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = self.dct(x)
        x_std, x_mean = torch.std_mean(torch.abs(x_fft), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = self.idct(uncertainty)
        uncertainty = self.conv(uncertainty*self.beta)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_FreqV4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(self.norm(x))
        x_std, x_mean = torch.std_mean(torch.abs(x_fft), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty*self.beta)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_FreqV5(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_std, x_mean = torch.std_mean(torch.log(torch.abs(x_fft)+1), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty*self.beta)
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_FreqV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_std, x_mean = torch.std_mean(torch.abs(x_fft), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = uncertainty*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty)*self.beta
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class Uncertainty_Map_FreqV3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim//4, dim, 3, padding=1))
        self.convx = nn.Sequential(nn.Conv2d(1, dim // 2, 3, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(dim // 2, dim, 3, padding=1))
        # self.act = nn.GELU()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_std, x_mean = torch.std_mean(torch.abs(x_fft), dim=1, keepdim=True)
        uncertainty = x_std / (x_mean+1e-6)
        uncertainty = self.convx(uncertainty)*x_fft
        # print(x.shape, uncertainty.shape)
        uncertainty = torch.fft.irfft2(uncertainty)
        uncertainty = self.conv(uncertainty)*self.beta
        uncertainty = torch.sigmoid(uncertainty)
        return uncertainty

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        return flops
class ICCVAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops


class RecoordAttention(nn.Module):
    def __init__(
            self,
            channels=64,
            num_heads=1,
            kernel_size=19,
            pad=0,
            div_ker=False,
            deconv=False,
            reverse=True,
            bias=True, cs='global_channel'
    ):

        super().__init__()
        self.div_ker = div_ker
        self.pad = pad
        self.num_heads = num_heads
        self.deconv = deconv
        self.reverse = reverse
        if self.pad:
            self.pad_ = nn.ReflectionPad2d(pad)
        self.deform_kersize = channels // num_heads
        h, w = kernel_size, kernel_size
        x_coords = torch.linspace(-w // 2 + 1, w // 2, steps=w)  # .view(1, w).expand(h, w)
        y_coords = torch.linspace(-h // 2 + 1, h // 2, steps=h)  # .view(h, 1).expand(h, w)
        # print(x_coords, y_coords)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        self.offseth = nn.Parameter(grid_y, requires_grad=False)
        self.offsetw = nn.Parameter(grid_x, requires_grad=False)
        if div_ker:
            self.proj1 = nn.Sequential(nn.Linear(self.deform_kersize, self.deform_kersize),
                                      nn.GELU())
            self.proj2 = nn.Sequential(nn.Linear(self.deform_kersize, self.deform_kersize),
                                       nn.GELU())
            # self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1)*math.sqrt(self.deform_kersize), requires_grad=True)
        if self.deconv:
            self.xconv = nn.Conv2d(channels, num_heads, kernel_size=3, padding=1)
            self.yconv = nn.Conv2d(channels + num_heads, channels, kernel_size=3, padding=1)
        self.attn = ICCVAttention(channels, num_heads, bias=bias, cs=cs)
    def inv(self, x, offseth, offsetw):
        Hx, Wx = x.shape[-2:]
        x = rearrange(x, 'b (head c) h w -> b head c h w', c=self.deform_kersize)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        shift_h = repeat(offseth, 'b j c -> b j c h', h=Hx)
        x_phase_h = torch.einsum('bjch,w->bjchw', shift_h, range_w_fft) / Wx
        # print(x_phase_h.shape, x_phase_h.shape)
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)
        # print(x_h.shape)
        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)

        shift_w = repeat(offsetw, 'b j c -> b j c w', w=Wx)
        x_phase_w = torch.einsum('bjcw,h->bjchw', shift_w, range_h_fft) / Hx
        # print(v_fft.shape)
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)  # [:, :, :H, :W]
        return out

    def rev(self, x, offseth, offsetw):
        Hx, Wx = x.shape[-2:]
        x = rearrange(x, 'b (head c) h w -> b head c h w', c=self.deform_kersize)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        shift_h = repeat(offseth, 'b j c -> b j c h', h=Hx)
        x_phase_h = torch.einsum('bjch,w->bjchw', shift_h, range_w_fft) / Wx
        # print(x_phase_h.shape, x_phase_h.shape)
        v_fft = v_fft * torch.exp(-1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)
        # print(x_h.shape)
        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)

        shift_w = repeat(offsetw, 'b j c -> b j c w', w=Wx)
        x_phase_w = torch.einsum('bjcw,h->bjchw', shift_w, range_h_fft) / Hx
        # print(v_fft.shape)
        v_fft = v_fft * torch.exp(-1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)  # [:, :, :H, :W]
        return out

    def forward(self, x_in, kernel):
        # print(x.shape)
        if self.reverse:
            alpha = 1j
        else:
            alpha = -1j
        b, head, kh, kw = kernel.shape
        H, W = x_in.shape[-2:]
        assert head == self.num_heads
        ker = kernel.view(b, head, -1)
        ker_value, coord = torch.topk(ker, k=self.deform_kersize, dim=-1, sorted=True)
        offseth = repeat(self.offseth.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=head)
        offsetw = repeat(self.offsetw.unsqueeze(0).unsqueeze(0), 'b c h w -> (rb b) (rc c) h w', rb=b, rc=head)
        offseth = offseth.contiguous().view(b, head, -1)
        offsetw = offsetw.contiguous().view(b, head, -1)
        # print(ker.shape, offseth.shape)
        offseth = torch.gather(offseth, dim=-1, index=coord)  # offseth[coord]
        offsetw = torch.gather(offsetw, dim=-1, index=coord)  # offsetw[coord]
        # print(offseth)
        # print(offsetw)
        # offset = torch.cat([offseth.unsqueeze(-1), offsetw.unsqueeze(-1)], dim=-1)
        # offset = offset.view(b, 1, 1, -1)
        if self.pad:
            x = self.pad_(x_in)
        else:
            x = x_in
        # print(x.shape, offseth.shape)
        x = check_image_size(x, 2)
        out = self.inv(x, offseth, offsetw)
        # print(out.shape, shift_w.shape, v_fft.shape)
        if self.div_ker:
            ker_value = self.proj1(ker_value)
            out = out * ker_value.view(b, self.num_heads, self.deform_kersize, 1, 1)

        # print(out.shape)
        out = rearrange(out, 'b head c h w -> b (head c) h w')[:, :, :H + self.pad * 2, :W + self.pad * 2]
        # print(out.shape, self.alpha.shape, self.deform_kersize)
        if self.pad:
            out = out[:, :, self.pad:-self.pad, self.pad:-self.pad]

        out = self.attn(out)
        if self.pad:
            x = self.pad_(out)
        else:
            x = out
        # print(x.shape, offseth.shape)
        x = check_image_size(x, 2)
        out = self.rev(x, offseth, offsetw)
        # print(out.shape, shift_w.shape, v_fft.shape)
        if self.div_ker:
            ker_value = self.proj2(ker_value)
            out = out * ker_value.view(b, self.num_heads, self.deform_kersize, 1, 1)

        # print(out.shape)
        out = rearrange(out, 'b head c h w -> b (head c) h w')[:, :, :H + self.pad * 2, :W + self.pad * 2]
        # print(out.shape, self.alpha.shape, self.deform_kersize)
        if self.pad:
            out = out[:, :, self.pad:-self.pad, self.pad:-self.pad]

        if self.deconv:
            x_deconv = self.xconv(x_in)
            x_deconv = featurefilter_deconv_fft(x_deconv, kernel)
            out = self.yconv(torch.cat([x_deconv, out], dim=1))
        # out = self.proj_out(out)
        return out  # / ker_value.view(b, self.deform_kersize, 1, 1)
class MyAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim*2, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class TopkAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, topk=16, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='spatial_global', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        self.topk = topk
        # print(topk)
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn_local(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        if self.channel_mlp:
            if 'grid' in self.cs:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                                h1=Hx // self.grid_size,
                                w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
            else:
                v = rearrange(v, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                                h1=Hx // self.window_size,
                                w1=Wx // self.window_size, h=self.window_size, w=self.window_size)
            out = out * self.cmlp(v)
        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        # print(1)
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        v_, idx = torch.topk(v, k=self.topk, dim=-1)
        # _, idx = torch.topk(torch.abs(v), k=self.topk, dim=-1)
        # v_ = torch.gather(v, dim=-1, index=idx)
        k = torch.gather(k, dim=-1, index=idx)
        q = torch.gather(q, dim=-1, index=idx)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            v_ = (attn @ v_.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            v_ = v_.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            v_ = (attn @ v_)
        if self.block_mlp:
            v_ = v_ * self.mlp(v)
        out = v.scatter(-1, idx, v_)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn_local(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops

def cal_feature_redundence(feature, dim):
    import pandas as pd
    feature = feature.squeeze(0).flatten(1).cpu().numpy().T
    ft_pd = pd.DataFrame(feature)
    p = ft_pd.corr(method='pearson')
    p_np = np.abs(np.array(p))
    p_mean = p_np.sum() / dim ** 2
    return p_mean
class ICCVAttention_qktest(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=False, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.dim = dim
        self.dct = DCT2x()
        self.idct = IDCT2x()
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        self.cont = 1
        win_start = 8
        win_end = 384
        # win_start = 0
        # win_end = 384 - 8
        step = 8
        self.win_list = range(win_start, win_end + 1, step)
        # self.win_list = self.win_list[::-1]
        print(self.win_list)
        self.mse_list = [0] * len(self.win_list)
        self.mse_list_ = [0] * len(self.win_list)
        self.js_list = [0] * len(self.win_list)
        self.cosine_list = [0] * len(self.win_list)
        n = self.window_size_dct // self.window_size
        self.cosine_matrix = np.zeros((n, n))
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]
    def get_attn_lt(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        num_w = self.window_size_dct // self.window_size
        # num_padh = Hx // self.window_size - num_w
        # num_padw = Wx // self.window_size - num_w
        num_padh = num_w
        num_padw = num_w
        # q = q[:, :num_w, :num_w, ...]
        # k = k[:, :num_w, :num_w, ...]
        q = q[:, num_w:, num_w:, ...]
        k = k[:, num_w:, num_w:, ...]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            attn = rearrange(attn, 'b h1 w1 head c1 c2 -> (b c2 head) c1 h1 w1')
            # print(attn.shape,num_padh)
            # attn = F.pad(attn, (0, num_padh, 0, num_padw), mode='replicate')
            attn = F.pad(attn, (num_padh, 0, num_padw, 0), mode='replicate')
            # print(attn.shape)
            attn = rearrange(attn, '(b c2 head) c1 h1 w1 -> b h1 w1 head c1 c2', c2=self.dim//self.num_heads, head=self.num_heads)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, 'b h1 w1 head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, 'b h1 w1 head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]
    def get_attn_local_cosine(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn_forward = (q @ k.transpose(-2, -1)) * self.temperature
        attn_forward = attn_forward.softmax(dim=-1)
        # num_w = self.window_size_dct // self.window_size

        q_global = rearrange(q, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
                             h=self.window_size, w=self.window_size)
        k_global = rearrange(k, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
                             h=self.window_size, w=self.window_size)
        attn_global = (q_global @ k_global.transpose(-2, -1)) * self.temperature
        attn_global = attn_global.softmax(dim=-1)
        # attn_global = rearrange(attn_global, 'b head c1 c2-> (b head c1) c2')
        # for i, win in enumerate(self.win_list):
        #     num_w = win // self.window_size
        #     # q_x = q[:, num_w:, num_w:, ...]
        #     # k_x = k[:, num_w:, num_w:, ...]
        #     q_x = q[:, :num_w, :num_w, ...]
        #     k_x = k[:, :num_w, :num_w, ...]
        #     q_x = rearrange(q_x, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
        #                          h=self.window_size, w=self.window_size)
        #     k_x = rearrange(k_x, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
        #                          h=self.window_size, w=self.window_size)
        #     attn_w = (q_x @ k_x.transpose(-2, -1)) * self.temperature
        #     attn_w = attn_w.softmax(dim=-1)
        #     attn_w = rearrange(attn_w, 'b head c1 c2-> (b head c1) c2')
        #     cosin_d = torch.cosine_similarity(attn_w, attn_global)
        #     # kl_d = KL_loss(attn_map_x_.log(), attn_map_)
        #     # js_d = JS_divergence_torch(attn_map_x_, attn_map_)
        #     self.cosine_list[i] += cosin_d.mean().item()
        row, col = q.shape[1:3]
        for i in range(row):
            for j in range(col):
                attn_i = attn_forward[:, i, i, ...]  # attn_global
                attn_i = rearrange(attn_i, 'b head c1 c2-> (b head c1) c2')
                attn_j = attn_forward[:, j, j, ...] 
                attn_j = rearrange(attn_j, 'b head c1 c2-> (b head c1) c2')
                self.cosine_matrix[i, j] += torch.cosine_similarity(attn_i, attn_j).mean().item()
        # attn_target = attn_forward[:, 0, 0, ...]
        # attn_target = attn_forward[:, row//2, col//2, ...] # attn_global
        # attn_target = rearrange(attn_target, 'b head c1 c2-> (b head c1) c2')
        # for i in range(row):
        #     for j in range(col):
        #         attn_source = attn_forward[:, i, j, ...]
        #         attn_source = rearrange(attn_source, 'b head c1 c2-> (b head c1) c2')
        #         self.cosine_matrix[i, j] += torch.cosine_similarity(attn_source, attn_target).mean().item()

        self.cont += 1
        cnt_end = 100
        if self.cont == cnt_end:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            out_root = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ICCV2023/attn_map/net_decoder_level1_last'
            os.makedirs(out_root, exist_ok=True)
            # L1_list_m = [i / self.cont for i in self.mse_list]
            # L1_list_m_ = [i / self.cont for i in self.mse_list_]
            # cosine_list_ = [i / self.cont for i in self.cosine_list]
            # pd.DataFrame(cosine_list_).to_excel(os.path.join(out_root, 'local_attn_image_softmax_cosine_LF.xlsx'), index=False,
            #                                   header=False)
            cosine_matrix_sns = self.cosine_matrix / cnt_end
            sns_plot = plt.figure()
            sns.heatmap(cosine_matrix_sns, cmap='RdBu_r', linewidths=0.01, vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False, cbar=True) # Reds_r .invert_yaxis()
            # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
            out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-other_LN_DCT_local' + '.png')
            sns_plot.savefig(out_way, dpi=700)
            plt.close()
        out = (attn_forward @ v)
        if self.block_mlp:
            out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, 'b h1 w1 head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, 'b h1 w1 head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]
    def get_attn_global_cosine(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z b h1 w1 head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn_forward = (q @ k.transpose(-2, -1)) * self.temperature
        attn_forward = attn_forward.softmax(dim=-1)
        # num_w = self.window_size_dct // self.window_size

        q_global = rearrange(q, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
                             h=self.window_size, w=self.window_size)
        k_global = rearrange(k, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
                             h=self.window_size, w=self.window_size)
        attn_global = (q_global @ k_global.transpose(-2, -1)) * self.temperature
        attn_global = attn_global.softmax(dim=-1)
        # attn_forward = attn_global
        row, col = q.shape[1:3]
        # attn_target = attn_forward[:, 0, 0, ...]

        for i in range(row):
            for j in range(col):
                attn_target = attn_forward[:, i, i, ...]  # attn_global
                attn_target = rearrange(attn_target, 'b head c1 c2-> (b head c1) c2')
                attn_source = attn_forward[:, j, j, ...]
                attn_source = rearrange(attn_source, 'b head c1 c2-> (b head c1) c2')
                self.cosine_matrix[i, j] += torch.cosine_similarity(attn_source, attn_target).mean().item()
        # attn_target = attn_forward[:, row//2, col//2, ...] # attn_global
        # attn_target = rearrange(attn_target, 'b head c1 c2-> (b head c1) c2')
        # for i in range(row):
        #     for j in range(col):
        #         attn_source = attn_forward[:, i, j, ...]
        #         attn_source = rearrange(attn_source, 'b head c1 c2-> (b head c1) c2')
        #         self.cosine_matrix[i, j] += torch.cosine_similarity(attn_source, attn_target).mean().item()

        self.cont += 1
        cnt_end = 100
        if self.cont == cnt_end:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            out_root = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ICCV2023/attn_map/net_decoder_level1_last'
            os.makedirs(out_root, exist_ok=True)
            # L1_list_m = [i / self.cont for i in self.mse_list]
            # L1_list_m_ = [i / self.cont for i in self.mse_list_]
            # cosine_list_ = [i / self.cont for i in self.cosine_list]
            # pd.DataFrame(cosine_list_).to_excel(os.path.join(out_root, 'local_attn_image_softmax_cosine_LF.xlsx'), index=False,
            #                                   header=False)
            cosine_matrix_sns = self.cosine_matrix / cnt_end
            sns_plot = plt.figure()
            sns.heatmap(cosine_matrix_sns, cmap='RdBu_r', linewidths=0.01, vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False, cbar=True) # Reds_r .invert_yaxis()
            # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_global' + '.png')
            out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-other_LN_DCT_global' + '.png')
            sns_plot.savefig(out_way, dpi=700)
        v = rearrange(v, 'b h1 w1 head c (h w) -> b head c (h1 h w1 w)',
                             h=self.window_size, w=self.window_size)
        out = (attn_global @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx, w=Wx)
        return out[:, :, :H, :W]
    def get_qk_w(self, attn_map, q_dct, k_dct):
        import pandas as pd

        MSE = nn.MSELoss()
        attn_map_ = attn_map.softmax(dim=-1)
        attn_map_ = rearrange(attn_map_, 'b head c1 c2 -> (b head c1) c2')
        # N = 1e6
        KL_loss = nn.KLDivLoss()
        for i, win in enumerate(self.win_list):
            # print(win, q_dct.shape)
            # x_tensor_dct[:, :, :win, :win] = y_tensor_dct[:, :, :win, :win]
            q = rearrange(q_dct[:, :, :, :win, :win], 'b head c h w -> b head c (h w)')
            k = rearrange(k_dct[:, :, :, :win, :win], 'b head c h w -> b head c (h w)')
            # q = rearrange(q_dct[:, :, :, win:, win:], 'b head c h w -> b head c (h w)')
            # k = rearrange(k_dct[:, :, :, win:, win:], 'b head c h w -> b head c (h w)')
            attn_map_x = q @ k.transpose(-1, -2) * self.temperature
            attn_map_x_ = attn_map_x.softmax(dim=-1)
            # print(attn_map_x_.shape, attn_map_.shape)
            attn_map_x_ = rearrange(attn_map_x_, 'b head c1 c2-> (b head c1) c2')
            cosin_d = torch.cosine_similarity(attn_map_x_, attn_map_)
            # kl_d = KL_loss(attn_map_x_.log(), attn_map_)
            # js_d = JS_divergence_torch(attn_map_x_, attn_map_)
            self.js_list[i] += cosin_d.mean().item()
            # mse = MSE(attn_map, attn_map_x)
            # self.mse_list[i] += mse.item() # / N
            # mse_ = MSE(attn_map_, attn_map_x_)
            # self.mse_list_[i] += mse_.item() # * N
            # print('attn_map', attn_map_)
            # print('attn_map_x', attn_map_x_)
        # print(self.cont, self.js_list)
        self.cont += 1
        if self.cont == 1111:
            out_root = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ICCV2023/attn_map/net_decoder_level1_last'
            os.makedirs(out_root, exist_ok=True)
            # L1_list_m = [i / self.cont for i in self.mse_list]
            # L1_list_m_ = [i / self.cont for i in self.mse_list_]
            js_list_m_ = [i / self.cont for i in self.js_list]
            pd.DataFrame(js_list_m_).to_excel(os.path.join(out_root, 'attn_image_softmax_cosine_HF.xlsx'), index=False,
                                              header=False)
            # pd.DataFrame(L1_list_m).to_excel(os.path.join(out_root, 'attn_image_mse.xlsx'), index=False,
            #                                  header=False)
            # pd.DataFrame(L1_list_m_).to_excel(os.path.join(out_root, 'attn_image_softmax_mse.xlsx'), index=False,
            #                                       header=False)
            # pd.DataFrame(self.win_list).to_excel(os.path.join(out_root, 'win_list.xlsx'), index=False, header=False)
    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) h w -> z b head c (h w)', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn_ = attn.softmax(dim=-1)
            out = (attn_ @ v)
        q = rearrange(q, 'b head c (h w) -> b head c h w', head=self.num_heads, h=H, w=W)
        k = rearrange(k, 'b head c (h w) -> b head c h w', head=self.num_heads, h=H, w=W)
        q_dct = self.dct(q)
        k_dct = self.dct(k)
        # print(self.window_size_dct)
        # q_dct = rearrange(q_dct[:, :, :, :self.window_size_dct, :self.window_size_dct], 'b head c h w -> b head c (h w)')
        # k_dct = rearrange(k_dct[:, :, :, :self.window_size_dct, :self.window_size_dct], 'b head c h w -> b head c (h w)')
        # q_dct = rearrange(q_dct[:, :, :, self.window_size_dct:, self.window_size_dct:],
        #                   'b head c h w -> b head c (h w)')
        # k_dct = rearrange(k_dct[:, :, :, self.window_size_dct:, self.window_size_dct:],
        #                   'b head c h w -> b head c (h w)')
        # attn_dct = q_dct @ k_dct.transpose(-1, -2) * self.temperature
        # attn_dct_ = attn_dct.softmax(dim=-1)
        # out = (attn_dct_ @ v)
        self.get_qk_w(attn, q_dct, k_dct)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn_local_cosine(qkv)
        else:
            out = self.get_attn_global_cosine(qkv)
        out = self.project_out(out)
        return out
class ShiftAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros'):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False

        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        self.gama = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head h (c w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head h (c w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c h w', head=self.num_heads)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # b head h h
        v_fft = torch.fft.rfft(v, dim=-1)
        # angle = torch.angle(x_fft)
        x_phase = rearrange(self.pool(attn), 'b head h c -> b (head c) h')
        x_phase = torch.sigmoid(x_phase) * self.gama # b head h c

        w_range = torch.arange(0, v_fft.shape[-1], device=v.device)

        x_phase = torch.einsum('bch,w->bchw', x_phase, w_range)

        v_fft = v_fft * torch.exp(1j * x_phase)

        out =  torch.fft.irfft(v_fft, dim=-1)

        return out[:, :, :H, :W]
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))

        out = self.get_attn(qkv)

        out = self.project_out(out)
        return out
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class ShiftAttentionV2(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False

        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k
        self.qkv = nn.Conv2d(dim, dim * 2+dim_k, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 2+dim_k, dim * 2+dim_k, kernel_size=3,
                                stride=1, padding=1, groups=dim * 2+dim_k, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        self.max_shift = max_shift
        self.norm_A1 = nn.LayerNorm(num_wave)
        self.norm_A2 = nn.LayerNorm(num_wave)
        self.norm_p1 = nn.LayerNorm(num_wave)
        self.norm_p2 = nn.LayerNorm(num_wave)
        self.norm_f1 = nn.LayerNorm(num_wave)
        self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        # self.norm = LayerNorm2d(dim)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]

        v, q, k = torch.split(qkv, [self.dim, self.dim, self.dim_k], dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c h w', head=self.num_heads)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # b head c num_wave*3
        attn = rearrange(attn, 'b head c d -> b (head c) d')

        # attn = attn
        A, f, p = attn.chunk(3, dim=-1)
        A1, A2 = A.chunk(2, dim=-1)
        f1, f2 = f.chunk(2, dim=-1)
        p1, p2 = p.chunk(2, dim=-1)
        A1 = self.norm_A1(A1)
        A2 = self.norm_A2(A2)
        f1 = self.norm_f1(f1)
        f2 = self.norm_f2(f2)
        p1 = self.norm_p1(p1)
        p2 = self.norm_p2(p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=v.device)
        range_w = torch.arange(0, Wx, device=v.device)
        v_fft = torch.fft.rfft(v, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=v.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1.unsqueeze(-1) * torch.sin(f1.unsqueeze(-1) * (range_h_ + p1.unsqueeze(-1))) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        shift_h = shift_h * self.gama1
        # print(shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        v_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(v_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=v.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2.unsqueeze(-1) * torch.sin(f2.unsqueeze(-1) * (range_w_ + p2.unsqueeze(-1))) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))

        out = self.get_attn(qkv)

        out = self.project_out(out)
        return out
class ShiftAttentionV3(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False

        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k
        self.qkv = nn.Conv2d(dim, dim * 3+dim_k, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3+dim_k, dim * 3+dim_k, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3+dim_k, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        self.max_shift = max_shift
        self.norm_A1 = nn.LayerNorm(num_wave)
        self.norm_A2 = nn.LayerNorm(num_wave)
        self.norm_p1 = nn.LayerNorm(num_wave)
        self.norm_p2 = nn.LayerNorm(num_wave)
        self.norm_f1 = nn.LayerNorm(num_wave)
        self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        # self.norm = LayerNorm2d(dim)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]

        v, vx, q, k = torch.split(qkv, [self.dim, self.dim, self.dim, self.dim_k], dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c h w', head=self.num_heads)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # b head c num_wave*3
        attn = rearrange(attn, 'b head c d -> b (head c) d')

        # attn = attn
        A, f, p = attn.chunk(3, dim=-1)
        A1, A2 = A.chunk(2, dim=-1)
        f1, f2 = f.chunk(2, dim=-1)
        p1, p2 = p.chunk(2, dim=-1)
        A1 = self.norm_A1(A1)
        A2 = self.norm_A2(A2)
        f1 = self.norm_f1(f1)
        f2 = self.norm_f2(f2)
        p1 = self.norm_p1(p1)
        p2 = self.norm_p2(p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=v.device)
        range_w = torch.arange(0, Wx, device=v.device)
        v_fft = torch.fft.rfft(v, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=v.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1.unsqueeze(-1) * torch.sin(f1.unsqueeze(-1) * (range_h_ + p1.unsqueeze(-1))) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]
        # print(shift_h.shape, shift_h[0, :, :])

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        v_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(v_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=v.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2.unsqueeze(-1) * torch.sin(f2.unsqueeze(-1) * (range_w_ + p2.unsqueeze(-1))) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W] * self.act(vx[:, :, :H, :W])
    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))

        out = self.get_attn(qkv)

        out = self.project_out(out)
        return out
class ShiftAttentionV4(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.A1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.A2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.p1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.p2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.f1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.f2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f2, a=math.sqrt(5))

        self.norm_A1 = nn.LayerNorm(num_wave)
        self.norm_A2 = nn.LayerNorm(num_wave)
        self.norm_p1 = nn.LayerNorm(num_wave)
        self.norm_p2 = nn.LayerNorm(num_wave)
        self.norm_f1 = nn.LayerNorm(num_wave)
        self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        # self.norm = LayerNorm2d(dim)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]

        A1 = self.norm_A1(self.A1)
        A2 = self.norm_A2(self.A2)
        f1 = self.norm_f1(self.f1)
        f2 = self.norm_f2(self.f2)
        p1 = self.norm_p1(self.p1)
        p2 = self.norm_p2(self.p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1.unsqueeze(-1) * torch.sin(f1.unsqueeze(-1) * (range_h_ + p1.unsqueeze(-1))) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2.unsqueeze(-1) * torch.sin(f2.unsqueeze(-1) * (range_w_ + p2.unsqueeze(-1))) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):

        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = self.get_attn(x1) * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftAttentionV5(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.A1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.A2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.p1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.p2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.f1 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)
        self.f2 = nn.Parameter(torch.Tensor(1, dim, num_wave), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f2, a=math.sqrt(5))

        # self.norm_A1 = nn.LayerNorm(num_wave)
        # self.norm_A2 = nn.LayerNorm(num_wave)
        # self.norm_p1 = nn.LayerNorm(num_wave)
        # self.norm_p2 = nn.LayerNorm(num_wave)
        # self.norm_f1 = nn.LayerNorm(num_wave)
        # self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        # self.norm = LayerNorm2d(dim)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]

        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        b1, c1, d1 = self.A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = self.A1.unsqueeze(-1) * torch.sin(self.f1.unsqueeze(-1) * (range_h_ + self.p1.unsqueeze(-1))) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = self.A2.unsqueeze(-1) * torch.sin(self.f2.unsqueeze(-1) * (range_w_ + self.p2.unsqueeze(-1))) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):

        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = self.get_attn(x1) * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftAttentionV6(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.wave = nn.Parameter(torch.Tensor(1, dim, num_wave, 6), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.wave, a=math.sqrt(5))

        # self.norm_A1 = nn.LayerNorm(num_wave)
        # self.norm_A2 = nn.LayerNorm(num_wave)
        # self.norm_p1 = nn.LayerNorm(num_wave)
        # self.norm_p2 = nn.LayerNorm(num_wave)
        # self.norm_f1 = nn.LayerNorm(num_wave)
        # self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        self.norm = WLayerNorm2d(dim, num_wave, 6)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        wave = self.norm(self.wave)

        A1, A2, f1, f2, p1, p2 = wave.chunk(6, dim=-1)
        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1 * torch.sin(f1 * (range_h_ + p1)) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2 * torch.sin(f2 * (range_w_ + p2)) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):

        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = self.get_attn(x1) * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftAttentionV7(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.wave = nn.Parameter(torch.Tensor(1, dim, num_wave, 6), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.wave, a=math.sqrt(5))

        # self.norm_A1 = nn.LayerNorm(num_wave)
        # self.norm_A2 = nn.LayerNorm(num_wave)
        # self.norm_p1 = nn.LayerNorm(num_wave)
        # self.norm_p2 = nn.LayerNorm(num_wave)
        # self.norm_f1 = nn.LayerNorm(num_wave)
        # self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        self.norm = WLayerNorm2d(dim, num_wave, 6)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        wave = self.norm(self.wave)

        A1, A2, f1, f2, p1, p2 = wave.chunk(6, dim=-1)
        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1 * torch.sin(f1 * (range_h_ + p1)) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w

        # print(shift_h.shape, shift_h[0, :, :])
        
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1

        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2 * torch.sin(f2 * (range_w_ + p2)) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):
        x = self.get_attn(x)
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = x1 * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftAttentionV8(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        # self.wave = nn.Parameter(torch.Tensor(1, dim, num_wave, 6), requires_grad=True)
        #
        # torch.nn.init.kaiming_uniform_(self.wave, a=math.sqrt(5))
        self.A1 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)
        self.A2 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)
        self.p1 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)
        self.p2 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)
        self.f1 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)
        self.f2 = nn.Parameter(torch.Tensor(1, dim, num_wave, 1), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.p2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.f2, a=math.sqrt(5))
        # self.norm_A1 = nn.LayerNorm(num_wave)
        # self.norm_A2 = nn.LayerNorm(num_wave)
        # self.norm_p1 = nn.LayerNorm(num_wave)
        # self.norm_p2 = nn.LayerNorm(num_wave)
        # self.norm_f1 = nn.LayerNorm(num_wave)
        # self.norm_f2 = nn.LayerNorm(num_wave)
        # self.norm_h = nn.LayerNorm(dim)
        # self.norm_w = nn.LayerNorm(dim)
        # self.norm = WLayerNorm2d(dim, num_wave, 6)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        self.wave_bias_1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias_2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # torch.nn.init.xavier_uniform_(self.wave_bias_1, gain=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.wave_bias_2, gain=math.sqrt(5))
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        # wave = self.norm(self.wave)

        # A1, A2, f1, f2, p1, p2 = self.wave.chunk(6, dim=-1)
        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        b1, c1, d1 = self.A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = self.A1 * torch.sin(self.f1 * (range_h_ + self.p1)) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) + self.wave_bias_1 # b c w
        # shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = self.A2 * torch.sin(self.f2 * (range_w_ + self.p2)) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False) + self.wave_bias_2 # b c w

        # shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):
       
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = self.get_attn(x1) * self.act(x2)

        out = self.project_out(out)
        return out

class ShiftAttentionNoshift(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros',
                 wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim  # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        # torch.nn.init.xavier_uniform_(self.wave_bias_1, gain=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.wave_bias_2, gain=math.sqrt(5))
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)


    def forward(self, x):

        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = x1 * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftAttentionV9(nn.Module):
    def __init__(self, dim, num_heads, num_wave, bias, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros',
                 wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim  # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        self.wave_bias_1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias_2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # torch.nn.init.xavier_uniform_(self.wave_bias_1, gain=math.sqrt(5))
        # torch.nn.init.xavier_uniform_(self.wave_bias_2, gain=math.sqrt(5))
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        # wave = self.norm(self.wave)

        # A1, A2, f1, f2, p1, p2 = self.wave.chunk(6, dim=-1)
        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        # b1, c1, d1 = self.A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        b, c, h, w = x.shape
        shift_h = torch.ones([b, c, h], device=x.device) * self.wave_bias_1  # b c w
        # shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft) / Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        shift_w = torch.ones([b, c, w], device=x.device) * self.wave_bias_2  # b c w
        # shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]

    def forward(self, x):

        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        # print(x1.device)
        out = self.get_attn(x1) * self.act(x2)

        out = self.project_out(out)
        return out
class ShiftOp(nn.Module):
    def __init__(self, dim, num_heads=1, num_wave=7, bias=True, max_shift=[7, 7], window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros', wave_bias=False):
        super().__init__()

        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim # -2

        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        dim_k = num_wave * 3 * 2 * num_heads
        self.num_wave = num_wave
        self.dim = dim
        self.dim_k = dim_k

        self.max_shift = max_shift
        self.wave = nn.Parameter(torch.Tensor(1, dim, num_wave, 6), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.wave, a=math.sqrt(5))

        # self.norm_A1 = nn.LayerNorm(num_wave)
        # self.norm_A2 = nn.LayerNorm(num_wave)
        # self.norm_p1 = nn.LayerNorm(num_wave)
        # self.norm_p2 = nn.LayerNorm(num_wave)
        # self.norm_f1 = nn.LayerNorm(num_wave)
        # self.norm_f2 = nn.LayerNorm(num_wave)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        self.norm = WLayerNorm2d(dim, num_wave, 6)
        # self.pool = nn.AdaptiveAvgPool2d((None, dim//num_heads))
        # self.gama1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias = wave_bias
        self.act = nn.GELU()
        # if wave_bias:
        #     self.w_bias = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        # self.gama = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        wave = self.norm(self.wave)

        A1, A2, f1, f2, p1, p2 = wave.chunk(6, dim=-1)
        # A1 = self.norm_A1(self.A1)
        # A2 = self.norm_A2(self.A2)
        # f1 = self.norm_f1(self.f1)
        # f2 = self.norm_f2(self.f2)
        # p1 = self.norm_p1(self.p1)
        # p2 = self.norm_p2(self.p2)

        b1, c1, d1 = A1.shape[:3]
        # print('A1: ', A1, f1, p1)
        # print(v_fft.shape)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        range_h_ = repeat(range_h.view(1, 1, 1, -1), 'b c d h -> (b b1) (c c1) (d d1) h', b1=b1, c1=c1, d1=d1)
        shift_h = A1 * torch.sin(f1 * (range_h_ + p1)) # 2 * torch.pi *
        shift_h = torch.sum(shift_h, dim=-2, keepdim=False) # b c w
        shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        # shift_h = shift_h * self.gama1
        # print(shift_h.shape, shift_h[0,:,:])
        # shift_h = torch.sigmoid(shift_h) * self.max_shift[0]

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft)/Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        range_w_ = repeat(range_w.view(1, 1, 1, -1), 'b c d w -> (b b1) (c c1) (d d1) w', b1=b1, c1=c1, d1=d1)
        shift_w = A2 * torch.sin(f2 * (range_w_ + p2)) # 2 * torch.pi *
        shift_w = torch.sum(shift_w, dim=-2, keepdim=False)  # b c w

        shift_w = self.norm_w(shift_w.transpose(-2, -1)).transpose(-2, -1)
        # shift_w = shift_w * self.gama2
        # shift_w = torch.sigmoid(shift_w) * self.max_shift[1]

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]
    def forward(self, x):
        x = self.get_attn(x)
        return x
class ShiftOp2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wave_bias_1 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)
        self.wave_bias_2 = nn.Parameter(torch.zeros((1, dim, 1)), requires_grad=True)

    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, 2)
        Hx, Wx = x.shape[-2:]
        # wave = self.norm(self.wave)
        range_h = torch.arange(0, Hx, device=x.device)
        range_w = torch.arange(0, Wx, device=x.device)
        v_fft = torch.fft.rfft(x, dim=-1)
        range_w_fft = torch.arange(0, v_fft.shape[-1], device=x.device)
        b, c, h, w = x.shape
        shift_h = torch.ones([b, c, h], device=x.device) * self.wave_bias_1  # b c w
        # shift_h = self.norm_h(shift_h.transpose(-2, -1)).transpose(-2, -1)

        x_phase_h = torch.einsum('bch,w->bchw', shift_h, range_w_fft) / Wx
        v_fft = v_fft * torch.exp(1j * x_phase_h)
        x_h = torch.fft.irfft(v_fft, dim=-1)

        v_fft = torch.fft.rfft(x_h, dim=-2)
        range_h_fft = torch.arange(0, v_fft.shape[-2], device=x.device)
        shift_w = torch.ones([b, c, w], device=x.device) * self.wave_bias_2  # b c w

        x_phase_w = torch.einsum('bcw,h->bchw', shift_w, range_h_fft) / Hx
        v_fft = v_fft * torch.exp(1j * x_phase_w)
        out = torch.fft.irfft(v_fft, dim=-2)

        return out[:, :, :H, :W]

    def forward(self, x):
        return self.get_attn(x)

class STN(nn.Module):
    def __init__(self, n):
        super(STN, self).__init__()

        # 定位网络-线性层
        self.localization_linear = nn.Sequential(
            nn.LayerNorm(n),
            nn.Linear(in_features=n, out_features=n//2),
            nn.ReLU(),
            nn.Linear(in_features=n//2, out_features=2 * 3)
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                  0, 1, 0], dtype=torch.float))
    # 空间变换器网络，转发图片张量
    def stn(self, x):
        b, c, h, w = x.shape
        x2 = x.view(b, c, -1)
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)  # [1, 2, 3]
        # print(theta)
        '''
        2D空间变换初始θ参数应置为tensor([[[1., 0., 0.],
                                        [0., 1., 0.]]])
        '''
        # 网格生成器，根据θ建立原图片的坐标仿射矩阵
        grid = F.affine_grid(theta, x.size(), align_corners=True)  # [1, 28, 28, 2]
        # 采样器，根据网格对原图片进行转换，转发给CNN分类网络
        x = F.grid_sample(x, grid, align_corners=True)  # [1, 1, 28, 28]
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b*c, 1, h, w)
        # x = rearrange(x, 'b c h w -> b c h w')
        x = self.stn(x)
        return x.view(b, c, h, w)

class ProAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8,window_size_dct=9, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True, global_bias=False,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, dconv='qkv', out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.temp_adj = temp_adj
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.global_bias = global_bias
        self.dconv = dconv
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        if dconv == 'qkv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        elif dconv == 'vdconv':
            self.qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode)
        elif dconv == 'qkvodconv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        else:
            self.qkv_dwconv = nn.Identity()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            if dconv == 'odconv':
                self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            elif dconv == 'qkvodconv':
                self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            elif dconv == 'oddconv':
                self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,
                                                           stride=1, padding=1, groups=dim, bias=bias,
                                                           padding_mode=padding_mode))
            elif dconv == 'odeformconv':
                self.project_out = nn.Sequential(MDeformConv2d(dim, dim, kernel_size=3,
                                                               stride=1, padding=1, groups=dim, bias=bias,
                                                               kernel_size_offset=3, offset_groups=1, kernel_size_mask=3),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            else:
                self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    def get_adjust(self, v):
        x_adjust = None
        # b = v.shape[0]
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        # if self.coarse_mlp:
            # _, _, v_ = qkv.chunk(3, dim=1)
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / (x_adjust + 1e-8)
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / x_adjust
            # if self.global_bias:
            #     global_bias = torch.einsum('i,j->ij', x, y)
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        # if self.coarse_mlp:
        #     # _, _, v_ = qkv.chunk(3, dim=1)
        #     v_in = torch.mean(v_[:, :, :self.window_size_dct, :self.window_size_dct], dim=1, keepdim=True)
        #     # v_in = out[:, :, :self.window_size_dct,:self.window_size_dct]
        #     out[:, :, :self.window_size_dct, :self.window_size_dct] = out[:, :, :self.window_size_dct,:self.window_size_dct] + \
        #                                                               self.mlp_coarse(v_in)

        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        if self.coarse_mlp:
            # _, _, v_ = qkv.chunk(3, dim=1)
            v_in = torch.mean(x[:, :, :self.window_size_dct, :self.window_size_dct], dim=1, keepdim=True)
            # v_in = out[:, :, :self.window_size_dct,:self.window_size_dct]
            x[:, :, :self.window_size_dct, :self.window_size_dct] = x[:, :, :self.window_size_dct,:self.window_size_dct] + \
                                                                      self.mlp_coarse(v_in)
        if self.dconv == 'vdconv':
            qkv = self.qkv(self.qkv_dwconv(x))
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out

def generate_center_mask(H, W, r, ir=False):

    mask = torch.zeros([H, W])

    for x in range(H):
        for y in range(W):
            tmp = (x - H//2) ** 2 + (y - W//2) ** 2
            # print(tmp, x, y)
            if tmp <= r ** 2:
                mask[x, y] = 1.
            else:
                continue
    if ir:
        mask = torch.ones_like(mask) - mask
    return mask
def generate_qircle_mask(H, W, r, ir=False):

    mask_ = generate_center_mask(H*2, W*2, r, ir=ir)
    # w = W//2
    # mask = torch.zeros([H, W])
    # mask[:H // 2, :w] = mask_[-H // 2:, -w:]
    # mask[-H // 2:, :] = mask_[:H // 2, -w:]
    # print(mask_[-H:, -W:].shape)
    return mask_[-H:, -W:]
class QCircleMask_Idx():
    def __init__(self, window_size_dct=16, in_dim=32, out_dim=32):
        super().__init__()

        self.window_size_dct = window_size_dct
        mask = generate_qircle_mask(window_size_dct, window_size_dct, r=window_size_dct, ir=False)

        self.mask_i = rearrange(mask, 'h w -> (h w)')
        self.mask = self.mask_i
        # newmask = mask.reshape(-1)
        newmask = self.mask_i.numpy().tolist()
        newmask = list(filter(lambda number: number != 0, newmask))

        self.cont = len(newmask)
        _, sorted_index = torch.sort(self.mask_i, dim=-1, descending=True)
        sorted_indext = sorted_index[:self.cont]
        self.sorted_indexx = repeat(sorted_indext, 'n -> c n', c=in_dim)
        self.sorted_indexy = repeat(sorted_indext, 'n -> c n', c=out_dim)
        # print(self.cont)
        # self.cont_z = self.cont
        # self.idx = mask.nonzero() # torch.where(self.mask > 0.5)
        self.size = None


        self.x_idx = None
        self.i = 0
    def forward_gather(self, x):
        x = rearrange(x, 'b c h w -> b c (h w)')
        # if self.x_idx is None:
        #     print('init_idx: ', self.i)
        #     self.i += 1
        #     # self.size = x.shape
        #     # self.mask = repeat(self.mask_i, 'n -> b c n', b=self.size[0], c=self.dim)
        #     # self.x_idx = torch.where(self.mask > 0.5) # self.mask.nonzero()
        #     # _, sorted_index = torch.sort(self.mask_i, dim=-1, descending=True)
        #     # print(sorted_index.shape, x.shape)
        #
        #     # self.cont_z = self.cont * self.size[0] * self.dim
        #     # self.x_idx = sorted_index[:, :, :self.cont]
        #     self.x_idx = repeat(self.sorted_index, 'n -> b c n', b=x.shape[0], c=self.dim)
        #     if self.out_dim == self.dim:
        #         self.y_idx = self.x_idx
        #     else:
        #         # self.masky = repeat(self.mask_i, 'n -> b c n', b=self.size[0], c=self.out_dim)
        #         # _, sorted_indexy = torch.sort(self.masky, dim=-1, descending=True)
        #         # self.y_idx = sorted_indexy[:, :, :self.cont]
        #         self.y_idx = repeat(self.sorted_index[:self.cont], 'n -> b c n', b=x.shape[0], c=self.out_dim)
        #         # self.size = [self.size[0], self.out_dim, self.size[-1]]
            # print(self.x_idx.shape, x.shape, self.cont)
        x_idx = repeat(self.sorted_indexx, 'c n -> b c n', b=x.shape[0]).to(x.device)
        # print(x_idx)
        # x_idx = self.x_idx.to(x.device)
        x_feat = x.gather(dim=-1, index=x_idx)

        # self.mask = repeat(self.mask_i, 'n -> b c n', b=x.shape[0], c=x.shape[1])
        # self.x_idx = torch.where(self.mask > 0.5) # self.mask.nonzero()
        # _, sorted_index = torch.sort(self.mask, dim=-1, descending=True)
        # x_idx2 = sorted_index[:, :, :self.cont].to(x.device)
        # print('sub', torch.mean(x.gather(dim=-1, index=x_idx2) - x_feat))
        # print('x_feat: ', x_feat.shape)
        return x_feat # .view(self.size[0], self.dim, -1)
    def backward_scatter(self, x):
        y_idx = repeat(self.sorted_indexy, 'c n -> b c n', b=x.shape[0]).to(x.device)
        # print(y_idx.shape, x.shape)
        z = torch.zeros([x.shape[0], x.shape[1], self.window_size_dct**2], device=x.device)
        # y_idx = self.y_idx.to(x.device)
        # z[self.x_idx] = x.view(-1)
        return rearrange(z.scatter(-1, y_idx, x), 'b c (h w) -> b c h w', h=self.window_size_dct, w=self.window_size_dct)
class CoarseFreqAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=16, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True, global_bias=False,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, dconv='qkv', out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.temp_adj = temp_adj
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.global_bias = global_bias
        self.dconv = dconv
        self.window_size_dct = window_size_dct
        self.QC = QCircleMask_Idx(window_size_dct=window_size_dct, in_dim=dim*3, out_dim=dim)
        # self.mask = generate_center_mask(window_size_dct, window_size_dct, dim, r=window_size_dct, ir=False)
        # self.idx = self.mask.nonzero() # torch.where(self.mask > 0.5)
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False

        # if 'grid' in self.cs:
        #     N = grid_size ** 2
        #     self.k = grid_size
        # else:
        #     N = window_size ** 2
        #     self.k = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        if dconv == 'qkv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=0, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        else:
            self.qkv_dwconv = nn.Identity()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        # H, W = qkv.shape[-2:]

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qkv = self.QC.forward_gather(qkv)
        qkv = rearrange(qkv, 'b (z head c) n -> z b head c n', z=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        # if self.block_mlp:
        #     out = out * self.mlp(v)
        # elif self.block_graph:
        #     v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
        #     v_graph = self.graph(v_graph)
        #     v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #     out = out * v_graph

        out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        out = self.QC.backward_scatter(out)
        return out

    def forward(self, x):
        # H, W = x.shape[-2:]
        temp = torch.zeros_like(x)
        x = x[:, :, :self.window_size_dct+2, :self.window_size_dct+2]
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        temp[:, :, :self.window_size_dct, :self.window_size_dct] = out
        return temp
class CoarseMLP(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size_dct=16,):
        super().__init__()

        # self.skip_mul = skip_mul
        self.num_heads = num_heads

        self.window_size_dct = window_size_dct
        self.QC = QCircleMask_Idx(window_size_dct=window_size_dct, in_dim=dim, out_dim=dim)
        cnt = self.QC.cont
        self.mlp = nn.Sequential(nn.Linear(cnt, cnt*2, bias=bias),
                                 # nn.Dropout(0.5),
                                 # nn.GELU(),
                                 # nn.Linear(cnt, cnt, bias=bias),
                                 # nn.Dropout(0.5)
                                 # nn.GELU(),
                                 )
        self.act = nn.GELU()
        

    def get_attn(self, x):

        x = self.QC.forward_gather(x)
        # x, _ = self.lstm(x)
        # x = self.mlp(x)
        x1, x2 = self.mlp(x).chunk(2, dim=-1)
        x = x1 * x2
        x = self.QC.backward_scatter(x)
        return x

    def forward(self, x):
        out = self.get_attn(x)
        # if self.skip_mul:
        #     out = out * x
        return out
class CoarseFreqMLP(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size_dct=16, skip_mul=True,):
        super().__init__()

        self.skip_mul = skip_mul
        self.num_heads = num_heads

        self.window_size_dct = window_size_dct
        self.QC = QCircleMask_Idx(window_size_dct=window_size_dct, in_dim=dim, out_dim=dim)
        cnt = self.QC.cont
        # num_layers = 1
        # bidirectional = True
        # dim_lstm = dim * 2 if bidirectional else dim
        self.c_mlp_in = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, dim, bias=bias),
                                      # nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers,
                                      #         bias=bias, batch_first=True, dropout=0., bidirectional=False),
                                      )
        # self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers,
        #                     bias=bias, batch_first=True, dropout=0., bidirectional=bidirectional)

        self.mlp = nn.Sequential(nn.Linear(cnt, cnt, bias=bias),
                                 nn.Dropout(0.5),
                                 nn.GELU(),
                                 nn.Linear(cnt, cnt, bias=bias),
                                 nn.Dropout(0.5)
                                 # nn.GELU(),
                                 )
        self.c_mlp_out = nn.Sequential(nn.Linear(dim, dim, bias=bias),)

    def get_attn(self, x):

        x = self.QC.forward_gather(x)
        x = self.c_mlp_in(x.transpose(-1, -2)).transpose(-1, -2)
        # x, _ = self.lstm(x)
        x = self.mlp(x)
        x = self.c_mlp_out(x.transpose(-1, -2)).transpose(-1, -2)
        # print(x.shape)
        # x = self.mlp(x.transpose(-1, -2)) # * x
        # x = self.c_mlp_out(x.transpose(-1, -2)).transpose(-1, -2)
        # out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        x = self.QC.backward_scatter(x)
        return x

    def forward(self, x):
        # H, W = x.shape[-2:]
        temp = torch.zeros_like(x)
        x = x[:, :, :self.window_size_dct, :self.window_size_dct] # +2

        out = self.get_attn(x)
        if self.skip_mul:
            out = out * x
        # out = self.project_out(out)
        temp[:, :, :self.window_size_dct, :self.window_size_dct] = out
        return temp
class CoarseFreqLSTM(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size_dct=16, skip_mul=True,):
        super().__init__()

        self.skip_mul = skip_mul
        self.num_heads = num_heads

        self.window_size_dct = window_size_dct
        self.QC = QCircleMask_Idx(window_size_dct=window_size_dct, in_dim=dim, out_dim=dim)
        cnt = self.QC.cont
        num_layers = 1
        bidirectional = True
        dim_lstm = dim * 2 if bidirectional else dim
        self.c_mlp_in = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, dim, bias=bias),
                                      # nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers,
                                      #         bias=bias, batch_first=True, dropout=0., bidirectional=False),
                                      )
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=num_layers,
                            bias=bias, batch_first=True, dropout=0., bidirectional=bidirectional)

        # self.mlp = nn.Sequential(nn.Linear(cnt, cnt, bias=bias),
        #                          nn.GELU(),
        #                          nn.Linear(cnt, cnt, bias=bias),
        #                          nn.Dropout(0.5)
        #                          # nn.GELU(),
        #                          )
        self.c_mlp_out = nn.Sequential(nn.Linear(dim_lstm, dim, bias=bias),)

    def get_attn(self, x):

        x = self.QC.forward_gather(x)
        x = self.c_mlp_in(x.transpose(-1, -2))
        x, _ = self.lstm(x)
        x = self.c_mlp_out(x).transpose(-1, -2)
        # print(x.shape)
        # x = self.mlp(x.transpose(-1, -2)) # * x
        # x = self.c_mlp_out(x.transpose(-1, -2)).transpose(-1, -2)
        # out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        x = self.QC.backward_scatter(x)
        return x

    def forward(self, x):
        # H, W = x.shape[-2:]
        temp = torch.zeros_like(x)
        x = x[:, :, :self.window_size_dct, :self.window_size_dct] # +2

        out = self.get_attn(x)
        if self.skip_mul:
            out = out * x
        # out = self.project_out(out)
        temp[:, :, :self.window_size_dct, :self.window_size_dct] = out
        return temp
class ProGridAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True, global_bias=False,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, dconv='qkv', out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.temp_adj = temp_adj
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        # self.grid_size = grid_size
        self.cs = cs
        self.global_bias = global_bias
        self.dconv = dconv
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        if dconv == 'qkv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        elif dconv == 'vdconv':
            self.qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode)
        else:
            self.qkv_dwconv = nn.Identity()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            if dconv == 'odconv':
                self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            elif dconv == 'odeformconv':
                self.project_out = nn.Sequential(MDeformConv2d(dim, dim, kernel_size=3,
                                                               stride=1, padding=1, groups=dim, bias=bias,
                                                               kernel_size_offset=3, offset_groups=1, kernel_size_mask=3),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            else:
                self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    def get_adjust(self, v):
        x_adjust = None
        # b = v.shape[0]
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)

        Hx, Wx = qkv.shape[-2:]
        grid_sizeh = Hx // self.window_size
        grid_sizew = Wx // self.window_size
        # if 'grid' in self.cs:
        qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                        h=grid_sizeh, w=grid_sizew)
        # else:
        #     qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
        #                     h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / (x_adjust + 1e-8)
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / x_adjust
            # if self.global_bias:
            #     global_bias = torch.einsum('i,j->ij', x, y)
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        # if 'grid' in self.cs:
        out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//grid_sizeh,
                        w1=Wx//grid_sizew, h=grid_sizeh, w=grid_sizew)
        # else:
        #     out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
        #                     w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        if self.dconv == 'vdconv':
            qkv = self.qkv(self.qkv_dwconv(x))
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out

class GProAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True, global_bias=False,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, dconv='qkv', out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.temp_adj = temp_adj
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.global_bias = global_bias
        self.dconv = dconv
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        if dconv == 'qkv':
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        elif dconv == 'vdconv':
            self.qkv_dwconv = nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode)
        else:
            self.qkv_dwconv = nn.Identity()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            if dconv == 'odconv':
                self.project_out = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim, bias=bias, padding_mode=padding_mode),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            elif dconv == 'odeformconv':
                self.project_out = nn.Sequential(MDeformConv2d(dim, dim, kernel_size=3,
                                                               stride=1, padding=1, groups=dim, bias=bias,
                                                               kernel_size_offset=3, offset_groups=1, kernel_size_mask=3),
                                                 nn.Conv2d(dim, dim, kernel_size=1, bias=True))
            else:
                self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    def get_adjust(self, v):
        x_adjust = None
        # b = v.shape[0]
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z b (h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z b (h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / (x_adjust + 1e-8)
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / x_adjust
            if self.global_bias:
                global_bias = torch.einsum('bxhcn,bxhjn->bhcj', q, k)
                attn = attn * global_bias.unsqueeze(1)
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b x head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b x (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        if 'grid' in self.cs:
            out = rearrange(out, 'b (h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, 'b (h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        if self.dconv == 'vdconv':
            qkv = self.qkv(self.qkv_dwconv(x))
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out
class ProComplexAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)

        qkv = check_image_size(qkv, self.window_size)
        # qkv_real, qkv_imag = qkv.chunk(2, dim=1)
        # qkv = torch.complex(qkv_real, qkv_imag)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out
class SProAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.1, dim_k=6, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.dim = dim
        self.dim_k = dim_k
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        # if self.block_mlp:
        #
        #     self.mlp = nn.Sequential(
        #         # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
        #         # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
        #         nn.Linear(N, N, bias=True),
        #         nn.GELU(),
        #         # nn.Linear(N, N, bias=True),
        #         # nn.Conv2d(dim, dim, 1, bias=True)
        #     )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.x_conv = nn.Sequential(nn.Conv2d(dim, dim+dim_k, kernel_size=1, bias=bias),
                                    nn.Conv2d(dim+dim_k, dim+dim_k, kernel_size=3,
                                              stride=1, padding=1, groups=dim + dim_k, bias=bias, padding_mode=padding_mode)
                                    )

        self.qkv_conv = nn.Linear(dim, dim * 3, bias=bias)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        # H, W = qkv.shape[-2:]

        qkv = rearrange(qkv, 'b n (z head c) -> z b head c n', z=3, head=self.num_heads)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        # elif self.block_graph:
        #     v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
        #     v_graph = self.graph(v_graph)
        #     v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #     out = out * v_graph

        out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        return out

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.x_conv(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x, x_k = torch.split(x, [self.dim, self.dim_k], dim=1)
        x_k, _ = torch.topk(x_k, dim=1, k=2)
        x_k_diff = x_k[:, 0, :] - x_k[:, 1, :]

        _, sorted_index = torch.sort(x_k_diff, dim=-1, descending=True)
        # print(sorted_index.shape, x.shape)
        high_idx = sorted_index[:, :int(H*W*self.percent_mask)]
        # high_idx = high_idx.unsqueeze(1)
        high_idx = repeat(high_idx, 'd n -> d c n', c=self.dim)
        high_feat = x.gather(dim=-1, index=high_idx)
        # print(high_feat.shape)
        qkv = self.qkv_conv(high_feat.transpose(-1, -2))
        out_k = self.get_attn(qkv)
        out = x.scatter(-1, high_idx, out_k)
        # print(out.shape, out_k.shape)
        out = rearrange(out, 'b c (h w) -> b c h w', h=H, w=W)
        out = self.project_out(out)
        return out
class MSProAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.1, dim_k=6, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.dim = dim
        self.dim_k = dim_k
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        # if self.block_mlp:
        #
        #     self.mlp = nn.Sequential(
        #         # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
        #         # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
        #         nn.Linear(N, N, bias=True),
        #         nn.GELU(),
        #         # nn.Linear(N, N, bias=True),
        #         # nn.Conv2d(dim, dim, 1, bias=True)
        #     )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.x_conv = nn.Sequential(nn.Conv2d(dim, dim+dim_k, kernel_size=1, bias=bias),
                                    nn.Conv2d(dim+dim_k, dim+dim_k, kernel_size=3,
                                              stride=1, padding=1, groups=dim + dim_k, bias=bias, padding_mode=padding_mode)
                                    )

        self.qkv_conv = nn.Linear(dim, dim * 3, bias=bias)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        # H, W = qkv.shape[-2:]

        qkv = rearrange(qkv, 'b n (z head c) -> z b head c n', z=3, head=self.num_heads)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        # elif self.block_graph:
        #     v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
        #     v_graph = self.graph(v_graph)
        #     v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #     out = out * v_graph

        out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        return out

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.x_conv(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x, x_k = torch.split(x, [self.dim, self.dim_k], dim=1)

        x_k = torch.softmax(x_k, dim=1)
        _, sorted_index = torch.sort(x_k, dim=-1, descending=True)
        # print(sorted_index.shape, x.shape)
        high_idx = sorted_index[:, :, :int(H*W*self.percent_mask)]
        # high_idx = high_idx.unsqueeze(1)
        out = torch.zeros_like(x)
        for i in range(high_idx.shape[1]):
            high_idx = repeat(high_idx[:, i, :], 'd n -> d c n', c=self.dim)
            high_feat = x.gather(dim=-1, index=high_idx)
            # print(high_feat.shape)
            qkv = self.qkv_conv(high_feat.transpose(-1, -2))
            out_k = self.get_attn(qkv)
            out = out.scatter(-1, high_idx, out_k)
        # out = x.scatter(-1, high_idx, out_k)
        # print(out.shape, out_k.shape)
        out = rearrange(out, 'b c (h w) -> b c h w', h=H, w=W)
        out = self.project_out(out)
        return out
class MSubProAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.1, dim_k=6, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.dim = dim
        self.dim_k = dim_k
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            self.N = grid_size ** 2
            self.k = grid_size
        else:
            self.N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(self.N, self.N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.x_conv = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias),
                                    nn.Conv2d(dim*2, dim*2, kernel_size=3,
                                              stride=1, padding=1, groups=dim*2, bias=bias, padding_mode=padding_mode)
                                    )

        self.qkv_conv = nn.Linear(dim, dim * 3, bias=bias)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        # H, W = qkv.shape[-2:]

        qkv = rearrange(qkv, 'b n (z head c) -> z b head c n', z=3, head=self.num_heads)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph

        out = rearrange(out, 'b head c n -> b (head c) n', head=self.num_heads)
        return out

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.x_conv(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x, x_k = torch.split(x, [self.dim, self.dim], dim=1)
        x_k = torch.abs(x-x_k)
        x_k, _ = torch.topk(x_k, dim=1, k=self.dim_k)
        x_k = torch.softmax(x_k, dim=1)
        x_k, sorted_index = torch.sort(x_k, dim=-1, descending=True)
        # print(sorted_index.shape, x.shape)
        high_idx = sorted_index[:, :, :self.N] # int(H*W*self.percent_mask)]
        x_k = x_k[:, :, :self.N]
        # high_idx = high_idx.unsqueeze(1)
        out = x
        b, c, nx = x.shape
        out_i = torch.zeros([self.dim_k, b, c, nx], device=x.device)
        for i in range(high_idx.shape[1]):
            # outi = torch.zeros_like(x)
            high_idx = repeat(high_idx[:, i, :], 'd n -> d c n', c=self.dim)
            high_feat = x.gather(dim=-1, index=high_idx)
            # print(high_feat.shape)
            qkv = self.qkv_conv(high_feat.transpose(-1, -2))
            out_k = self.get_attn(qkv)
            out_i[i, :, :, :] = out_i[i, :, :, :].scatter_(-1, high_idx, out_k*x_k[:, i, :].unsqueeze(1))
            out = out + out_i[i, :, :, :]

        # out = x.scatter(-1, high_idx, out_k)
        # print(out.shape, out_k.shape)
        # out = out
        out = rearrange(out, 'b c (h w) -> b c h w', h=H, w=W)
        out = self.project_out(out)
        return out
class Sparse_act(nn.Module):
    def __init__(self, dim, bias, window_size=8,
                 act=nn.ReLU, padding_mode='zeros', cs='window', percent_mask=0.5, dim_k=2, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.act = act
        self.dct = dct
        self.dim = dim
        self.dim_k = dim_k

        self.window_size = window_size
        self.percent_mask = percent_mask
        self.act = act()

        self.cs = cs
        if 'grid' in self.cs:
            self.N = int(window_size ** 2 * percent_mask)
            self.k = window_size
        else:
            self.N = int(window_size ** 2 * percent_mask)
            self.k = window_size

        self.x_conv = nn.Sequential(nn.Conv2d(dim, 2+dim, kernel_size=1, bias=bias),
                                    nn.Conv2d(2+dim, 2+dim, kernel_size=3,
                                              stride=1, padding=1, groups=2+dim, bias=bias, padding_mode=padding_mode)
                                    )


    def forward(self, x):
        H, W = x.shape[-2:]
        v = self.x_conv(x)
        v = check_image_size(v, self.window_size)
        # x = check_image_size(x, self.window_size)
        # x = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c (h w)', h=self.window_size, w=self.window_size)
        v = rearrange(v, 'b c (h1 h) (w1 w) -> (b h1 w1) c (h w)', h=self.window_size, w=self.window_size)
        x, v = torch.split(v, [self.dim, 2], dim=1)
        # out = x.clone()
        x_k, _ = torch.topk(v, dim=1, k=2)
        # v1, v2 = torch.split(x, [1, 1], dim=1)
        x_k = torch.abs(x_k[:, 0, :] - x_k[:, 1, :])

        x_k, sorted_index = torch.sort(x_k, dim=-1, descending=True)
        # print(sorted_index.shape, x.shape)
        high_idx = sorted_index[:, :self.N] # int(H*W*self.percent_mask)]

        high_idx = repeat(high_idx, 'd n -> d c n', c=self.dim)
        high_feat = x.gather(dim=-1, index=high_idx)
        high_feat = self.act(high_feat)
        
        # out = torch.ones_like(x)
        out = x * torch.scatter(x, -1, high_idx, high_feat) # x.scatter_(-1, high_idx, high_feat)
        out = rearrange(out, '(b h1 w1) c (h w) -> b c (h1 h) (w1 w)', h1=H//self.window_size,
                        w1=W//self.window_size, h=self.window_size, w=self.window_size )

        return out[:, :, :H, :W]
class ProReAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.reatten_matrix = nn.Conv2d(self.num_heads, self.num_heads, 1, 1)
        # self.var_norm = nn.InstanceNorm2d(self.num_heads)
        N_s = N if 'spatial' in self.cs else dim // num_heads
        self.dynamic_scale = nn.Parameter(torch.ones(1, N_s, N_s), requires_grad=True)
        self.inner_bias = nn.Parameter(torch.zeros(num_heads, N_s, N_s), requires_grad=True)
        self.outer_bias = nn.Parameter(torch.zeros(num_heads, N_s, N_s), requires_grad=True)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn * self.dynamic_scale + self.inner_bias
            attn = attn.softmax(dim=-1)
            # attn = self.var_norm(self.reatten_matrix(attn))
            attn = self.reatten_matrix(attn) + self.outer_bias
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn * self.dynamic_scale + self.inner_bias
            attn = attn.softmax(dim=-1)
            # attn = self.var_norm(self.reatten_matrix(attn))
            attn = self.reatten_matrix(attn) + self.outer_bias
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out
class DualAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=True, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim = dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = True if 'mlp' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        if 'grid' in self.cs:
            N = grid_size ** 2
            self.k = grid_size
        else:
            N = window_size ** 2
            self.k = window_size
        if self.block_mlp:

            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        elif self.block_graph:
            self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 6, dim * 6, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 6, bias=bias, padding_mode=padding_mode)
        self.norm1 = LayerNorm(dim)
        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (z head c) (h h1) (w w1) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        elif self.block_graph:
            v_graph = rearrange(v, 'b head c (h w) -> b (head c) h w', h=self.k, w=self.k)
            v_graph = self.graph(v_graph)
            v_graph = rearrange(v_graph, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = out * v_graph
        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads, h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        return out[:, :, :H, :W]
    def get_fft(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = qkv.shape[-2:]
        # print(qk.shape, v.shape)
        if 'grid' in self.cs:
            qk = rearrange(qk, 'b (z c) (h h1) (w w1) -> z (b h1 w1) c h w', z=2,
                            h=self.grid_size, w=self.grid_size)
        else:
            qk = rearrange(qk, 'b (z c) (h1 h) (w1 w) -> z (b h1 w1) c h w', z=2,
                            h=self.window_size, w=self.window_size)
        qk = torch.fft.rfft2(qk)
        q, k = qk[0], qk[1]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        out = q * torch.conj(k)
        out = torch.fft.irfft2(out)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h h1) (w w1)', h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        out = self.norm1(out)
        out= out * v

        return out[:, :, :H, :W]
    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv1, qkv2 = qkv.chunk(2, dim=1)
        out1 = self.get_attn(qkv1)
        out2 = self.get_fft(qkv2)
        out = torch.cat([out1, out2], dim=1)
        out = self.project_out(out)
        return out
class PhaseAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim=dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs


        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        self.norm1 = LayerNorm(dim)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = qkv.shape[-2:]
        # print(qk.shape, v.shape)
        if 'grid' in self.cs:
            qk = rearrange(qk, 'b (z c) (h h1) (w w1) -> z (b h1 w1) c h w', z=2,
                            h=self.grid_size, w=self.grid_size)
        else:
            qk = rearrange(qk, 'b (z c) (h1 h) (w1 w) -> z (b h1 w1) c h w', z=2,
                            h=self.window_size, w=self.window_size)
        qk = torch.fft.rfft2(qk)
        # qk = torch.exp(torch.angle(qk) * 1j)
        q, k = qk[0], qk[1]
        k = torch.exp(torch.angle(k) * 1j)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        out = q * torch.conj(k)
        out = torch.fft.irfft2(out)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h h1) (w w1)', h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        out = self.norm1(out)
        out= out * v

        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out
class FFTSAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim=dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        self.norm1 = LayerNorm(dim)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = qkv.shape[-2:]
        # print(qk.shape, v.shape)
        if 'grid' in self.cs:
            qk = rearrange(qk, 'b (z c) (h h1) (w w1) -> z (b h1 w1) c h w', z=2,
                            h=self.grid_size, w=self.grid_size)
        else:
            qk = rearrange(qk, 'b (z c) (h1 h) (w1 w) -> z (b h1 w1) c h w', z=2,
                            h=self.window_size, w=self.window_size)
        qk = torch.fft.rfft2(qk)
        q, k = qk[0], qk[1]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        out = q * torch.conj(k)
        out = torch.fft.irfft2(out)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h h1) (w w1)', h1=Hx//self.grid_size,
                            w1=Wx//self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=Hx//self.window_size,
                            w1=Wx//self.window_size, h=self.window_size, w=self.window_size)
        out = self.norm1(out)
        out= out * v

        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out
class FFTOrthoAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim=dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        self.ortho = OrthoT2d(window_size, window_size)
        self.norm1 = LayerNorm(dim)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = qkv.shape[-2:]
        window_sizeh, window_sizew = Hx // self.window_size, Wx // self.window_size
        qk = rearrange(qk, 'b (z c) (h1 h) (w1 w) -> z b c h w h1 w1', z=2,
                        h=window_sizeh, w=window_sizew)
        qk = self.ortho(qk, False)
        qk = rearrange(qk, 'z b c h w h1 w1 -> z b c h1 w1 h w ')
        qk = torch.fft.rfft2(qk)
        q, k = qk[0], qk[1]
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        out = q * torch.conj(k)
        out = torch.fft.irfft2(out)
        out = rearrange(out, 'b c h1 w1 h w -> b c h w h1 w1')
        out = self.ortho(out, True)
        out = rearrange(out, 'b c h w h1 w1 -> b c (h1 h) (w1 w)') # , h1=self.window_size, w1=self.window_size, h=window_sizeh, w=window_sizew
        out = self.norm1(out)
        out = out * v

        return out[:, :, :H, :W]

    def forward(self, x):
        # _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        out = self.get_attn(qkv)
        out = self.project_out(out)
        return out

class OrthoLSTM(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None, num_layers=1, bidirectional=True):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim=dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.qkv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3,
                                    stride=1, padding=1, groups=dim*2, bias=bias, padding_mode=padding_mode)
        self.ortho = OrthoT2d(window_size, window_size)
        if 'hv' in cs:
            lstm_dim = dim//2
            self.lstm_h = nn.LSTM(input_size=lstm_dim,hidden_size=lstm_dim,num_layers=num_layers,
                            bias=True,batch_first=True,dropout=0.,bidirectional=True)
            self.lstm_v = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, num_layers=num_layers,
                                      bias=True, batch_first=True, dropout=0., bidirectional=True)
            hid_dim = lstm_dim * 4 if bidirectional else lstm_dim * 2
        else:
            lstm_dim = dim
            self.lstm = nn.LSTM(input_size=lstm_dim, hidden_size=lstm_dim, num_layers=num_layers,
                                  bias=True, batch_first=True, dropout=0., bidirectional=True)
            hid_dim = lstm_dim * 2 if bidirectional else lstm_dim 
        
        self.norm1 = LayerNorm(hid_dim)
        if proj_out:
            self.project_out = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, x):
        H, W = x.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        x = check_image_size(x, self.window_size)
        # qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = x.shape[-2:]
        # if 'grid' in self.cs:
        k = self.grid_size if 'grid' in self.cs else self.window_size
        window_sizeh, window_sizew = Hx // k, Wx // k
        if 'grid' in self.cs:
            x = rearrange(x, 'b c (h1 h) (w1 w) -> b c h w h1 w1', h=window_sizeh, w=window_sizew)
        else:
            x = rearrange(x, 'b c (h h1) (w w1) -> b c h w h1 w1', h=window_sizeh, w=window_sizew)
        x = self.ortho(x, False)

        if 'hv' in self.cs:
            x_h, x_w = x.chunk(2, dim=1)
            x_w = rearrange(x_w, 'b c h w h1 w1 -> (b h1 h w1) w c')
            x_h = rearrange(x_h, 'b c h w h1 w1 -> (b h1 w w1) h c')
            x_w, _ = self.lstm_h(x_w)
            x_h, _ = self.lstm_v(x_h)
            x_w = rearrange(x_w, '(b h1 w w1) h c -> b c h w h1 w1', h=window_sizeh, w=window_sizew, h1=k, w1=k)
            x_h = rearrange(x_h, '(b h1 w w1) h c -> b c h w h1 w1', h=window_sizeh, w=window_sizew, h1=k, w1=k)
            out = torch.cat([x_h, x_w], dim=1)
        else:
            x = rearrange(x, 'b c h w h1 w1 -> (b h1 w1) (w h) c')
            x, _ = self.lstm(x)
            out = rearrange(x, '(b h1 w1) (w h) c -> b c h w h1 w1', h=window_sizeh, w=window_sizew, h1=k, w1=k)
        # out = rearrange(out, 'b c h1 w1 h w -> b c h w h1 w1')
        out = self.ortho(out, True)
        if 'grid' in self.cs:
            out = rearrange(out, 'b c h w h1 w1 -> b c (h1 h) (w1 w)')  # , h1=self.window_size, w1=self.window_size, h=window_sizeh, w=window_sizew
        else:
            out = rearrange(out, 'b c h w h1 w1 -> b c (h h1) (w w1)')
        out = self.norm1(out[:, :, :H, :W])
        # out = out * v

        return out

    def forward(self, x):
        # _, _, h, w = x.shape
        x = self.qkv_dwconv(self.qkv(x))
        x, v = x.chunk(2, dim=1)
        # _, _, H, W = qkv.shape
        out = self.project_out(self.get_attn(x)) * v
        # out = out
        return out


class OrthoSample(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None, num_layers=1,
                 hv=False):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim = dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim  # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.hv = hv
        self.grid = True if 'grid' in cs else False
        self.k = grid_size if 'grid' in cs else window_size
        self.qkv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        inference = True if 'inference' in cs else False
        self.winp = WindowPartition(self.k)
        self.winr = WindowReverse(self.k)
        if hv:
            self.ortho = OrthoT2d(window_size, window_size, inference=inference)
        else:
            self.ortho = OrthoT1d(window_size**2, inference=inference)

        self.act = nn.GELU()
        # self.norm1 = LayerNorm(hid_dim)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, x):
        H, W = x.shape[-2:]
        if self.window_size is not None and not self.grid:
            x, batch_list = self.winp(x)
        else:
            x = check_image_size(x, self.k)
            x = rearrange(x, 'b c (h1 h) (w1 w) -> (b h w) c h1 w1', h1=self.k, w1=self.k)
        # qk,v = torch.split(qkv, [self.dim*2, self.dim], dim=1)
        Hx, Wx = x.shape[-2:]
        # if 'grid' in self.cs:

        window_sizeh, window_sizew = Hx // self.k, Wx // self.k
        # if 'grid' in self.cs:
        #     x = rearrange(x, 'b c (h1 h) (w1 w) -> b c h w h1 w1', h1=self.k, w1=self.k)
        # else:
        #     x = rearrange(x, 'b c (h h1) (w w1) -> b c h w h1 w1', h1=self.k, w1=self.k)
        if not self.hv:
            x = rearrange(x, 'b c h1 w1 -> b c (h1 w1)')
        out = self.ortho(x, False)
        if not self.hv:
            out = rearrange(out, 'b c (h1 w1) -> b c h1 w1', h1=self.k, w1=self.k)
        # print(out.shape)
        # out = self.ortho(out, True)
        if 'grid' in self.cs:
            out = rearrange(out,
                            '(b h w) c h1 w1 -> b c (h1 h) (w1 w)', h=window_sizeh, w=window_sizew)  # , h1=self.window_size, w1=self.window_size
        else:
            out = self.winr(out, H, W, batch_list)
            # out = rearrange(outH, 'b c h w h1 w1 -> b c (h h1) (w w1)')
        # out = self.norm1(out[:, :, :H, :W])
        # out = out * v

        return self.act(out[:, :, :H, :W])

    def forward(self, x):
        # _, _, h, w = x.shape
        x = self.qkv_dwconv(self.qkv(x))
        x, v = x.chunk(2, dim=1)
        # _, _, H, W = qkv.shape
        out = self.project_out(self.get_attn(x)) * v
        # out = out
        return out
class OrthoConv(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, out_dir=None, num_layers=1,
                 hv=False):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dim = dim
        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.norm_dim = norm_dim  # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.grid_size = grid_size
        self.cs = cs
        self.hv = hv
        self.grid = True if 'grid' in cs else False
        self.k = grid_size if 'grid' in cs else window_size
        self.qkv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2, bias=bias, padding_mode=padding_mode)
        self.inference = True if 'inference' in cs else False

        self.ortho = OrthoConv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False, inference=self.inference)

        self.act = nn.GELU()
        # self.norm1 = LayerNorm(hid_dim)
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, x):
        out = self.ortho(x)
        return self.act(out)

    def forward(self, x):
        # _, _, h, w = x.shape
        x = self.qkv_dwconv(self.qkv(x))
        x, v = x.chunk(2, dim=1)
        # _, _, H, W = qkv.shape
        out = self.project_out(self.get_attn(x)) * v
        # out = out
        return out
class DCTCAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        if window_size_dct is None:
            window_size_dct = window_size
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs
        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        if self.window_size_dct and not self.window_size:
            self.winp_dct = WindowPartition(window_size_dct, shift_size=0)
            self.winr_dct = WindowReverse(window_size_dct, shift_size=0)
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = block_mlp
        if block_mlp:


            N = window_size_dct ** 2
            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    # def get_adjust(self, v):
    #     x_adjust = None
    #     b = v.shape[0]
    #     if self.temp_adj in ['max', 'mean', 'max_sub']:
    #         x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
    #     elif self.temp_adj in ['linear', 'linear_sub']:
    #         if 'sub' not in self.temp_adj:
    #             x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
    #         else:
    #             x_adjust = self.pool(v.view(b, self.num_heads, -1))
    #         x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
    #     return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        if self.window_size is not None:
            qkv, batch_list = self.winp(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix')
        #     os.makedirs(out_dir, exist_ok=True)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H//Hx, w1=W//Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     if self.num_heads > 1:
        #         attn_matrix = attn_save[:, :, 0]
        #     else:
        #         attn_matrix = attn_save[:, :]
        #     sns_plot = plt.figure()
        #
        #     sns.heatmap(attn_matrix, cmap='Reds_r', square=True, linewidths=0.) # .invert_yaxis()
        #     x_major_locator = MultipleLocator(32)
        #     # 把x轴的刻度间隔设置为，并存在变量里
        #     y_major_locator = MultipleLocator(32)
        #     # 把y轴的刻度间隔设置为，并存在变量里
        #     ax = plt.gca()
        #     # ax为两条坐标轴的实例
        #     ax.xaxis.set_major_locator(x_major_locator)
        #     # 把x轴的主刻度设置为1的倍数
        #     ax.yaxis.set_major_locator(y_major_locator)
        #     plt.grid(True, ls = '-', lw = 0.25)
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix' +'.jpg')
        #     sns_plot.savefig(out_way, dpi=700)
        #     plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     for a in range(4):
        #         # for b in range(4):
        #         dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dimx:(a+1)*dimx, a*dimx:(a+1)*dimx]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         sns_plot = plt.figure()
        #         sns.heatmap(attn_matrixi, square=True, cmap='Reds_r', linewidths=0.)
        #         # sns.heatmap(attn_matrixi, cmap='Reds_r', linewidths=0., cbar=False,
        #         #             xticklabels=False, yticklabels=False) # .invert_yaxis()
        #         # print(y)
        #         #
        #         # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #         # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #         # plt.show()
        #         # sns.heatmap(attn_matrix, cmap='Reds_r', linewidths=0.)  # .invert_yaxis()
        #         x_major_locator = MultipleLocator(32)
        #         # 把x轴的刻度间隔设置为，并存在变量里
        #         y_major_locator = MultipleLocator(32)
        #         # 把y轴的刻度间隔设置为，并存在变量里
        #         ax = plt.gca()
        #         # ax为两条坐标轴的实例
        #         ax.xaxis.set_major_locator(x_major_locator)
        #         # 把x轴的主刻度设置为1的倍数
        #         ax.yaxis.set_major_locator(y_major_locator)
        #         plt.grid(True, ls = '-', lw = 0.25)
        #         out_way = os.path.join(out_dir, 'attn_matrix-local_'+str(a) + '_'+str(b) +'.jpg')
        #         sns_plot.savefig(out_way, dpi=700)
        #         plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     JS_list = []
        #     for a in range(H // Hx):
        #         # for b in range(4):
        #         # dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dim:(a+1)*dim, a*dim:(a+1)*dim]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         attn_matrixi = attn_matrixi.flatten()
        #         JS_list.append(attn_matrixi)
        #     js_matrix = np.zeros([H // Hx, H // Hx])
        #     for i in range(len(JS_list)):
        #         for j in range(len(JS_list)):
        #             js_matrix[i, j] = JS_divergence(JS_list[i], JS_list[j])
        #     sns_plot = plt.figure()
        #     sns.heatmap(js_matrix, cmap='Reds_r', linewidths=0.01).invert_yaxis()
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix_JS' + '.jpg')
        #     sns_plot.savefig(out_way, dpi=700)

        # print(attn.shape)
        out = (attn @ v)
        if self.block_mlp:
            if (not self.window_size) or (self.window_size != self.window_size_dct):
                v = rearrange(v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                v, batch_list_ = self.winp_dct(v)
                # print(v.shape)
                v = rearrange(v, 'b c h w -> b c (h w)')
                v = self.mlp(v)
                v = rearrange(v, 'b c (h w) -> b c h w', h=self.window_size_dct, w=self.window_size_dct)
                v = self.winr_dct(v, H, W, batch_list_)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                out = out * v
            else:
                out = out * self.mlp(v)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
        else:
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                            w=Wx)
        # print(out.shape)
        if self.window_size is not None:
            out = self.winr(out, H, W, batch_list)
        return out

    def forward(self, x):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)
        qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        out = self.get_attn(qkv)

        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')
        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out
class DCTCAttention_lowfreq(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix')
        #     os.makedirs(out_dir, exist_ok=True)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H//Hx, w1=W//Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     if self.num_heads > 1:
        #         attn_matrix = attn_save[:, :, 0]
        #     else:
        #         attn_matrix = attn_save[:, :]
        #     sns_plot = plt.figure()
        #
        #     sns.heatmap(attn_matrix, cmap='Reds_r', square=True, linewidths=0.) # .invert_yaxis()
        #     x_major_locator = MultipleLocator(32)
        #     # 把x轴的刻度间隔设置为，并存在变量里
        #     y_major_locator = MultipleLocator(32)
        #     # 把y轴的刻度间隔设置为，并存在变量里
        #     ax = plt.gca()
        #     # ax为两条坐标轴的实例
        #     ax.xaxis.set_major_locator(x_major_locator)
        #     # 把x轴的主刻度设置为1的倍数
        #     ax.yaxis.set_major_locator(y_major_locator)
        #     plt.grid(True, ls = '-', lw = 0.25)
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix' +'.jpg')
        #     sns_plot.savefig(out_way, dpi=700)
        #     plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     for a in range(4):
        #         # for b in range(4):
        #         dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dimx:(a+1)*dimx, a*dimx:(a+1)*dimx]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         sns_plot = plt.figure()
        #         sns.heatmap(attn_matrixi, square=True, cmap='Reds_r', linewidths=0.)
        #         # sns.heatmap(attn_matrixi, cmap='Reds_r', linewidths=0., cbar=False,
        #         #             xticklabels=False, yticklabels=False) # .invert_yaxis()
        #         # print(y)
        #         #
        #         # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #         # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #         # plt.show()
        #         # sns.heatmap(attn_matrix, cmap='Reds_r', linewidths=0.)  # .invert_yaxis()
        #         x_major_locator = MultipleLocator(32)
        #         # 把x轴的刻度间隔设置为，并存在变量里
        #         y_major_locator = MultipleLocator(32)
        #         # 把y轴的刻度间隔设置为，并存在变量里
        #         ax = plt.gca()
        #         # ax为两条坐标轴的实例
        #         ax.xaxis.set_major_locator(x_major_locator)
        #         # 把x轴的主刻度设置为1的倍数
        #         ax.yaxis.set_major_locator(y_major_locator)
        #         plt.grid(True, ls = '-', lw = 0.25)
        #         out_way = os.path.join(out_dir, 'attn_matrix-local_'+str(a) + '_'+str(b) +'.jpg')
        #         sns_plot.savefig(out_way, dpi=700)
        #         plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     JS_list = []
        #     for a in range(H // Hx):
        #         # for b in range(4):
        #         # dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dim:(a+1)*dim, a*dim:(a+1)*dim]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         attn_matrixi = attn_matrixi.flatten()
        #         JS_list.append(attn_matrixi)
        #     js_matrix = np.zeros([H // Hx, H // Hx])
        #     for i in range(len(JS_list)):
        #         for j in range(len(JS_list)):
        #             js_matrix[i, j] = JS_divergence(JS_list[i], JS_list[j])
        #     sns_plot = plt.figure()
        #     sns.heatmap(js_matrix, cmap='Reds_r', linewidths=0.01).invert_yaxis()
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix_JS' + '.jpg')
        #     sns_plot.savefig(out_way, dpi=700)

        # print(attn.shape)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                        w=Wx)
        # print(out.shape)
        return out

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x[:, :, :h//2, :w//2]))

        _, _, H, W = qkv.shape
        # _, _, x = qkv.chunk(3, dim=1)
        out = x.clone()
        out[:, :, :H, :W] = self.get_attn(qkv )

        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        out = self.project_out(out)
        return out
class DCTCAttention_highfreq(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix')
        #     os.makedirs(out_dir, exist_ok=True)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H//Hx, w1=W//Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     if self.num_heads > 1:
        #         attn_matrix = attn_save[:, :, 0]
        #     else:
        #         attn_matrix = attn_save[:, :]
        #     sns_plot = plt.figure()
        #
        #     sns.heatmap(attn_matrix, cmap='Reds_r', square=True, linewidths=0.) # .invert_yaxis()
        #     x_major_locator = MultipleLocator(32)
        #     # 把x轴的刻度间隔设置为，并存在变量里
        #     y_major_locator = MultipleLocator(32)
        #     # 把y轴的刻度间隔设置为，并存在变量里
        #     ax = plt.gca()
        #     # ax为两条坐标轴的实例
        #     ax.xaxis.set_major_locator(x_major_locator)
        #     # 把x轴的主刻度设置为1的倍数
        #     ax.yaxis.set_major_locator(y_major_locator)
        #     plt.grid(True, ls = '-', lw = 0.25)
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix' +'.jpg')
        #     sns_plot.savefig(out_way, dpi=700)
        #     plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     for a in range(4):
        #         # for b in range(4):
        #         dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dimx:(a+1)*dimx, a*dimx:(a+1)*dimx]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         sns_plot = plt.figure()
        #         sns.heatmap(attn_matrixi, square=True, cmap='Reds_r', linewidths=0.)
        #         # sns.heatmap(attn_matrixi, cmap='Reds_r', linewidths=0., cbar=False,
        #         #             xticklabels=False, yticklabels=False) # .invert_yaxis()
        #         # print(y)
        #         #
        #         # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #         # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #         # plt.show()
        #         # sns.heatmap(attn_matrix, cmap='Reds_r', linewidths=0.)  # .invert_yaxis()
        #         x_major_locator = MultipleLocator(32)
        #         # 把x轴的刻度间隔设置为，并存在变量里
        #         y_major_locator = MultipleLocator(32)
        #         # 把y轴的刻度间隔设置为，并存在变量里
        #         ax = plt.gca()
        #         # ax为两条坐标轴的实例
        #         ax.xaxis.set_major_locator(x_major_locator)
        #         # 把x轴的主刻度设置为1的倍数
        #         ax.yaxis.set_major_locator(y_major_locator)
        #         plt.grid(True, ls = '-', lw = 0.25)
        #         out_way = os.path.join(out_dir, 'attn_matrix-local_'+str(a) + '_'+str(b) +'.jpg')
        #         sns_plot.savefig(out_way, dpi=700)
        #         plt.close()

        # if self.out_dir:
        #     out_dir = os.path.join(self.out_dir, 'attn_matrix_local')
        #     os.makedirs(out_dir, exist_ok=True)
        #     # attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head h1 w1 c1 c2', h1=H//Hx, w1=W//Wx)
        #     attn_save = rearrange(attn, '(b h1 w1) head c1 c2 -> b head (h1 c1) (w1 c2)', h1=H // Hx, w1=W // Wx)
        #     # attn_save = attn_save.cpu().detach().squeeze(0).numpy()
        #     # attn_save = kornia.tensor_to_image(attn_save, keepdim=False)
        #     # for bi in range(attn_save.shape[0]):
        #     # print(attn_save.shape)
        #     # if self.num_heads > 1:
        #     #     attn_matrix = attn_save[0, 0, ...]
        #     # else:
        #     attn_matrix = attn_save[0, 0, :, :]
        #     # print(attn_matrix.shape)
        #     dim=32
        #     JS_list = []
        #     for a in range(H // Hx):
        #         # for b in range(4):
        #         # dimx = dim * H // Hx // 4
        #         attn_matrixi = attn_matrix[a*dim:(a+1)*dim, a*dim:(a+1)*dim]
        #         # attn_matrixi = rearrange(attn_matrixi, 'h1 w1 c1 c2 -> (h1 c1) (w1 c2)')
        #         attn_matrixi = kornia.tensor_to_image(attn_matrixi, keepdim=False)
        #         attn_matrixi = attn_matrixi.flatten()
        #         JS_list.append(attn_matrixi)
        #     js_matrix = np.zeros([H // Hx, H // Hx])
        #     for i in range(len(JS_list)):
        #         for j in range(len(JS_list)):
        #             js_matrix[i, j] = JS_divergence(JS_list[i], JS_list[j])
        #     sns_plot = plt.figure()
        #     sns.heatmap(js_matrix, cmap='Reds_r', linewidths=0.01).invert_yaxis()
        #     # print(y)
        #     #
        #     # plt.xlabel('DeepRFT',fontsize=12, color='k') #x轴label的文本和字体大小
        #     # plt.ylabel('DeepRFT',fontsize=12, color='k') #y轴label的文本和字体大小
        #     # plt.show()
        #     out_way = os.path.join(out_dir, 'attn_matrix_JS' + '.jpg')
        #     sns_plot.savefig(out_way, dpi=700)

        # print(attn.shape)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                        w=Wx)
        # print(out.shape)
        return out

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x[:, :, -h//2:, -w//2:]))

        _, _, H, W = qkv.shape
        # _, _, x = qkv.chunk(3, dim=1)
        out = x.clone()
        out[:, :, -H:, -W:] = self.get_attn(qkv)

        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        out = self.project_out(out)
        return out
def JS_divergence(P,Q):
    M = (P+Q)/2
    return 0.5*scipy.stats.entropy(P, M)+0.5*scipy.stats.entropy(Q, M)
def JS_divergence_torch(P,Q):
    M = (P+Q)/2
    return 0.5*F.kl_div(P.log(), M, reduction='mean')+0.5*F.kl_div(Q.log(), M, reduction='mean')
class DCTMLP(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False, out_dir=None):
        super().__init__()

        self.out_dir = out_dir
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs
        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)

        self.block_mlp = block_mlp

        N = window_size_dct ** 2
        self.mlp = nn.Sequential(
            # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.Linear(N, N, bias=True),
            nn.GELU(),
            # nn.Linear(N, N, bias=True),
            # nn.Conv2d(dim, dim, 1, bias=True)
        )
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)


        if proj_out:
            self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    # def get_adjust(self, v):
    #     x_adjust = None
    #     b = v.shape[0]
    #     if self.temp_adj in ['max', 'mean', 'max_sub']:
    #         x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
    #     elif self.temp_adj in ['linear', 'linear_sub']:
    #         if 'sub' not in self.temp_adj:
    #             x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
    #         else:
    #             x_adjust = self.pool(v.view(b, self.num_heads, -1))
    #         x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
    #     return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        if self.window_size is not None:
            qkv, batch_list = self.winp(qkv)
        Hx, Wx = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b c h w -> b c (h w)')
        out = self.mlp(qkv)
        out = rearrange(out, 'b c (h w) -> b c h w', h=Hx,
                        w=Wx)

        # print(out.shape)
        if self.window_size is not None:
            out = self.winr(out, H, W, batch_list)
        return out

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        out = self.get_attn(qkv)

        out = self.project_out(out)
        return out
class DCTSAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False, cmlp=False):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs
        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        if self.window_size_dct and not self.window_size:
            self.winp_dct = WindowPartition(window_size_dct, shift_size=0)
            self.winr_dct = WindowReverse(window_size_dct, shift_size=0)
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = block_mlp
        self.cmlp = cmlp
        if block_mlp:
            N = window_size_dct ** 2
            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        if cmlp:
            self.chmlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    # def get_adjust(self, v):
    #     x_adjust = None
    #     b = v.shape[0]
    #     if self.temp_adj in ['max', 'mean', 'max_sub']:
    #         x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
    #     elif self.temp_adj in ['linear', 'linear_sub']:
    #         if 'sub' not in self.temp_adj:
    #             x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
    #         else:
    #             x_adjust = self.pool(v.view(b, self.num_heads, -1))
    #         x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
    #     return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        if self.window_size is not None:
            qkv, batch_list = self.winp(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        if self.cmlp:
            v_m = self.chmlp(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1))  # .contiguous())
        # print(attn.shape, out.shape)
        out = out.transpose(-2, -1)

        if self.block_mlp:
            if (not self.window_size) or (self.window_size != self.window_size_dct):
                v = rearrange(v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                v, batch_list_ = self.winp_dct(v)
                # print(v.shape)
                v = rearrange(v, 'b c h w -> b c (h w)')
                v = self.mlp(v)
                v = rearrange(v, 'b c (h w) -> b c h w', h=self.window_size_dct, w=self.window_size_dct)
                v = self.winr_dct(v, H, W, batch_list_)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                out = out * v
            else:
                out = out * self.mlp(v)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
        else:
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                            w=Wx)
        if self.cmlp:
            out = out * v_m
        # print(out.shape)
        if self.window_size is not None:
            out = self.winr(out, H, W, batch_list)
        return out

    def forward(self, x):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)
        qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        out = self.get_attn(qkv)

        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')
        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out
class DCTCAttention_mhl(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs
        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        if self.window_size_dct and not self.window_size:
            self.winp_dct = WindowPartition(window_size_dct, shift_size=0)
            self.winr_dct = WindowReverse(window_size_dct, shift_size=0)
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = block_mlp
        if block_mlp:
            N = window_size_dct ** 2

            self.bias_weight = nn.Parameter(torch.zeros(1, num_heads, 1, N), requires_grad=True)
            self.mlp_weight = nn.Parameter(torch.Tensor(num_heads, N, N), requires_grad=True)
            # x = torch.einsum('knz,bchwkn->bchwkz', self.mlp_weight, x)
            torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))
            self.gelu =nn.GELU()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()
    # def get_adjust(self, v):
    #     x_adjust = None
    #     b = v.shape[0]
    #     if self.temp_adj in ['max', 'mean', 'max_sub']:
    #         x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
    #     elif self.temp_adj in ['linear', 'linear_sub']:
    #         if 'sub' not in self.temp_adj:
    #             x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
    #         else:
    #             x_adjust = self.pool(v.view(b, self.num_heads, -1))
    #         x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
    #     return x_adjust
    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        if self.window_size is not None:
            qkv, batch_list = self.winp(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # if x_adjust is not None:
        #     if 'sub' in self.temp_adj:
        #         attn = attn - x_adjust
        #     else:
        #         attn = attn / (x_adjust + 1e-6)

        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        out = (attn @ v)
        if self.block_mlp:
            v = torch.einsum('hnz,bhcn->bhcz', self.mlp_weight, v)
            out = out * self.gelu(v + self.bias_weight)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                            w=Wx)
        else:
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                            w=Wx)
        # print(out.shape)
        if self.window_size is not None:
            out = self.winr(out, H, W, batch_list)
        return out

    def forward(self, x):
        _, _, h, w = x.shape
        # if self.dct:
        #     x = self.dct2d(x)
        qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        out = self.get_attn(qkv)

        # if self.modulate:
        #     out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
        #     out = out * torch.sigmoid(self.modulater)
        #     out = rearrange(out, 'b head c h w -> b (head c) h w')
        # if self.dct:
        #     out = self.idct2d(out)

        out = self.project_out(out)
        return out
class DCTCAttention_savefeature(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5, block_mlp=False):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm

        self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.window_size_dct = window_size_dct
        self.mask_dct = mask_dct
        self.cs = cs
        if self.window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        if self.window_size_dct and not self.window_size:
            self.winp_dct = WindowPartition(window_size_dct, shift_size=0)
            self.winr_dct = WindowReverse(window_size_dct, shift_size=0)
        # if self.temp_adj == 'mean':
        #     self.pool = nn.AdaptiveAvgPool2d([1, 1])
        # elif self.temp_adj in ['max', 'max_sub']:
        #     self.pool = nn.AdaptiveMaxPool2d([1, 1])
        # elif self.temp_adj in ['linear', 'linear_sub']:
        #     self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        self.block_mlp = block_mlp
        if block_mlp:
            N = window_size_dct ** 2
            self.mlp = nn.Sequential(
                # MLP_interk_linear([window_size, window_size], num_heads=1, bias=True, winp=False, winr=False),
                # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
                nn.Linear(N, N, bias=True),
                nn.GELU(),
                # nn.Linear(N, N, bias=True),
                # nn.Conv2d(dim, dim, 1, bias=True)
            )
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        if self.window_size is not None:
            qkv, batch_list = self.winp(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # x_adjust = self.get_adjust(v)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # if x_adjust is not None:
        #     if 'sub' in self.temp_adj:
        #         attn = attn - x_adjust
        #     else:
        #         attn = attn / (x_adjust + 1e-6)

        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        out = (attn @ v)
        if self.block_mlp:
            if (not self.window_size) or (self.window_size != self.window_size_dct):
                v = rearrange(v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                v, batch_list_ = self.winp_dct(v)
                # print(v.shape)
                v = rearrange(v, 'b c h w -> b c (h w)')
                v = self.mlp(v)
                v = rearrange(v, 'b c (h w) -> b c h w', h=self.window_size_dct, w=self.window_size_dct)
                v = self.winr_dct(v, H, W, batch_list_)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
                out = out * v
            else:
                out = out * self.mlp(v)
                out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                                w=Wx)
        else:
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                            w=Wx)
        # print(out.shape)
        if self.window_size is not None:
            out = self.winr(out, H, W, batch_list)
        return out

    def forward(self, x, out_dir=None):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)
        qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        out = self.get_attn(qkv)

        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')
        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out
class MLP_intrak_attn(nn.Module):
    def __init__(self, patch_num=8, num_heads=1, bias=False, act=None, winp=True):
        super().__init__()
        N = patch_num ** 2
        self.patch_num = patch_num
        if act == 'softmax':
            self.act = nn.Softmax(-1)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, N, num_heads, 1, 1), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.zeros(N, num_heads), requires_grad=True)

    def forward(self, x):
        mlp_weight = self.act(self.mlp_weight)
        x = torch.einsum('jh,bjhcd->bjhcd', mlp_weight, x)
        if self.bias:
            x = x + self.bias_weight
        return x
class MLP_intra(nn.Module):
    def __init__(self, patch_num=8, bias=False, act=None):
        super().__init__()
        N = patch_num ** 2
        self.patch_num = patch_num
        if act == 'softmax':
            self.act = nn.Softmax(-1)
        else:
            self.act = nn.Identity()
        self.bias = bias
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1,N,1,1,1), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.Tensor(N, N), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))

    def forward(self, x):
        mlp_weight = self.act(self.mlp_weight)
        # print(x.shape)
        x = torch.einsum('ij,bjchw->bichw', mlp_weight, x)
        if self.bias:
            x = x + self.bias_weight
        return x
class MLP_intrak(nn.Module):
    def __init__(self, patch_num=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.patch_num = patch_num if isinstance(patch_num, list) else [patch_num, patch_num]
        N = patch_num[0] * patch_num[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, N, num_heads, 1, 1, 1), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.Tensor(N, N, num_heads), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))

    def forward(self, x, H=None, W=None):
        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.patch_num)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b (h1 w1) head c h w', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
        # mlp_weight = self.mlp_weight
        # print(x.shape)
        x = torch.einsum('ijd,bjdchw->bidchw', self.mlp_weight, x)
        # print(x.shape)
        if self.bias:
            # print(self.bias_weight.shape)
            x = x + self.bias_weight
        # print(x.shape)
        if self.winr:
            x = rearrange(x, 'b (h1 w1) head c h w -> b (head c) (h1 h) (w1 w)', h1=self.patch_num[0], w1=self.patch_num[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)
class MLP_interk(nn.Module):
    def __init__(self, window_size=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
        N = window_size[0] * window_size[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, 1, 1, num_heads, 1, N), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.Tensor(N, N, num_heads), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))

    def forward(self, x, H=None, W=None):

        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.window_size)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b h1 w1 head c (h w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
        # mlp_weight = self.mlp_weight
        x = torch.einsum('ijd, bhwdcj -> bhwdcj', self.mlp_weight, x)
        if self.bias:
            x = x + self.bias_weight
        if self.winr:
            x = rearrange(x, 'b h1 w1 head c (h w) -> b (head c) (h1 h) (w1 w)', h=self.window_size[0], w=self.window_size[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)

class WDCTAttnV1(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_MaxNorm(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        mag = self.pool(mag) / (self.max_pool(mag) + 1e-6)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_noabs(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)

        mag = self.pool(x)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_L1Norm(nn.Module):
    def __init__(self, c, window_size=8, dct2d=True):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct2d = dct2d
        if dct2d:
            self.dct = DCT2x()
            self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, x):
        if self.dct2d:
            x = self.dct(x)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        mag = torch.nn.functional.normalize(self.pool(mag), dim=1)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        if self.dct2d:
            x = self.idct(x)
        return x
class WDCTAttnV1_L1Norm_sin(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        mag = torch.nn.functional.normalize(self.pool(mag), dim=1)
        x = x * torch.sin(self.sca(mag))
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_L1Norm_noabs(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        # mag = torch.abs(x)
        mag = torch.nn.functional.normalize(self.pool(x), dim=1)
        x = x * torch.sin(self.sca(mag))
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_L1Norm_sigmoid(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        mag = torch.nn.functional.normalize(self.pool(mag), dim=1)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_sin(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        x = x * torch.sin(self.sca(mag))
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV1_log(nn.Module):
    def __init__(self, c, window_size=8):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        mag = self.pool(mag)
        mag = torch.log(mag+1.)
        x = x * self.sca(mag)
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV2(nn.Module):
    def __init__(self, c, window_size=8, pool='avg'):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        if pool == 'avg':
            pool2d = nn.AdaptiveAvgPool2d(1)
        else:
            pool2d = nn.AdaptiveMaxPool2d(1)
        self.sca = nn.Sequential(
            LayerNorm2d(c),
            pool2d,
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        x = x * self.sca(torch.abs(x))
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV3(nn.Module):
    def __init__(self, c, window_size=8, pool='avg'):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        if pool == 'avg':
            pool2d = nn.AdaptiveAvgPool2d(1)
        else:
            pool2d = nn.AdaptiveMaxPool2d(1)
        self.sca = nn.Sequential(
            pool2d,
            LayerNorm2d(c),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        # print(mag.max(dim=1))
        sca_attn = self.sca(mag)
        # print(sca_attn.max(dim=1))
        x = x * sca_attn
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x
class WDCTAttnV4(nn.Module):
    def __init__(self, c, window_size=8, pool='avg'):
        super().__init__()

        self.window_size = window_size
        # Simplified Channel Attention
        if pool == 'avg':
            pool2d = nn.AdaptiveAvgPool2d(1)
        else:
            pool2d = nn.AdaptiveMaxPool2d(1)
        self.sca = nn.Sequential(
            pool2d,
            LayerNorm2d(c),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dct = DCT2x()
        self.idct = IDCT2x()
        # self.norm1 = LayerNorm2d(c)
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, inp):
        x = self.dct(inp)
        _, _, H, W = x.shape
        # print(x.shape)
        x, batch_list = self.winp(x)
        mag = torch.abs(x)
        # print(mag.max(dim=1))
        sca_attn = self.sca(mag)
        # print(sca_attn.max(dim=1))
        x = x * sca_attn
        x = self.winr(x, H, W, batch_list)
        x = self.idct(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]
        x1 = check_image_size(x1, self.window_size)
        x2 = check_image_size(x2, self.patch_num)
        x1 = self.inter(x1)
        x2 = self.intra(x2)
        # print(x1.shape, )
        x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x)
class MLP_Mixer(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]
        x1 = check_image_size(x1, self.window_size)
        x2 = check_image_size(x2, self.patch_num)
        x1 = self.inter(x1)
        x2 = self.intra(x2)
        # print(x1.shape, )
        x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x)
class MLP_MixerV2(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None),
            nn.GELU(),
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None)
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim // 2, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None),
            nn.GELU(),
            MLP_interk(dim // 2, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None)
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]
        x1 = check_image_size(x1, self.window_size)
        x2 = check_image_size(x2, self.patch_num)
        x1 = self.inter(x1)
        x2 = self.intra(x2)
        # print(x1.shape, )
        x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x)
class MLP_MixerV3(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_interk(dim, window_size=window_size, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_intrak(dim, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        # x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]

        x = check_image_size(x, self.window_size)
        x = check_image_size(x, self.patch_num)
        x = self.intra(x) * self.inter(x)

        # print(x1.shape, )
        # x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x[:, :, :H, :W])
class MLP_MixerV3x(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim*2, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_interk(dim, window_size=window_size, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=False,
                       act=None),
            nn.GELU(),
            MLP_intrak(dim, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=False, winr=True,
                       act=None)
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]

        x1 = check_image_size(x1, self.window_size)
        x2 = check_image_size(x2, self.patch_num)
        x2 = self.intra(x2)
        x1 = self.inter(x1)
        x = x1[:, :, :H, :W] * x2[:, :, :H, :W]

        # print(x1.shape, )
        # x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x)
class MLP_MixerV5(nn.Module):
    def __init__(self, dim, num_heads=1, window_size=8, patch_num=8, bias=False, proj_out=True, act=None):
        super().__init__()
        self.patch_num = patch_num
        self.window_size = window_size
        self.proj_in = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.inter = nn.Sequential(
            MLP_interk(dim, window_size=window_size, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None),
        )
        self.intra = nn.Sequential(
            MLP_intrak(dim, patch_num=patch_num, num_heads=num_heads, bias=bias, winp=True, winr=True,
                       act=None),
            nn.GELU()
        )
        if proj_out:
            self.proj_out = nn.Conv2d(dim, dim, 1, bias=bias)
        else:
            self.proj_out = nn.Identity()

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = x.chunk(2, dim=1)
        H, W = x.shape[-2:]

        x1 = check_image_size(x1, self.window_size)
        x2 = check_image_size(x2, self.patch_num)
        x2 = self.intra(x2)
        x1 = self.inter(x1)
        x = x1[:, :, :H, :W] * x2[:, :, :H, :W]
        # print(x1.shape, )
        # x = torch.cat([x1[:, :, :H, :W], x2[:, :, :H, :W]], dim=1)
        return self.proj_out(x)
class Conv_intrak(nn.Module):
    def __init__(self, dim, kernel_size=3, patch_num=8, num_heads=1, bias=False, act=None, winp=True):
        super().__init__()
        # N = patch_num ** 2
        self.patch_num = patch_num
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.num_heads = num_heads
        self.mlp_weight = nn.Conv2d(dim//num_heads, dim//num_heads, kernel_size, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.patch_num)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> (b h1 w1 head) c h w', head=self.num_heads, h1=self.patch_num, w1=self.patch_num)
        # mlp_weight = self.mlp_weight
        x = self.mlp_weight(x)

        if self.winp:
            x = rearrange(x, '(b h1 w1 head) c h w -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=self.patch_num, w1=self.patch_num)
        return self.act(x[:, :, :H, :W])
class DConv_intrak(nn.Module):
    def __init__(self, dim, kernel_size=3, patch_num=8, groups=1, bias=False, act=None, winp=True):
        super().__init__()
        # N = patch_num ** 2
        self.patch_num = patch_num
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        # self.num_heads = num_heads
        self.mlp_weight = nn.Conv2d(dim, dim, kernel_size, dilation=patch_num,
                                    padding=patch_num*(kernel_size//2), bias=bias, groups=groups)

    def forward(self, x):
        # if self.winp:
        #     H, W = x.shape[-2:]
        #     x = check_image_size(x, self.patch_num)
        #     x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> (b h1 w1 head) c h w', head=self.num_heads, h1=self.patch_num, w1=self.patch_num)
        # mlp_weight = self.mlp_weight
        # print(x.shape)
        x = self.mlp_weight(x)
        # print(x.shape)
        # if self.winp:
        #     x = rearrange(x, '(b h1 w1 head) c h w -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=self.patch_num, w1=self.patch_num)
        return self.act(x) # [:, :, :H, :W])
class DCTAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, num_k=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True, num_k_modulater_out=False, num_k_conv_out=False, num_k_modulater_attn=False,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='channel',
                 act=False, mask_dct=False, padding_mode='zeros', percent_mask=0.5):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        if cs in ['spatial_channel', 'spatial_channel_grid',
                  'channel_channel_grid', 'spatial_spatial_grid']:
            self.num_heads = num_heads // 2 if num_heads > 1 else 1
        else:
            self.num_heads = num_heads
        self.act = act
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size
        self.percent_mask = percent_mask
        self.num_k = num_k
        self.mask_dct = mask_dct
        self.cs = cs
        self.num_k_conv_out = num_k_conv_out
        self.num_k_modulater_attn = num_k_modulater_attn
        self.num_k_modulater_out = num_k_modulater_out
        if self.window_size and 'num_k' not in self.cs:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)

        # if self.act:
        #     self.relu = nn.ReLU(inplace=True)
        # else:
        #     self.relu = nn.Identity()
        # print(self.temp_adj)

        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['max_head']:
            self.conv_temp = nn.Conv2d(dim, num_heads, 1, bias=bias)
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        elif self.temp_adj == 'abs_log':
            self.pool = None
        elif self.temp_adj == 'IN':
            self.normq = nn.InstanceNorm2d(num_heads)
            self.normk = nn.InstanceNorm2d(num_heads)
        elif self.temp_adj == 'LN':
            self.normq = nn.LayerNorm(num_heads)
            self.normk = nn.LayerNorm(num_heads)
        if qk_LN:
            # self.norm = LayerNorm(dim, True)
            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
                                       groups=dim * 2, bias=bias, padding_mode=padding_mode)
            if v_proj:
                self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                          padding=1, groups=dim, bias=bias, padding_mode=padding_mode)
            else:
                self.v = nn.Identity()
                self.v_dwconv = nn.Identity()
        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))
        # N = window_size ** 2
        if num_k_modulater_out or num_k_conv_out:
            act_mlp = None # 'relu' # None # 'softmax'
            bias_m = False
            # self.k_m1 = nn.Parameter(torch.Tensor(N, N), requires_grad=True)
            if num_k_modulater_out:
                if cs in ['channel']:
                    self.k_m2 = MLP_intrak(num_k, bias=bias_m, act=act_mlp, winp=True)
                else:
                    self.k_m2 = MLP_intrak(num_k, bias=bias_m, act=act_mlp, winp=False)
            if num_k_conv_out:
                self.k_conv2 = DConv_intrak(dim, 3, patch_num=num_k, groups=1, act=act_mlp, winp=True)
                # nn.Parameter(torch.Tensor(N, N), requires_grad=True)
                # self.k_m1 = nn.Parameter(torch.zeros(N), requires_grad=True)
                # self.k_m2 = nn.Parameter(torch.zeros(N), requires_grad=True)
                # torch.nn.init.kaiming_uniform_(self.k_m1, a=math.sqrt(5))
                # torch.nn.init.kaiming_uniform_(self.k_m2, a=math.sqrt(5))
        if num_k_modulater_attn:
            # N = window_size ** 2
            bias_a = True
            self.k_m1 = MLP_intrak_attn(num_k, num_heads, bias_a) # nn.Parameter(torch.zeros(N), requires_grad=True)
        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Identity()
    def get_adjust(self, x):
        x_adjust = None
        b = x.shape[0]
        if self.temp_adj in ['max', 'mean', 'max_sub', 'max_head']:
            x_adjust = self.pool(torch.abs(x))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        elif self.temp_adj in ['linear', 'linear_sub']:
            if 'sub' not in self.temp_adj:
                x_adjust = self.pool(torch.abs(x.view(b, self.num_heads, -1)))
            else:
                x_adjust = self.pool(x.view(b, self.num_heads, -1))
            x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
        return x_adjust
    def get_attn_grid(self, x, cs='channel_grid'):
        if self.qk_LN:
            z = torch.nn.functional.normalize(x, dim=1)
            qk = self.qk_dwconv(self.qk(z))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        _, _, H, W = qkv.shape
        h, w = qkv.shape[-2:]
        qkv = check_image_size(qkv, self.num_k)
        # print(qkv.shape)
        q, k, v = qkv.chunk(3, dim=1)

        _, _, H, W = q.shape
        q = rearrange(q, 'b (head c) (k1 h) (k2 w) -> (b h w) head c (k1 k2)', head=self.num_heads, k1=self.num_k, k2=self.num_k)
        k = rearrange(k, 'b (head c) (k1 h) (k2 w) -> (b h w) head c (k1 k2)', head=self.num_heads, k1=self.num_k, k2=self.num_k)
        v = rearrange(v, 'b (head c) (k1 h) (k2 w) -> (b h w) head c (k1 k2)', head=self.num_heads, k1=self.num_k, k2=self.num_k)
        if self.temp_adj in ['max_head']:
            x_adjust = self.conv_temp(x)
            if self.window_size is not None and (
                    H > self.window_size or W > self.window_size) and 'num_k' not in self.cs:
                x_adjust, _ = self.winp(x_adjust)
            x_adjust = self.get_adjust(x_adjust)
        else:
            x_adjust = self.get_adjust(v)
        # print(x_adjust.shape)
        if self.temp_adj == 'abs_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.abs(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        elif self.temp_adj == 'relu_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.relu(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        else:
            if self.temp_adj in ['IN', 'LN']:
                q = self.normq(q)
                k = self.normq(k)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        if cs == 'channel_grid':
            attn = (q @ k.transpose(-2, -1))
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn * self.temperature - x_adjust
                else:
                    attn = attn * self.temperature / (x_adjust + 1e-6)
            else:
                attn = attn * self.temperature
            attn = attn.softmax(dim=-1)
            # print(attn.shape)
            out = (attn @ v)
            out = rearrange(out, '(b h w) head c (k1 k2) -> b (head c) (k1 h) (k2 w)', head=self.num_heads,
                            h=H // self.num_k, w=W // self.num_k, k1=self.num_k, k2=self.num_k)
        else:
            attn = (q.transpose(-2, -1) @ k)
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn * self.temperature - x_adjust
                else:
                    attn = attn * self.temperature / (x_adjust + 1e-6)
            else:
                attn = attn * self.temperature
            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            out = rearrange(out, '(b h w) head (k1 k2) c -> b (head c) (k1 h) (k2 w)', head=self.num_heads,
                            h=H // self.num_k, w=W // self.num_k, k1=self.num_k, k2=self.num_k)
        if self.num_k_modulater_out:
            out = self.k_m2(out)
        return out[:, :, :h, :w].contiguous()
    def get_attn(self, x, cs='channel'):
        if self.qk_LN:
            z = torch.nn.functional.normalize(x, dim=1)
            qk = self.qk_dwconv(self.qk(z))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
            # print(qkv.shape)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        _, _, H, W = qkv.shape
        H, W = qkv.shape[-2:]
        if self.window_size is not None and (H > self.window_size or W > self.window_size) and 'num_k' not in self.cs:
            qkv, batch_list = self.winp(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        # print(qkv.shape)
        if self.temp_adj == 'abs_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.abs(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        elif self.temp_adj == 'relu_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.relu(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        elif self.temp_adj in ['IN', 'LN']:
            q = self.normq(q)
            k = self.normq(k)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        if self.temp_adj in ['max_head']:
            x_adjust = self.conv_temp(x)
            if self.window_size is not None and (
                    H > self.window_size or W > self.window_size) and 'num_k' not in self.cs:
                x_adjust, _ = self.winp(x_adjust)
            x_adjust = self.get_adjust(x_adjust)
        else:
            x_adjust = self.get_adjust(v)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        q = torch.nn.functional.normalize(q, dim=-2)
        k = torch.nn.functional.normalize(k, dim=-2)

        if 'channel' in cs:
            attn = (q @ k.transpose(-2, -1))
            # print(x_adjust)
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn * self.temperature - x_adjust
                else:
                    attn = attn * self.temperature / (x_adjust + 1e-6)
            else:
                attn = attn * self.temperature
            if self.temp_adj == 'L1Norm':
                attn = torch.nn.functional.normalize(attn, dim=-1)
            attn = attn.softmax(dim=-1)
            # print(attn.shape)
            out = (attn @ v)
            # print(out.shape)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx, w=Wx)
        else:
            attn = (q.transpose(-2, -1) @ k)
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn * self.temperature - x_adjust
                else:
                    attn = attn * self.temperature / (x_adjust + 1e-6)
            else:
                attn = attn * self.temperature
            attn = attn.softmax(dim=-1)

            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads,
                            h=Hx, w=Wx)
        if self.window_size is not None and (H > self.window_size or W > self.window_size) and 'num_k' not in self.cs:
            out = self.winr(out, H, W, batch_list)
        if self.num_k_modulater_out:
            # print('1')
            out = self.k_m2(out)
        return out
    def get_attn_numk(self, x, cs='channel_numk'):
        if self.qk_LN:
            z = torch.nn.functional.normalize(x, dim=1)
            qk = self.qk_dwconv(self.qk(z))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        _, _, H, W = qkv.shape
        H, W = qkv.shape[-2:]

        qkv = check_image_size(qkv, self.num_k)
        qkv = rearrange(qkv, 'b c (h1 h) (w1 w) -> b (h1 w1) c h w', h1=self.num_k, w1=self.num_k)
        q, k, v = qkv.chunk(3, dim=2)

        Hx, Wx = qkv.shape[-2:]
        q = rearrange(q, 'b k (head c) h w -> b k head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b k (head c) h w -> b k head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b k (head c) h w -> b k head c (h w)', head=self.num_heads)
        if self.temp_adj in ['max_head']:
            x_adjust = self.conv_temp(x)
            x_adjust = self.get_adjust(rearrange(x_adjust, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w'))
        else:
            x_adjust = self.get_adjust(rearrange(v, 'b k head c d -> (b k) head c d'))
        # print(q.shape)
        if self.qk_norm:
            # print(self.qk_norm)
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        # print(cs)
        # if 'channel' in cs:
        # print(cs)
        attn = (q @ k.transpose(-2, -1))
        # if self.num_k_modulater:
        #     attn = torch.einsum('j,bjhcd->bjhcd', self.k_m1, attn)
            # attn = torch.einsum('ij,bjhcd->bihcd', self.k_m1, attn)# self.k_m1 @ attn
        if x_adjust is not None:
            x_adjust = rearrange(x_adjust, '(b k) head c d -> b k head c d', k=self.num_k ** 2)
            if 'sub' in self.temp_adj:
                attn = attn * self.temperature - x_adjust
            else:
                attn = attn * self.temperature / (x_adjust + 1e-6)
        else:
            attn = attn * self.temperature
        if self.num_k_modulater_attn:
            # print(attn.shape, self.k_m1.shape)
            attn = self.k_m1(attn) # torch.einsum('j,bjhcd->bjhcd', self.k_m1, attn)
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        out = (attn @ v)
        # print(out.shape)
        out = rearrange(out, 'b k head c (h w) -> b k head c h w', h=Hx,
                        w=Wx)
        # else:
        #     attn = (q.transpose(-2, -1) @ k) * self.temperature
        #     # if self.num_k_modulater:
        #     #     attn = self.k_m1 @ attn
        #     if x_adjust is not None:
        #         if 'sub' in self.temp_adj:
        #             attn = attn - x_adjust
        #         else:
        #             attn = attn / (x_adjust + 1e-6)
        #     attn = attn.softmax(dim=-1)
        #
        #     out = (attn @ v.transpose(-2, -1))  # .contiguous())
        #     # print(attn.shape, out.shape)
        #     out = rearrange(out, 'b k head (h w) c -> b k (head c) h w', head=self.num_heads,
        #                     h=Hx, w=Wx)
        if self.num_k_modulater_out:
            # out = torch.einsum('j,bjchw->bjchw', self.k_m2, out)
            # out = torch.einsum('ij,bjchw->bichw', self.k_m2, out)
            out = self.k_m2(out)
            # out = self.k_m2 @ out
        out = rearrange(out, 'b (h1 w1) head c h w -> b (head c) (h1 h) (w1 w)', h1=self.num_k, w1=self.num_k)
        out = out[:, :, :H, :W]
        return out
    def forward(self, inp):
        _, _, h, w = inp.shape
        if self.dct:
            inp = self.dct2d(inp)
        # if self.window_size_dct and self.mask_dct:
        #     x = inp[:, :, :int(self.window_size_dct*self.percent_mask), :int(self.window_size_dct*self.percent_mask)]
        # elif self.mask_dct:
        #     x = inp[:, :, :int(h*self.percent_mask), :int(w*self.percent_mask)]
        # else:
        #     x = inp
        # if self.modulate:
        #     x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.num_heads)
        #     if self.modulater.shape[-2:] != x.shape[-2:]:
        #         modulater = kornia.geometry.resize(self.modulater, x.shape[-2:])
        #     else:
        #         modulater = self.modulater
        #     x = x * torch.sigmoid(modulater)
        #     x = rearrange(x, 'b head c h w -> b (head c) h w')
        # save_root = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results'
        # save_feature(os.path.join(save_root, 'dct_feature'), x, 'log')

        x = inp
        if self.cs in ['spatial', 'channel']:
            out = self.get_attn(x, cs=self.cs)
        if self.cs in ['channel_num_k']:
            out = self.get_attn_numk(x, cs=self.cs)
        elif self.cs in ['spatial_grid', 'channel_grid']:
            out = self.get_attn_grid(x, cs=self.cs)
        # elif self.cs in ['channel_spatial']:
        #     qkv1, qkv2 = torch.chunk(qkv, 2, dim=1)
        #     out1 = self.get_attn(x, qkv1, cs='channel')
        #     out2 = self.get_attn(x, qkv2, cs='spatial')
        #     out = torch.cat([out1, out2], dim=1)

        if self.modulate:
            # print('1')
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            if self.modulater.shape[-2:] != out.shape[-2:]:
                modulater = kornia.geometry.resize(self.modulater, out.shape[-2:])
            else:
                modulater = self.modulater
            out = out * torch.sigmoid(modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')
        if self.num_k_conv_out:
            # print('2')
            out = self.k_conv2(out)
        if self.dct:
            out = self.idct2d(out)
        # if self.window_size_dct and self.mask_dct:
        #     inp[:, :, :int(self.window_size_dct*self.percent_mask), :int(self.window_size_dct*self.percent_mask)] = out
        #     out = inp
        # elif self.mask_dct:
        #     inp[:, :, :int(h*self.percent_mask), :int(w*self.percent_mask)] = out
        #     out = inp
        out = self.project_out(out)
        return out
class DCTAttention_temp_HV(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1, cs='dct_H_inter'):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        if cs == 'HV_inter':
            self.num_heads = num_heads // 2 if num_heads > 1 else 1
        else:
            self.num_heads = num_heads
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        self.window_size = window_size

        # print(self.temp_adj)
        self.cs = cs
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        elif self.temp_adj == 'abs_log':
            self.pool = None
        elif self.temp_adj == 'IN':
            self.normq = nn.InstanceNorm2d(num_heads)
            self.normk = nn.InstanceNorm2d(num_heads)
        elif self.temp_adj == 'LN':
            self.normq = nn.LayerNorm(num_heads)
            self.normk = nn.LayerNorm(num_heads)
        if qk_LN:
            self.norm = LayerNorm(dim, True)
            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
            if v_proj:
                self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            else:
                self.v = nn.Identity()
                self.v_dwconv = nn.Identity()
        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 3, bias=bias)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            if 'dct2d' in self.cs:
                self.dct1d = DCT2x()
                self.idct1d = IDCT2x()
            else:
                self.dct1d = DCT1x(dim=-1)
                self.idct1d = IDCT1x(dim=-1)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))

        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Identity()
    def get_attn(self, qkv, cs='spatial'):
        q, k, v = qkv.chunk(3, dim=1)
        b, _, Wx = qkv.shape
        q = rearrange(q, 'b (head c) w -> b head c w', head=self.num_heads)
        k = rearrange(k, 'b (head c) w -> b head c w', head=self.num_heads)
        v = rearrange(v, 'b (head c) w -> b head c w', head=self.num_heads)
        x_adjust = None
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        elif self.temp_adj in ['linear', 'linear_sub']:
            if 'sub' not in self.temp_adj:
                x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
            else:
                x_adjust = self.pool(v.view(b, self.num_heads, -1))
            x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
            # print(x_adjust.shape)
        elif self.temp_adj == 'abs_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.abs(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        elif self.temp_adj == 'relu_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.relu(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        else:
            if self.temp_adj in ['IN', 'LN']:
                q = self.normq(q)
                k = self.normq(k)
            x_adjust = None
        # print(x_adjust.shape)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if cs == 'channel':
            attn = (q @ k.transpose(-2, -1)) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / (x_adjust + 1e-6)

            attn = attn.softmax(dim=-1)
            # print(attn.shape)
            out = (attn @ v)
            # print(out.shape)
            out = rearrange(out, 'b head c w -> b (head c) w', head=self.num_heads,
                            w=Wx)
        else:
            attn = (q.transpose(-2, -1) @ k) * self.temperature
            if x_adjust is not None:
                if 'sub' in self.temp_adj:
                    attn = attn - x_adjust
                else:
                    attn = attn / (x_adjust + 1e-6)

            attn = attn.softmax(dim=-1)

            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = rearrange(out, 'b head w c -> b (head c) w', head=self.num_heads, w=Wx)
        return out
    def forward(self, x):
        _, _, h, w = x.shape
        if self.cs == 'dct_V_inter':
            x = x.transpose(-2, -1)
        if self.dct:
            x = self.dct1d(x)

        if self.qk_LN:
            qk = self.qk_dwconv(self.qk(self.norm(x)))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))
        _, _, H, W = qkv.shape
        # qkv = rearrange(qkv, 'b c h w -> (b w) c h')
        qkv = rearrange(qkv, 'b c h w -> (b h) c w')
        # if self.cs in ['dct_H_inter', 'dct_V_inter']:
        out = self.get_attn(qkv, cs='spatial')
        # else:
        #     raise ValueError
        # out = rearrange(out, '(b w) c h -> b c h w', w=W)
        out = rearrange(out, '(b h) c w -> b c h w', h=H)
        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        if self.dct:
            out = self.idct1d(out)
        if self.cs == 'dct_V_inter':
            out = out.transpose(-2, -1)
        out = self.project_out(out)
        return out
class ChannelAttention_temp(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=True,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False, norm_dim=-1,
                 padding_mode='zeros'):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = norm_dim # -2
        # print(self.temp_adj)
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj in ['max', 'max_sub']:
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj in ['linear', 'linear_sub']:
            self.pool = nn.Linear((dim//num_heads) * window_size**2, 1)
        elif self.temp_adj == 'abs_log':
            self.pool = None
        elif self.temp_adj == 'IN':
            self.normq = nn.InstanceNorm2d(num_heads)
            self.normk = nn.InstanceNorm2d(num_heads)
        elif self.temp_adj == 'LN':
            self.normq = nn.LayerNorm(num_heads)
            self.normk = nn.LayerNorm(num_heads)
        if qk_LN:
            self.norm = LayerNorm(dim, True)
            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
            if v_proj:
                self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            else:
                self.v = nn.Identity()
                self.v_dwconv = nn.Identity()
        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x()
            self.idct2d = IDCT2x()
        self.num_heads = num_heads
        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))
        self.window_size = window_size
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)

        if self.qk_LN:
            qk = self.qk_dwconv(self.qk(self.norm(x)))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))

        normalize = self.qk_norm
        _, _, H, W = qkv.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        q, k, v = qkv.chunk(3, dim=1)
        b, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x_adjust = None
        if self.temp_adj in ['max', 'mean', 'max_sub']:
            x_adjust = self.pool(torch.abs(v))  # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
        elif self.temp_adj in ['linear', 'linear_sub']:
            if 'sub' not in self.temp_adj:
                x_adjust = self.pool(torch.abs(v.view(b, self.num_heads, -1)))
            else:
                x_adjust = self.pool(v.view(b, self.num_heads, -1))
            x_adjust = x_adjust.view(b, self.num_heads, 1, 1)
            # print(x_adjust.shape)
        elif self.temp_adj == 'abs_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.abs(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        elif self.temp_adj == 'relu_log':
            qk = torch.cat([q, k], dim=1)
            qk = torch.log(torch.relu(qk) + 1.)
            q, k = torch.chunk(qk, 2, dim=1)
        else:
            if self.temp_adj in ['IN', 'LN']:
                q = self.normq(q)
                k = self.normq(k)
            x_adjust = None
        # print(x_adjust.shape)
        if normalize:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        if x_adjust is not None:
            if 'sub' in self.temp_adj:
                attn = (q @ k.transpose(-2, -1)) * self.temperature - x_adjust
            else:
                attn = (q @ k.transpose(-2, -1)) * self.temperature / (x_adjust + 1e-6)
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        # print(attn.shape)

        out = (attn @ v)
        # print(out.shape)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=Hx,
                         w=Wx)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out
class SpatialAttention_temp(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=False,
                 qk_norm=True, qk_LN=False, v_proj=True, proj_out=True,
                 mean_cut=False, temp_adj=None, temp_div=False):
        super().__init__()
        self.temp_adj = temp_adj
        self.qk_norm = qk_norm
        self.dct = dct
        self.modulate = modulater
        self.qk_LN = qk_LN
        self.mean_cut = mean_cut
        self.norm_dim = -1 # -2
        # print(self.temp_adj)
        if self.temp_adj == 'mean':
            self.pool = nn.AdaptiveAvgPool2d([1, 1])
        elif self.temp_adj == 'max':
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj == 'conv':
            self.pool = nn.AdaptiveMaxPool2d([1, 1])
        elif self.temp_adj == 'IN':
            self.normq = nn.InstanceNorm2d(num_heads)
            self.normk = nn.InstanceNorm2d(num_heads)
        elif self.temp_adj == 'LN':
            self.normq = nn.LayerNorm(num_heads)
            self.normk = nn.LayerNorm(num_heads)
        if qk_LN:
            self.norm = LayerNorm(dim, True)
            self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
            if v_proj:
                self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
                self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            else:
                self.v = nn.Identity()
                self.v_dwconv = nn.Identity()
        else:
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                        stride=1, padding=1, groups=dim * 3, bias=bias)
        if modulater:
            self.modulater = nn.Parameter(torch.zeros(1, num_heads, 1, window_size_dct, window_size_dct))

        if self.dct:
            self.dct2d = DCT2x(64, 64)
            self.idct2d = IDCT2x(64, 64)
        self.num_heads = num_heads
        if temp_div:
            self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) / math.sqrt(dim))
        self.window_size = window_size
        if proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        if self.dct:
            x = self.dct2d(x)

        if self.qk_LN:
            qk = self.qk_dwconv(self.qk(self.norm(x)))
            v = self.v_dwconv(self.v(x))
            qkv = torch.cat([qk, v], dim=1)
        else:
            qkv = self.qkv_dwconv(self.qkv(x))

        _, _, H, W = qkv.shape
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        q, k, v = qkv.chunk(3, dim=1)
        _, _, Hx, Wx = qkv.shape
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        if self.temp_adj in ['max', 'mean']:
            x_adjust = self.pool(torch.abs(v)) + 1e-6 # torch.cat([q, k], dim=2) * self.pool(torch.abs(k))
            temperature = self.temperature / x_adjust

        if self.norm_dim:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q.transpose(-2, -1) @ k) * temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1))  # .contiguous())
        # print(attn.shape, out.shape)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads,
                        h=Hx, w=Wx)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        if self.modulate:
            out = rearrange(out, 'b (head c) h w -> b head c h w', head=self.num_heads)
            out = out * torch.sigmoid(self.modulater)
            out = rearrange(out, 'b head c h w -> b (head c) h w')

        if self.dct:
            out = self.idct2d(out)

        out = self.project_out(out)
        return out
class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy', dct=False, qk_norm=True, norm_dim=-1, proj_out=True):
        super().__init__()
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dct = dct
        self.norm_dim = norm_dim
        self.window_size_dct = window_size_dct
        self.norm_dct = 'ortho'
        self.proj_out = proj_out
        if self.dct:
            self.dct2d = DCT2x(self.window_size_dct, self.window_size_dct)
            self.idct2d = IDCT2x(self.window_size_dct, self.window_size_dct)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        if self.proj_out:
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.window_size = window_size
        # self.k = 8
        self.sca = sca
        if sca:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        # print(self.k)
        # self.window_size = None

    def forward(self, x):
        b, c, H, W = x.shape
        # x = check_image_size(x, self.k, mode='reflect')
        # _, _, H, W = x.shape
        # x = self.D_L(x)
        #
        if self.dct:
            x = self.dct2d(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        # if self.dct:
        #     qkv = rearrange(qkv, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        #     # qkvx = dct_2d_torch(qkv, self.norm_dct)
        #     qkv = self.dct2d(qkv)
        #     # print(torch.mean(qkv - qkvx))
        #     qkv = rearrange(qkv, 'b c k1 k2 h w -> b (c h w) k1 k2')
        if self.shift_method == 'roll' and self.shift_size != 0:
            qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            if self.shift_method == 'xy' and self.shift_size != 0:
                qkv, batch_list = window_partitionxy(qkv, self.window_size, start=[self.shift_size, self.shift_size])
            else:
                # if self.shift_size == 0:
                qkv, batch_list = window_partitionx(qkv, self.window_size)

        out = get_attn_spatial(qkv, self.temperature1, num_heads=self.num_heads,
                                   normalize=self.qk_norm, norm_dim=self.norm_dim)

        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            if self.shift_method == 'xy' and self.shift_size != 0:
                out = window_reversexy(out, self.window_size, H, W, batch_list, start=[self.shift_size, self.shift_size])
            else:  #  self.window_size is not None and (H != self.window_size or W != self.window_size):
                # if self.shift_size == 0:
                out = window_reversex(out, self.window_size, H, W, batch_list)

        if self.shift_method == 'roll' and self.shift_size != 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        if self.dct:
            # out = rearrange(out, 'b (c h w) k1 k2 -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
            # outx = idct_2d_torch(out, self.norm_dct)
            out = self.idct2d(out)
            # print(torch.mean(out-outx))
            # out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        # out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        # out = self.U_L(out)
        if self.proj_out:
            out = self.project_out(out)
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out
class CrossSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy', dct=False, qk_norm=True):
        super().__init__()
        # self.norm1 = LayerNorm(dim)
        # self.norm2 = LayerNorm(dim)
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dct = dct
        self.window_size_dct = window_size_dct
        norm_dct = True
        if self.dct:
            self.dct2d = DCT2(self.window_size_dct, norm_dct)
            self.idct2d = IDCT2(self.window_size_dct, norm_dct)
        # self.qkv1 = nn.Sequential(nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
        #                           nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        #                           )
        # self.qkv2 = nn.Sequential(nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
        #                           nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
        #                                     bias=bias)
        #                           )
        self.qk = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, kernel_size=1, bias=bias),
                                nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4,
                                            bias=bias)
                                )
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.window_size = window_size
        # self.k = 8
        self.v_weight = nn.Parameter(torch.ones((2, dim, 1, 1)), requires_grad=True)
        # self.v_2 = nn.Parameter(torch.randn((1, dim, 1, 1)), requires_grad=True)
        # print(self.k)
        # self.window_size = None

    def forward(self, x1, x2):
        b, c, H, W = x1.shape
        # x1_ = self.norm1(x1)
        # x2_ = self.norm2(x2)
        # x = check_image_size(x, self.k, mode='reflect')
        # _, _, H, W = x.shape
        # x = self.D_L(x)
        # qkv1 = self.qkv1(x1_)
        # qkv2 = self.qkv2(x2_)
        qk = self.qk(torch.cat([x1, x2], dim=1))
        q1, k1, q2, k2 = torch.chunk(qk, 4, 1)
        # q1, k1, v1 = torch.chunk(qkv1, 3, 1)
        # q2, k2, v2 = torch.chunk(qkv2, 3, 1)
        # qkv1 = torch.cat([q1, k2, v1], dim=1)
        # qkv2 = torch.cat([q2, k1, v2], dim=1)
        qkv1 = torch.cat([q1, k2, x1], dim=1)
        qkv2 = torch.cat([q2, k2, x2], dim=1)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv1, batch_list = window_partitionx(qkv1, self.window_size)
            qkv2, _ = window_partitionx(qkv2, self.window_size)
        # qkv = rearrange(qkv, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        normalize = self.qk_norm
        out1 = get_attn_spatial(qkv1, self.temperature1, num_heads=self.num_heads,
                                   normalize=normalize, norm_dim=-2)
        out2 = get_attn_spatial(qkv2, self.temperature2, num_heads=self.num_heads,
                                normalize=normalize, norm_dim=-2)
        v_weight = torch.softmax(self.v_weight, dim=0)
        v_weight1, v_weight2 = torch.chunk(v_weight, 2, 0)
        # print(v_weight1.shape)
        out = v_weight1 * out1 + v_weight2 * out2
        # out = self.v_1 * out1 + self.v_2 * out2
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        # out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        # out = self.U_L(out)
        # out = self.project_out(out) #  + x2) #
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out # + x1 #
class CrossSpatialAttention2(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy', dct=False, qk_norm=True):
        super().__init__()
        # self.norm1 = LayerNorm(dim)
        # self.norm2 = LayerNorm(dim)
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dct = dct
        self.window_size_dct = window_size_dct
        norm_dct = True
        if self.dct:
            self.dct2d = DCT2(self.window_size_dct, norm_dct)
            self.idct2d = IDCT2(self.window_size_dct, norm_dct)
        # self.qkv1 = nn.Sequential(nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
        #                           nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        #                           )
        # self.qkv2 = nn.Sequential(nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias),
        #                           nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
        #                                     bias=bias)
        #                           )
        self.qk = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, kernel_size=1, bias=bias),
                                nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4,
                                            bias=bias)
                                )
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.window_size = window_size
        # self.k = 8
        self.v_weight = nn.Parameter(torch.ones((2, dim, 1, 1)), requires_grad=True)
        # self.v_2 = nn.Parameter(torch.randn((1, dim, 1, 1)), requires_grad=True)
        # print(self.k)
        # self.window_size = None

    def forward(self, x):
        b, c, H, W = x.shape
        x1, x2 = torch.chunk(x, 2, dim=1)
        # x1_ = self.norm1(x1)
        # x2_ = self.norm2(x2)
        # x = check_image_size(x, self.k, mode='reflect')
        # _, _, H, W = x.shape
        # x = self.D_L(x)
        # qkv1 = self.qkv1(x1_)
        # qkv2 = self.qkv2(x2_)
        qk = self.qk(x)
        q1, k1, q2, k2 = torch.chunk(qk, 4, 1)
        # q1, k1, v1 = torch.chunk(qkv1, 3, 1)
        # q2, k2, v2 = torch.chunk(qkv2, 3, 1)
        # qkv1 = torch.cat([q1, k2, v1], dim=1)
        # qkv2 = torch.cat([q2, k1, v2], dim=1)
        qkv1 = torch.cat([q1, k2, x1], dim=1)
        qkv2 = torch.cat([q2, k2, x2], dim=1)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv1, batch_list = window_partitionx(qkv1, self.window_size)
            qkv2, _ = window_partitionx(qkv2, self.window_size)
        # qkv = rearrange(qkv, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        normalize = self.qk_norm
        out1 = get_attn_spatial(qkv1, self.temperature1, num_heads=self.num_heads,
                                   normalize=normalize, norm_dim=-2)
        out2 = get_attn_spatial(qkv2, self.temperature2, num_heads=self.num_heads,
                                normalize=normalize, norm_dim=-2)
        v_weight = torch.softmax(self.v_weight, dim=0)
        v_weight1, v_weight2 = torch.chunk(v_weight, 2, 0)
        # print(v_weight1.shape)
        out = v_weight1 * out1 + v_weight2 * out2
        # out = self.v_1 * out1 + self.v_2 * out2
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        # out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        # out = self.U_L(out)
        # out = self.project_out(out) #  + x2) #
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out # + x1 #
class DCTSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy'):
        super().__init__()

        self.dct2d = DCT2(self.window_size_dct)
        self.idct2d = IDCT2(self.window_size_dct)
        self.attn = SpatialAttention(dim, num_heads, bias, window_size, window_size_dct, shift_size, sca,
                 shift_method, dct=False, qk_norm=False)

    def forward(self, x):
        b, c, H, W = x.shape
        x = check_image_size(x, self.window_size, mode='reflect')

        x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        x = self.dct2d(x)
        x = rearrange(x, 'b c k1 k2 h w -> (b h w) c k1 k2')
        out = self.attn(x)
        out = rearrange(out, '(b h w) c k1 k2 -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        out = self.idct2d(out)
        out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out[:, :, :H, :W].contiguous()
class DCTChannelSpatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy'):
        super().__init__()
        self.sca = sca
        self.window_size_dct = window_size_dct
        self.window_size = window_size
        self.dct2d = DCT2(self.window_size_dct)
        self.idct2d = IDCT2(self.window_size_dct)
        self.sattn = SpatialAttention(dim, num_heads, bias, window_size, window_size_dct, shift_size, sca,
                 shift_method,dct=False)
        self.cattn = SpatialAttention(dim, num_heads, bias, window_size_dct, window_size_dct, shift_size, sca,
                 shift_method,dct=False)
    def forward(self, x):
        b, c, H, W = x.shape
        x = check_image_size(x, self.window_size_dct, mode='reflect')

        x = rearrange(x, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        x = self.dct2d(x)
        x = rearrange(x, 'b c k1 k2 h w -> b c (h k1) (w k2)')
        # print(x.shape)
        out = self.sattn(x)+x
        out = rearrange(out, 'b c (h k1) (w k2) -> b c (k1 h) (k2 w)', h=self.window_size_dct, w=self.window_size_dct)
        out = self.cattn(out)
        out = rearrange(out, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        out = self.idct2d(out)
        out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out[:, :, :H, :W].contiguous()
class SpatialChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, shift_size=0, sca=False,
                 shift_method='xy', inference=False):
        super().__init__()
        self.inference = inference
        self.num_heads = num_heads // 2 if num_heads > 1 else 1
        self.shift_size = shift_size
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        # self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # self.D_L = DownLift(dim)
        # self.U_L = UpLift(dim)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.window_size = 8
        self.k = 8
        self.sca = sca
        if sca:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        # if self.inference:
        #     self.window_gen = Window_Local(overlap_size=0, window_size=(self.window_size, self.window_size), qkv=True)
        # print(self.k)
        # self.window_size = None

    def forward(self, x):
        # b, c, H, W = x.shape
        # x = check_image_size(x, self.k, mode='reflect')
        _, _, H, W = x.shape
        # x = self.D_L(x)
        # print(x.shape)
        qkv = self.qkv_dwconv(self.qkv(x))
        # if self.inference:
        #     qkv = self.window_gen.grids(qkv)
        #     cnt, batchsize = qkv.shape[:2]
        #     qkv = rearrange(qkv, 'm b c h w -> (m b) c h w')
        # else:
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            qkv, batch_list = window_partitionx(qkv, self.window_size)
        qkv1, qkv2 = torch.chunk(qkv, 2, dim=1)
        # qkv = rearrange(qkv, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
        out1 = get_attn_channel(qkv1, self.temperature1, num_heads=self.num_heads,
                                   normalize=True, norm_dim=-2)
        out2 = get_attn_spatial(qkv2, self.temperature2, num_heads=self.num_heads,
                               normalize=True, norm_dim=-2)
        out = torch.cat([out1, out2], dim=1)
        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        # print(out.shape)
        # if self.inference:
        #     out = rearrange(out, '(m b) c h w -> m b c h w', m=cnt, b=batchsize)
        #     out = self.window_gen.grids_inverse(out)
        # else:
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)
        # out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
        # out = self.U_L(out)
        out = self.project_out(out) #
        # out = self.project_out(out[:, :, :H, :W])  # [:, :, :H, :W]
        return out
class SpatialGridAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', inference=False, dct=False, qk_norm=True):
        super().__init__()
        self.inference = inference
        self.num_heads = num_heads # // 2 if num_heads > 1 else 1
        self.dct = dct
        norm_dct = True
        self.qk_norm = qk_norm
        if dct:
            self.dct2d = DCT2(window_size_dct, norm_dct)
            self.idct2d = IDCT2(window_size_dct, norm_dct)
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(self.num_heads, 1, 1, 1))
        self.window_size_dct = window_size_dct
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = window_size
        self.sca = sca
        if sca:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        # if self.inference:
        #     self.window_gen = Window_Local(overlap_size=0, window_size=(self.window_size, self.window_size), qkv=True)
        # print(self.k)
        # self.window_size = None

    def forward(self, x):
        # b, c, H, W = x.shape
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # qkv, batch_list = window_partitionx(qkv, self.window_size_dct)
        if self.dct:
            # if self.inference:

            # b, c, H, W = x.shape
            qkv = check_image_size(qkv, self.window_size_dct, mode='reflect')

            qkv = rearrange(qkv, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
            qkv = self.dct2d(qkv)
            qkv = rearrange(qkv, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
            normalize = False
        else:
            normalize = self.qk_norm
        out = get_attn_spatial_grid(qkv, self.temperature1, num_heads=self.num_heads, num_k=self.k,
                               normalize=normalize, norm_dim=-2)
        if self.dct:
            out = rearrange(out, 'b c (k1 h) (k2 w) -> b c k1 k2 h w', h=self.window_size_dct, w=self.window_size_dct)
            out = self.idct2d(out)
            out = rearrange(out, 'b c k1 k2 h w -> b c (k1 h) (k2 w)')
            # print(out.shape)
            # if self.inference:
        # out = window_reversex(out, self.window_size_dct, h, w, batch_list)
        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        # print(out.shape)

        # out = self.U_L(out)
        # out = self.project_out(out) #
        out = self.project_out(out[:, :, :h, :w].contiguous())  #
        return out
class ChannelGridAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, window_size_dct=8, sca=False,
                 shift_method='xy', modulater=False, dct=False):
        super().__init__()
        self.modulate = modulater
        if modulater:
            self.modulater = nn.Parameter(torch.ones(self.num_heads, 1, window_size_dct, window_size_dct))
        self.num_heads = num_heads # // 2 if num_heads > 1 else 1
        self.dct = dct

        if dct:
            self.dct2d = DCT2x(window_size_dct, window_size_dct)
            self.idct2d = IDCT2x(window_size_dct, window_size_dct)
        self.shift_method = shift_method
        self.temperature1 = nn.Parameter(torch.ones(self.num_heads, 1, 1, 1))
        self.window_size_dct = window_size_dct
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = window_size
        self.sca = sca
        if sca:
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        # if self.inference:
        #     self.window_gen = Window_Local(overlap_size=0, window_size=(self.window_size, self.window_size), qkv=True)
        # print(self.k)
        # self.window_size = None

    def forward(self, x):
        # b, c, H, W = x.shape
        _, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        if self.dct:
            if self.inference:
                qkv, batch_list = window_partitionx(qkv, self.window_size_dct)
            qkv = self.dct2d(qkv)
        out = get_attn_channel_grid(qkv, self.temperature1, num_heads=self.num_heads, num_k=self.k,
                               normalize=True, norm_dim=-1)
        if self.dct:
            out = self.idct2d(out)
            # print(out.shape)
            if self.inference:
                out = window_reversex(out, self.window_size_dct, h, w, batch_list)
        if self.sca:
            out = out.contiguous()
            out = self.sca(out) * out
        out = self.project_out(out)  # [:, :, :H, :W]
        return out

########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        print(q.shape)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

class Attentiom_UFormer(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 modulator=True, cross_modulator=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        # if min(self.input_resolution) <= self.win_size:
        #     self.shift_size = 0
        #     self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(self.win_size * self.win_size, dim)  # modulator
        else:
            self.modulator = None

        # if cross_modulator:
        #     self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
        #     self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
        #                                 proj_drop=drop,
        #                                 token_projection=token_projection, )
        #     self.norm_cross = norm_layer(dim)
        # else:
        self.cross_modulator = None

        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


    def with_pos_embed(self, tensor, pos):
        # print(tensor.shape, pos.shape)
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # x = rearrange(x, 'b h w c -> b (h w) c')
        # print(x.shape, self.win_size)
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)) #
            input_mask_windows, _ = window_partitionx(input_mask, self.win_size)  # nW, win_size, win_size, 1
            input_mask_windows = input_mask_windows.permute(0, 2, 3, 1)
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, 1, H, W)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, :, h, w] = cnt
                    cnt += 1
            shift_mask_windows, _ = window_partitionx(shift_mask, self.win_size)  # nW, 1, win_size, win_size
            shift_mask_windows = shift_mask_windows.permute(0, 2, 3, 1)
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        # if self.cross_modulator is not None:
        #     shortcut = x
        #     x_cross = self.norm_cross(x)
        #     x_cross = self.cross_attn(x_cross, self.cross_modulator.weight)
        #     x = shortcut + x_cross

        # shortcut = x
        # print(x.shape)
        # x = self.norm1(x)
        # x = x.view(B, H, W, C)
        # x = rearrange(x, 'b c h w -> b h w c')
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        else:
            shifted_x = x

        # partition windows
        x_windows, batch_list = window_partitionx(shifted_x, self.win_size)  # nW*B, C, win_size, win_size  N*C->C
        x_windows = rearrange(x_windows, 'b c h w -> b h w c')  # x_windows.permute(0, 2, 3, 1)
        # print(x_windows.shape)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        # attn_windows = attn_windows.permute(0, 3, 1, 2)
        attn_windows = rearrange(attn_windows, 'b h w c -> b c h w')
        shifted_x = window_reversex(attn_windows, self.win_size, H, W, batch_list)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)
        # x = rearrange(x, 'b h w c -> b c h w')
        del attn_mask
        return x # .view(B, H, W, C)
class Attention_DCT_branch(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, pad_size=0, stride=1):
        super(Attention_DCT_branch, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        n = window_size ** 2
        self.dct1d = DCT1d(window_size=n)
        self.idct1d = IDCT1d(window_size=n)
        self.pad_size = pad_size
        self.stride = stride
        self.mode = 'reflect'
        if self.mode != 'reflect':
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))

        self.act = nn.PReLU()
        self.conv1 = nn.Conv2d(n, n, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1, groups=n, bias=bias)
        self.conv3 = nn.Conv2d(n, n, kernel_size=1, bias=bias)

    def forward(self, x):

        b, c, H, W = x.shape
        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        if x.shape[-2:] != self.divisor.shape[-2:]:
            self.input_ones = torch.ones_like(x)
            self.fold = nn.Fold(output_size=x.shape[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)
        x = self.dct1d(x)
        if self.mode != 'reflect':
            h, w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                        W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            h, w = (H + self.pad_size - self.window_size) // self.stride + 1, (
                    W + self.pad_size - self.window_size) // self.stride + 1
        x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        short_cut = x
        x = self.conv2(self.conv1(x))
        x = self.act(x)
        x = self.conv3(x) + short_cut
        x = rearrange(x, 'b c h w -> b c (h w)', h=h, w=w)
        x = self.idct1d(x)

        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size:, self.pad_size:]
        x = rearrange(x, '(b c) k h w -> b (c k) h w', b=b, c=c, k=1)
        # out = self.project_out(x)
        return x
class AttentionHV(nn.Module):
    def __init__(self, head_num):
        super(AttentionHV, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out

class IntraHV_SA(nn.Module):
    def __init__(self, dim, head_num, window_size=None, shift_size=0, dct='', norm_type='LayerNorm', modulater=False):
        super(IntraHV_SA, self).__init__()
        self.dct = dct # True if 'dct' in dct else False
        self.norm_type = norm_type
        self.window_size = window_size
        self.shift_size = shift_size
        if dct == 'dct2d':
            self.dctx = DCT2x()
            self.idctx = IDCT2x()
        elif dct == 'dct1d_w':
            self.dctx = DCT1x(-1)
            self.idctx = IDCT1x(-1)
        elif dct == 'dct1d_h':
            self.dctx = DCT1x(-2)
            self.idctx = IDCT1x(-2)
        elif dct == 'dct1d_hv':
            self.dctxh = DCT1x(-2)
            self.idctxh = IDCT1x(-2)
            self.dctxv = DCT1x(-1)
            self.idctxv = IDCT1x(-1)
        elif dct == 'dct1d_vh':
            self.dctxh = DCT1x(-1)
            self.idctxh = IDCT1x(-1)
            self.dctxv = DCT1x(-2)
            self.idctxv = IDCT1x(-2)
        self.hidden_size = dim // 2
        if norm_type == 'LayerNorm':
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = LayerNorm2d(dim // 2)
                self.attention_norm_v = LayerNorm2d(dim // 2)
            else:
                self.attention_norm = LayerNorm2d(dim)
        elif norm_type == 'InstanceNorm':
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = nn.InstanceNorm2d(dim // 2)
                self.attention_norm_v = nn.InstanceNorm2d(dim // 2)
            else:
                self.attention_norm = nn.InstanceNorm2d(dim)
        else:
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = nn.Identity()
                self.attention_norm_v = nn.Identity()
            else:
                self.attention_norm = nn.Identity()

        self.head_num = head_num
        if dct == 'dct1d_hv' or dct == 'dct1d_vh':
            self.conv_input_h = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, padding=0)
            self.conv_input_v = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, padding=0)
        else:
            self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.attn = AttentionHV(head_num=self.head_num)

        self.modulater = modulater
        if modulater and (dct in ['dct2d', 'dct1d_w', 'dct1d_h']):
            self.dct_mask = nn.Parameter(torch.FloatTensor(1, 1, window_size, window_size), requires_grad=True)
    def forward(self, x):
        # short_cut = x
        # print(x.shape)
        if self.shift_size > 0:
            if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
                feature_h, feature_v = torch.chunk(x, 2, dim=1)
                feature_h = torch.roll(feature_h, shifts=(-self.shift_size), dims=(-1))
                feature_v = torch.roll(feature_v, shifts=(-self.shift_size), dims=(-2))
                x = torch.cat([feature_h, feature_v], dim=1)
            else:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        B, C, H, W = x.size()
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        b, c, h, w = x.shape
        if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
            feature_h, feature_v = torch.chunk(x, 2, dim=1)
            feature_h = self.dctxh(feature_h)
            feature_v = self.dctxv(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            feature_h = self.attention_norm_h(feature_h)
            feature_v = self.attention_norm_v(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            feature_h = self.conv_input_h(feature_h)
            feature_v = self.conv_input_v(feature_v)
        else:
            if self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
                x = self.dctx(x)
                if self.modulater:
                    x = x * torch.sigmoid(self.dct_mask)
            x = self.attention_norm(x)
            x = self.conv_input(x)
            feature_h, feature_v = torch.chunk(x, 2, dim=1)
        # print(x_input[0])

        feature_h = feature_h.permute(0, 2, 3, 1).contiguous()
        feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
        feature_h = feature_h.view(b * h, w, c//2)

        feature_v = feature_v.view(b * w, h, c//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if h == w:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
        attention_output_h = attention_output_h.view(b, h, w, c//2).permute(0, 3, 1, 2).contiguous()
        attention_output_v = attention_output_v.view(b, w, h, c//2).permute(0, 3, 2, 1).contiguous()
        if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
            attention_output_h = self.idctxh(attention_output_h)
            attention_output_v = self.idctxv(attention_output_v)
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
            # attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        elif self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
            attn_out = self.idctx(attn_out)
            # attn_out = self.fuse_out(attn_out)
        else:
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            attn_out = window_reversex(attn_out, self.window_size, H, W, batch_list)
        if self.shift_size > 0:
            if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
                attention_output_h, attention_output_v = torch.chunk(attn_out, 2, dim=1)
                attention_output_h = torch.roll(attention_output_h, shifts=(-self.shift_size), dims=(-1))
                attention_output_v = torch.roll(attention_output_v, shifts=(-self.shift_size), dims=(-2))
                attn_out = torch.cat([attention_output_h, attention_output_v], dim=1)
            else:
                attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        attn_out = self.fuse_out(attn_out)
        # x = attn_out + short_cut
        return attn_out

class InterHV_SA(nn.Module):
    def __init__(self,dim, head_num, window_size=None, shift_size=0, dct='', norm_type='LayerNorm', modulater=False):
        super(InterHV_SA, self).__init__()
        self.dct = dct  # True if 'dct' in dct else False
        self.norm_type = norm_type
        self.shift_size = shift_size
        self.window_size = window_size
        if dct == 'dct2d':
            self.dctx = DCT2x()
            self.idctx = IDCT2x()
        elif dct == 'dct1d_w':
            self.dctx = DCT1x(-1)
            self.idctx = IDCT1x(-1)
        elif dct == 'dct1d_h':
            self.dctx = DCT1x(-2)
            self.idctx = IDCT1x(-2)
        elif dct == 'dct1d_hv':
            self.dctxh = DCT1x(-2)
            self.idctxh = IDCT1x(-2)
            self.dctxv = DCT1x(-1)
            self.idctxv = IDCT1x(-1)
        elif dct == 'dct1d_vh':
            self.dctxh = DCT1x(-1)
            self.idctxh = IDCT1x(-1)
            self.dctxv = DCT1x(-2)
            self.idctxv = IDCT1x(-2)
        self.hidden_size = dim
        if norm_type == 'LayerNorm':
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = LayerNorm2d(dim // 2)
                self.attention_norm_v = LayerNorm2d(dim // 2)
            else:
                self.attention_norm = LayerNorm2d(dim)
        elif norm_type == 'InstanceNorm':
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = nn.InstanceNorm2d(dim // 2)
                self.attention_norm_v = nn.InstanceNorm2d(dim // 2)
            else:
                self.attention_norm = nn.InstanceNorm2d(dim)
        else:
            if dct == 'dct1d_hv' or dct == 'dct1d_vh':
                self.attention_norm_h = nn.Identity()
                self.attention_norm_v = nn.Identity()
            else:
                self.attention_norm = nn.Identity()

        self.head_num = head_num
        if dct == 'dct1d_hv' or dct == 'dct1d_vh':
            self.conv_input_h = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, padding=0)
            self.conv_input_v = nn.Conv2d(dim // 2, dim // 2, kernel_size=1, padding=0)
        else:
            self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.head_num = head_num

        # self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = AttentionHV(head_num=self.head_num)
        self.modulater = modulater
        if modulater and (dct in ['dct2d', 'dct1d_w', 'dct1d_h']):
            self.dct_mask = nn.Parameter(torch.FloatTensor(1, 1, window_size, window_size), requires_grad=True)

    def forward(self, x):

        # short_cut = x
        if self.shift_size > 0:
            if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
                feature_h, feature_v = torch.chunk(x, 2, dim=1)
                feature_h = torch.roll(feature_h, shifts=(-self.shift_size), dims=(-1))
                feature_v = torch.roll(feature_v, shifts=(-self.shift_size), dims=(-2))
                x = torch.cat([feature_h, feature_v], dim=1)
            else:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        B, C, H, W = x.size()
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        b, c, h, w = x.shape
        if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
            feature_h, feature_v = torch.chunk(x, 2, dim=1)
            feature_h = self.dctxh(feature_h)
            feature_v = self.dctxv(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            # feature_v = feature_v.permute(0, 2, 1, 3).contiguous()
            # feature_h = feature_h.permute(0, 3, 2, 1).contiguous()
            feature_h = self.attention_norm_h(feature_h)
            feature_v = self.attention_norm_v(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            # feature_v = feature_v.permute(0, 2, 1, 3).contiguous()
            # feature_h = feature_h.permute(0, 3, 2, 1).contiguous()
            feature_h = self.conv_input_h(feature_h)
            feature_v = self.conv_input_v(feature_v)
        else:
            if self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
                x = self.dctx(x)
                if self.modulater:
                    x = x * torch.sigmoid(self.dct_mask)
            # feature_v = self.dctx(feature_v)
            # print(x.shape)
            x = self.attention_norm(x)
            x = self.conv_input(x)
            feature_h, feature_v = torch.chunk(x, 2, dim=1)

        feature_h = torch.chunk(self.conv_h(feature_h), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(feature_v), 3, dim=1)

        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        horizontal_groups = horizontal_groups.view(3*b, h, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*b, w, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]

        if h == w:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
        attention_output_h = attention_output_h.view(b, h, c//2, w).permute(0, 2, 1, 3).contiguous()
        attention_output_v = attention_output_v.view(b, w, c//2, h).permute(0, 2, 3, 1).contiguous()
        if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
            attention_output_h = self.idctxh(attention_output_h)
            attention_output_v = self.idctxv(attention_output_v)
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
            # attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        elif self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
            attn_out = self.idctx(attn_out)
            # attn_out = self.fuse_out(attn_out)
        else:
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            attn_out = window_reversex(attn_out, self.window_size, H, W, batch_list)
        if self.shift_size > 0:
            if self.dct == 'dct1d_hv' or self.dct == 'dct1d_vh':
                attention_output_h, attention_output_v = torch.chunk(attn_out, 2, dim=1)
                attention_output_h = torch.roll(attention_output_h, shifts=(-self.shift_size), dims=(-1))
                attention_output_v = torch.roll(attention_output_v, shifts=(-self.shift_size), dims=(-2))
                attn_out = torch.cat([attention_output_h, attention_output_v], dim=1)
            else:
                attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        attn_out = self.fuse_out(attn_out)
        # x = attn_out + short_cut
        return attn_out

class DCTLayerNorm2d(nn.Module):
    def __init__(self, dim, window_size=None, dct='dct2d', norm_type='LayerNorm', modulater=False):
        super(DCTLayerNorm2d, self).__init__()
        self.dct = dct  # True if 'dct' in dct else False
        self.norm_type = norm_type
        self.window_size = window_size
        if dct == 'dct2d':
            self.dctx = DCT2x()
            self.idctx = IDCT2x()
        elif dct == 'dct1d_w':
            self.dctx = DCT1x(-1)
            self.idctx = IDCT1x(-1)
        elif dct == 'dct1d_h':
            self.dctx = DCT1x(-2)
            self.idctx = IDCT1x(-2)
        elif dct == 'dct1d_hv':
            self.dctxh = DCT1x(-2)
            self.idctxh = IDCT1x(-2)
            self.dctxv = DCT1x(-1)
            self.idctxv = IDCT1x(-1)

        if norm_type == 'LayerNorm':
            if dct == 'dct1d_hv':
                self.attention_norm_h = LayerNorm2d(dim // 2)
                self.attention_norm_v = LayerNorm2d(dim // 2)
            else:
                self.attention_norm = LayerNorm2d(dim)
        elif norm_type == 'InstanceNorm':
            if dct == 'dct1d_hv':
                self.attention_norm_h = nn.InstanceNorm2d(dim // 2)
                self.attention_norm_v = nn.InstanceNorm2d(dim // 2)
            else:
                self.attention_norm = nn.InstanceNorm2d(dim)
        else:
            if dct == 'dct1d_hv':
                self.attention_norm_h = nn.Identity()
                self.attention_norm_v = nn.Identity()
            else:
                self.attention_norm = nn.Identity()

        self.modulater = modulater
        if modulater and (dct in ['dct2d', 'dct1d_w', 'dct1d_h']):
            self.dct_mask = nn.Parameter(torch.FloatTensor(1, 1, window_size, window_size), requires_grad=True)

    def forward(self, x):

        # short_cut = x
        B, C, H, W = x.size()
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        b, c, h, w = x.shape
        if self.dct == 'dct1d_hv':
            feature_h, feature_v = torch.chunk(x, 2, dim=1)
            feature_h = self.dctxh(feature_h)
            feature_v = self.dctxv(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            # feature_v = feature_v.permute(0, 2, 1, 3).contiguous()
            # feature_h = feature_h.permute(0, 3, 2, 1).contiguous()
            feature_h = self.attention_norm_h(feature_h)
            feature_v = self.attention_norm_v(feature_v)
            # feature_h = feature_h.permute(0, 2, 1, 3).contiguous()
            # feature_v = feature_v.permute(0, 3, 2, 1).contiguous()
            # feature_v = feature_v.permute(0, 2, 1, 3).contiguous()
            # feature_h = feature_h.permute(0, 3, 2, 1).contiguous()
        else:
            if self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
                x = self.dctx(x)
                if self.modulater:
                    x = x * torch.sigmoid(self.dct_mask)
            # feature_v = self.dctx(feature_v)
            # print(x.shape)
            x = self.attention_norm(x)
        if self.dct == 'dct1d_hv':
            attention_output_h = self.idctxh(feature_h)
            attention_output_v = self.idctxv(feature_v)
            attn_out = torch.cat((attention_output_h, attention_output_v), dim=1)
        elif self.dct in ['dct2d', 'dct1d_w', 'dct1d_h']:
            attn_out = self.idctx(x)
        else:
            attn_out = x
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            attn_out = window_reversex(attn_out, self.window_size, H, W, batch_list)
        # x = attn_out + short_cut

        return attn_out

def save_feature(out_dir, x, min_max='min_max'):
    os.makedirs(out_dir, exist_ok=True)
    img_save = kornia.tensor_to_image(x.cpu())
    for c in range(img_save.shape[-1]):
        # print(img_save.shape)
        if len(img_save.shape) > 3:
            img_save_c = img_save[0, :, :, c]
        else:
            img_save_c = img_save[:, :, c]
        if min_max == 'min_max':
            x_min, x_max = img_save_c.min(), img_save_c.max()
            img_save_c = (img_save_c - x_min) / (x_max - x_min)
        elif min_max == 'log':
            img_save_c = np.log(np.abs(img_save_c)+1.)
            img_save_c = img_save_c / img_save_c.max()
        # else:
        #     img_save_c = img_save_c
        cv2.imwrite(os.path.join(
            out_dir,
            str(c) + '.jpg'
        ), img_save_c * 255.)

def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)

    mean = var_mean[1]

    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature

class MlpBlock(nn.Module):
    """A 1-hidden-layer MLP block, applied over the last dimension."""
    def __init__(self, mlp_dim , dropout_rate=0.,use_bias=True):
        super().__init__()
        self.mlp_dim=mlp_dim
        self.dropout_rate=dropout_rate
        self.use_bias=use_bias
        self.fc1 = nn.Linear(self.mlp_dim, self.mlp_dim,bias=self.use_bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim,bias=self.use_bias)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

def block_images_einops_block(x, patch_size, num_heads):
    """Image to patches."""
    batch, channels, height, width = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = rearrange(
        x, "n (head c) (gh fh) (gw fw) -> n c (gh gw) (head fh fw)", head=num_heads,
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

def unblock_images_einops_block(x, grid_size, patch_size, num_heads):
    """patches to images."""
    x = rearrange(
          x, "n c (gh gw) (head fh fw) -> n (head c) (gh fh) (gw fw)",
          gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1], head=num_heads)
    return x

def block_images_einops_grid(x, grid_size, num_heads):
    """Image to patches."""
    batch, channels, height, width = x.shape
    patch_size_height = height // grid_size[0]
    patch_size_width = width // grid_size[1]
    x = rearrange(
        x, "n (head c) (gh fh) (gw fw) -> n c (fh fw) (head gh gw)", head=num_heads,
        gh=grid_size[0], gw=grid_size[1], fh=patch_size_height, fw=patch_size_width)
    return x

def unblock_images_einops_grid(x, grid_size, patch_size, num_heads):
    """patches to images."""
    x = rearrange(
          x, "n c (fh fw) (head gh gw) -> n (head c) (gh fh) (gw fw)",
          gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1], head=num_heads)
    return x
class GridGatingUnit(nn.Module):#缺n          dim                                                        n1 = x.shape[-3]    n2,
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, n1, dim, use_bias=True):
        super().__init__()
        self.bias = use_bias

        # self.layernorm = LayerNorm(dim)
        self.fc = nn.Sequential(
            # LayerNorm(dim),
            nn.Linear(n1,n1,bias=self.bias),
            nn.GELU()
        )
    def forward(self, x):
        # c = x.size(1)
        # c = c//2
        # u, v = torch.split(x, c, dim=1)
        # # v = self.layernorm(v)
        # v = self.fc(v)
        x = self.fc(x)
        return x # u * (v + 1.)
class MLP_MultiHead_interk_linear(nn.Module):
    def __init__(self, window_size=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
        N = window_size[0] * window_size[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, 1, 1, 1, num_heads, N), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.Tensor(num_heads, N, N), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))

    def forward(self, x, H=None, W=None):

        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.window_size)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b h1 w1 c head (h w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
        # mlp_weight = self.mlp_weight
        x = torch.einsum('knz,bchwkn->bchwkz', self.mlp_weight, x)
        if self.bias:
            x = x + self.bias_weight
        if self.winr:
            x = rearrange(x, 'b h1 w1 c head (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)
class MLP_MultiHead_intrak_linear(nn.Module):
    def __init__(self, patch_num=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.patch_num = patch_num if isinstance(patch_num, list) else [patch_num, patch_num]
        N = patch_num[0] * patch_num[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        # self.mlp_weight = nn.Linear(num_heads * N, num_heads * N, bias=bias)
        if bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, 1, 1, 1,num_heads, N), requires_grad=True)
        self.mlp_weight = nn.Parameter(torch.Tensor(num_heads, N, N), requires_grad=True)
        # x = torch.einsum('knz,bchwkn->bchwkz', self.mlp_weight, x)
        torch.nn.init.kaiming_uniform_(self.mlp_weight, a=math.sqrt(5))

    def forward(self, x, H=None, W=None):
        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.patch_num)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b c h w head (h1 w1)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
        # mlp_weight = self.mlp_weight
        # print(x.shape)
        x = torch.einsum('knz,bchwkn->bchwkz', self.mlp_weight, x)
        if self.bias:
            x = x + self.bias_weight
        # print(x.shape)
        # print(x.shape)
        if self.winr:
            x = rearrange(x, 'b c h w head (h1 w1) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)
class MLP_intrak_linear(nn.Module):
    def __init__(self, patch_num=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.patch_num = patch_num if isinstance(patch_num, list) else [patch_num, patch_num]
        N = patch_num[0] * patch_num[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        self.mlp_weight = nn.Linear(num_heads * N, num_heads * N, bias=bias)
    def forward(self, x, H=None, W=None):
        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.patch_num)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b c h w (head h1 w1)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
        # mlp_weight = self.mlp_weight
        # print(x.shape)
        x = self.mlp_weight(x)
        # print(x.shape)
        # print(x.shape)
        if self.winr:
            x = rearrange(x, 'b c h w (head h1 w1) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)
class DCT_inv_quan(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.Qy = nn.Parameter(torch.tensor([[[[16, 11, 10, 16, 24, 40, 51, 61],
                                       [12, 12, 14, 19, 26, 58, 60, 55],
                                       [14, 13, 16, 24, 40, 57, 69, 56],
                                       [14, 17, 22, 29, 51, 87, 80, 62],
                                       [18, 22, 37, 56, 68, 109, 103, 77],
                                       [24, 35, 55, 64, 81, 104, 113, 92],
                                       [49, 64, 78, 87, 103, 121, 120, 101],
                                       [72, 92, 95, 98, 112, 100, 103, 99]]]], dtype=torch.float)/100., requires_grad=True)
        # print(self.Qy.shape)
        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.window_size = window_size # if isinstance(window_size, list) else [window_size, window_size]
        self.winp = WindowPartition(window_size)
        self.winr = WindowReverse(window_size)

    def forward(self, x):
        H, W = x.shape[-2:]
        x, batch_list = self.winp(x)
        x = self.dct(x)
        x = x * self.Qy
        x = self.idct(x)
        x = self.winr(x, H, W, batch_list)
        return x
class MLP_interk_linear(nn.Module):
    def __init__(self, window_size=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
        N = self.window_size[0] * self.window_size[1]
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        self.mlp_weight = nn.Linear(num_heads * N, num_heads * N, bias=bias)

    def forward(self, x, H=None, W=None):

        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.window_size)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b h1 w1 c (head h w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
        # mlp_weight = self.mlp_weight
        x = self.mlp_weight(x)
        if self.winr:
            x = rearrange(x, 'b h1 w1 c (head h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
            if H and W:
                x = x[:, :, :H, :W]
        return self.act(x)
class MLP_intrak_linear2(nn.Module):
    def __init__(self, patch_num=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.patch_num = patch_num if isinstance(patch_num, list) else [patch_num, patch_num]
        N = patch_num[0] * patch_num[1]

        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        self.mlp_weight = nn.Sequential(nn.Linear(num_heads * N, num_heads * N, bias=bias),
                                        nn.GELU(),
                                        nn.Linear(num_heads * N, num_heads * N, bias=bias)
                                        )
    def forward(self, x, H=None, W=None):
        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.patch_num)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b c h w (head h1 w1)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
        # mlp_weight = self.mlp_weight
        # print(x.shape)
        x = self.mlp_weight(x)
        # print(x.shape)
        # print(x.shape)
        if self.winr:
            x = rearrange(x, 'b c h w (head h1 w1) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=self.patch_num[0], w1=self.patch_num[1])
            if H and W:
                x = x[:, :, :H, :W]
        return x
class MLP_interk_linear2(nn.Module):
    def __init__(self, window_size=8, num_heads=1, bias=False, act=None, winp=True, winr=True):
        super().__init__()

        self.window_size = window_size if isinstance(window_size, list) else [window_size, window_size]
        N = window_size[0] * window_size[1]

        self.bias = bias
        self.winp = winp
        self.winr = winr
        self.num_heads = num_heads
        self.mlp_weight =  nn.Sequential(nn.Linear(num_heads * N, num_heads * N, bias=bias),
                                        nn.GELU(),
                                        nn.Linear(num_heads * N, num_heads * N, bias=bias)
                                        )

    def forward(self, x, H=None, W=None):

        if self.winp:
            H, W = x.shape[-2:]
            x = check_image_size(x, self.window_size)
            x = rearrange(x, 'b (head c) (h1 h) (w1 w) -> b h1 w1 c (head h w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
        # mlp_weight = self.mlp_weight
        x = self.mlp_weight(x)
        if self.winr:
            x = rearrange(x, 'b h1 w1 c (head h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h=self.window_size[0], w=self.window_size[1])
            if H and W:
                x = x[:, :, :H, :W]
        return x

class BlockGatingUnit(nn.Module):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self, n2, dim, use_bias=True):
        super().__init__()
        self.bias = use_bias
        # self.layernorm = LayerNorm(dim)
        self.n2=n2
        self.fc = nn.Sequential(
            # LayerNorm(dim),
            nn.Linear(n2,n2,bias=self.bias),
            nn.GELU()
         )

    def forward(self, x):
        # c = x.size(1)
        # c = c//2
        # u, v = torch.split(x, c, dim=1)
        # v = self.layernorm(v)
        # v = self.fc(v)
        x = self.fc(x)
        return x # u * (v + 1.)
class GridGmlpLayer(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, dim,grid_size, num_heads=1, use_bias=True,factor=2,dropout_rate=0., modulator=False):
        super().__init__()
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim//num_heads)
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.modulator = modulator
        self.gelu = nn.GELU()
        self.num_heads = num_heads
        n = grid_size[0] * grid_size[1]
        self.gridgatingunit = GridGatingUnit(num_heads * n, dim=dim//num_heads,use_bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        if modulator:
            self.modulator = nn.Embedding(dim//num_heads, n * num_heads)
        self.fc1 = nn.Sequential(
            # LayerNorm(dim // num_heads),
            nn.Conv2d(dim//num_heads,dim*self.factor//num_heads, 1,bias=self.bias),
        )
        self.fc2 = nn.Conv2d(dim*self.factor//num_heads,dim//num_heads, 1,bias=self.bias)
    def forward(self, x):
        n, num_channels, h, w = x.shape
        gh, gw = self.grid_size
        x = check_image_size(x, self.grid_size)
        H, W = x.shape[-2:]
        fh, fw = H // gh, W // gw
        x = block_images_einops_grid(x, grid_size=self.grid_size, num_heads=self.num_heads)
        # print(x.shape, self.grid_size)
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        # x = x.permute(0, 1, 3, 2)
        # y = self.layernorm(x)
        if self.modulator: # # print(self.modulator.weight.unsqueeze(1).unsqueeze(0).shape, x.shape)
            x = x * self.modulator.weight.unsqueeze(1).unsqueeze(0)
        y = self.fc1(x)
        y = self.gelu(y)
        # print(y.shape)
        y = self.gridgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = y # x + y
        # x = x.permute(0, 1, 3, 2)
        x = unblock_images_einops_grid(x, grid_size=(gh, gw), patch_size=(fh, fw), num_heads=self.num_heads)
        return x[:, :, :h, :w]
class BlockGmlpLayer(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self,dim, block_size, num_heads=1 ,use_bias=True,factor=2,dropout_rate=0., modulator=False):
        super().__init__()
        self.block_size = block_size
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.num_heads = num_heads
        # self.layernorm = LayerNorm(dim//num_heads)
        self.gelu = nn.GELU()
        self.dim=dim
        self.modulator = modulator
        n = block_size[0] * block_size[1]
        self.blockgatingunit = BlockGatingUnit(n2=num_heads * n,dim=self.dim//num_heads,use_bias=self.bias)
        if modulator:
            self.modulator = nn.Embedding(dim//num_heads, n * num_heads)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Sequential(
            # LayerNorm(dim // num_heads),
            nn.Conv2d(dim//num_heads,dim * self.factor//num_heads, 1,bias=self.bias)
        )
        self.fc2 = nn.Conv2d(dim * self.factor//num_heads,dim//num_heads, 1,bias=self.bias)
    def forward(self, x):
        n, num_channels, h, w = x.shape
        x = check_image_size(x, self.block_size)
        fh, fw = self.block_size
        H, W = x.shape[-2:]
        gh, gw = H // fh, W // fw
        x = block_images_einops_block(x, patch_size=(fh, fw), num_heads=self.num_heads)
        # MLP2: Local (block) mixing part, provides within-block communication.
        # y = self.layernorm(x)
        if self.modulator:
            x = x * self.modulator.weight.unsqueeze(1).unsqueeze(0)
        y = self.fc1(x)
        y = self.gelu(y)
        y = self.blockgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = y # x + y
        x = unblock_images_einops_block(x, grid_size=(gh, gw), patch_size=(fh, fw), num_heads=self.num_heads)
        return x[:, :, :h, :w]
class GridGmlpLayer2(nn.Module):
    """Grid gMLP layer that performs global mixing of tokens."""
    def __init__(self, dim,grid_size, num_heads=1, use_bias=True,factor=2,dropout_rate=0., modulator=False):
        super().__init__()
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim//num_heads)
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.modulator = modulator
        self.gelu = nn.GELU()
        self.num_heads = num_heads
        n = grid_size[0] * grid_size[1]
        self.gridgatingunit = MLP_intrak(grid_size, num_heads, bias=self.bias)
        self.dropout = nn.Dropout(self.drop)

        self.fc1 = nn.Sequential(
            # LayerNorm(dim // num_heads),
            nn.Conv2d(dim,dim*self.factor, 1,bias=self.bias),
        )
        self.fc2 = nn.Conv2d(dim*self.factor,dim, 1,bias=self.bias)
    def forward(self, x):
        y = self.fc1(x)
        y = self.gelu(y)
        # print(y.shape)
        y = self.gridgatingunit(y)
        # print(y.shape)
        y = self.fc2(y)
        y = self.dropout(y)
        return y
class BlockGmlpLayer2(nn.Module):
    """Block gMLP layer that performs local mixing of tokens."""
    def __init__(self,dim, block_size, num_heads=1 ,use_bias=True,factor=2,dropout_rate=0., modulator=False):
        super().__init__()
        self.block_size = block_size
        self.bias = use_bias
        self.factor = factor
        self.drop = dropout_rate
        self.num_heads = num_heads
        # self.layernorm = LayerNorm(dim//num_heads)
        self.gelu = nn.GELU()
        self.dim=dim
        self.modulator = modulator
        n = block_size[0] * block_size[1]
        self.blockgatingunit = MLP_interk(block_size, num_heads, bias=self.bias)
        self.dropout = nn.Dropout(self.drop)
        self.fc1 = nn.Sequential(
            # LayerNorm(dim // num_heads),
            nn.Conv2d(dim, dim * self.factor, 1,bias=self.bias)
        )
        self.fc2 = nn.Conv2d(dim * self.factor,dim, 1,bias=self.bias)
    def forward(self, x):

        y = self.fc1(x)
        y = self.gelu(y)
        y = self.blockgatingunit(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y
class ChannelModulater(nn.Module):
    def __init__(self, dim):
        super(ChannelModulater, self).__init__()

        self.vector_mask = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        return x * self.vector_mask
class ResidualMultiHeadMultiAxisGmlpLayer(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0., modulator=False):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        self.fc1 = nn.Conv2d(dim, dim * self.input_proj_factor, kernel_size=1, bias=self.bias)
        n_c = dim * self.input_proj_factor // 2
        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer(dim=n_c, num_heads=num_heads,grid_size=self.grid_size,
                                           factor=self.grid_gmlp_factor,use_bias=self.bias,
                                           dropout_rate=self.drop, modulator=modulator)
        self.blockgmlplayer = BlockGmlpLayer(dim=n_c, num_heads=num_heads,block_size=self.block_size,
                                             factor=self.block_gmlp_factor,  use_bias=self.bias,
                                             dropout_rate=self.drop, modulator=modulator)
        self.fc2 = nn.Conv2d(dim, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        # c = x.size(1)//2
        # u, v = torch.split(x, c, dim=1)
        u, v = x.chunk(2, dim=1)
        # GridGMLPLayer
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV2(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,
                 dropout_rate=0., modulator=False, window_size=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.window_size = window_size
        if window_size:
            self.winp = WindowPartition(window_size, shift_size=0)
            self.winr = WindowReverse(window_size, shift_size=0)
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        self.fc1 = nn.Conv2d(dim, dim * self.input_proj_factor, kernel_size=1, bias=self.bias)
        n_c = dim * self.input_proj_factor // 2

        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer(dim=n_c, num_heads=num_heads,grid_size=self.grid_size,
                                           factor=self.grid_gmlp_factor,use_bias=self.bias,
                                           dropout_rate=self.drop, modulator=modulator)
        self.blockgmlplayer = BlockGmlpLayer(dim=n_c, num_heads=num_heads,block_size=self.block_size,
                                             factor=self.block_gmlp_factor,  use_bias=self.bias,
                                             dropout_rate=self.drop, modulator=modulator)
        self.fc2 = nn.Conv2d(dim * self.input_proj_factor, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        # if self.window_size: # and 'dct' in self.cs:
        #     h, w = x.shape[-2:]
        #     x, batch_list = self.winp(x)
        x = self.fc1(x)
        x = self.gelu(x)
        # c = x.size(1)//2
        # u, v = torch.split(x, c, dim=1)
        u, v = x.chunk(2, dim=1)
        # GridGMLPLayer
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV3(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0., modulator=False):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        self.fc1 = nn.Conv2d(dim, dim * self.input_proj_factor, kernel_size=1, bias=self.bias)
        n_c = dim * self.input_proj_factor // 2
        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer2(dim=n_c, num_heads=num_heads,grid_size=self.grid_size,
                                           factor=self.grid_gmlp_factor,use_bias=self.bias,
                                           dropout_rate=self.drop, modulator=modulator)
        self.blockgmlplayer = BlockGmlpLayer2(dim=n_c, num_heads=num_heads,block_size=self.block_size,
                                             factor=self.block_gmlp_factor,  use_bias=self.bias,
                                             dropout_rate=self.drop, modulator=modulator)
        self.fc2 = nn.Conv2d(dim * self.input_proj_factor, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        # c = x.size(1)//2
        # u, v = torch.split(x, c, dim=1)
        u, v = x.chunk(2, dim=1)
        # GridGMLPLayer
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_cat(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='grid', block2='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_intrak_linear
        elif block1 == 'block':
            b1 = MLP_interk_linear
        if block2 == 'grid':
            b2 = MLP_intrak_linear
        elif block2 == 'block':
            b2 = MLP_interk_linear
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            b1(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            b2(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_cat_mh(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='grid', block2='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_MultiHead_intrak_linear
        elif block1 == 'block':
            b1 = MLP_MultiHead_interk_linear
        if block2 == 'grid':
            b2 = MLP_MultiHead_intrak_linear
        elif block2 == 'block':
            b2 = MLP_MultiHead_interk_linear
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            b1(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            b2(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiBlockMlpV1(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate

        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_MultiHead_intrak_linear
        elif block1 == 'block':
            b1 = MLP_MultiHead_interk_linear

        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(hidden_feature, hidden_feature, 1, bias=self.bias),
            nn.GELU(),
            b1(block_size, num_heads*2, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(hidden_feature, hidden_feature, 1, bias=self.bias),
            # nn.Dropout(0.5)
        )
        self.fc2 = nn.Conv2d(n_c, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        x = self.fc1(x)

        x = self.blockgmlplayer(x)

        u, v = x.chunk(2, dim=1)
        x = self.fc2(u*v)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiBlockMlpV2(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate

        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_MultiHead_intrak_linear
        elif block1 == 'block':
            b1 = MLP_MultiHead_interk_linear

        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(hidden_feature, hidden_feature, 1, bias=self.bias),
            nn.GELU(),
            b1(block_size, num_heads*2, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            # nn.Conv2d(hidden_feature, hidden_feature, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        x = self.fc1(x)

        x = self.blockgmlplayer(x)
        # u, v = x.chunk(2, dim=1)
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiBlockMlpV3(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate

        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_MultiHead_intrak_linear
        elif block1 == 'block':
            b1 = MLP_MultiHead_interk_linear

        self.dropout = nn.Dropout(self.drop)
        num_heads = num_heads * 2
        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            # nn.Conv2d(hidden_feature, hidden_feature, 3, groups=hidden_feature, padding=1, bias=self.bias),
            nn.Conv2d(hidden_feature, hidden_feature, 1, groups=num_heads, bias=self.bias),
            nn.GELU(),
            b1(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(hidden_feature, hidden_feature, 1, groups=num_heads, bias=self.bias),
            # nn.Conv2d(hidden_feature, hidden_feature, 3, groups=hidden_feature, padding=1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        x = self.fc1(x)

        x = self.blockgmlplayer(x)
        # u, v = x.chunk(2, dim=1)
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_catSD(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='block', block2='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        if block1 == 'grid':
            b1 = MLP_intrak_linear
        elif block1 == 'block':
            b1 = MLP_interk_linear
        if block2 == 'grid':
            b2 = MLP_intrak_linear
        elif block2 == 'block':
            b2 = MLP_interk_linear
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            b1(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            DCT2x(),
            LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            b2(block_size, num_heads, bias=self.bias),
            IDCT2x(),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias),

        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_chuan(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None, block1='grid', block2='block'):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        # n_c = hidden_feature // 2
        if block1 == 'grid':
            b1 = MLP_intrak_linear
        elif block1 == 'block':
            b1 = MLP_interk_linear
        if block2 == 'grid':
            b2 = MLP_intrak_linear
        elif block2 == 'block':
            b2 = MLP_interk_linear
        self.mlplayer = nn.Sequential(
            b1(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),
            b2(block_size, num_heads, bias=self.bias),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(self.drop)

        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        x = self.mlplayer(x)

        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMG_VX(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            # nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            # nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(n_b+n_g, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMG_VX_nofgelu(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            # nn.GELU(),

            # nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            # nn.GELU(),
            # nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(n_b+n_g, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV4(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.gridgmlplayer2(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV5(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(n_c, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.gridgmlplayer2(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        # x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV6(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(n_c*2, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.gridgmlplayer2(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV7(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            LayerNorm2d(n_g//2),
            # nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            # nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),


        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            LayerNorm2d(n_b//2),

            # nn.GELU(),
            MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
        )
        self.pr1_in = nn.Conv2d(n_c, n_g, 1, bias=self.bias)
        self.pr2_in = nn.Conv2d(n_c, n_b, 1, bias=self.bias)
        self.pr1_out = nn.Conv2d(n_g//2, n_c, 1, bias=self.bias)
        self.pr2_out = nn.Conv2d(n_b//2, n_c, 1, bias=self.bias)
        self.fc2 = nn.Conv2d(n_c*2, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.pr1_in(u)
        u1, u2 = u.chunk(2, 1)
        u = self.gridgmlplayer(u1) * u2
        u = self.pr1_out(u)
        # BlockGMLPLayer
        v = self.pr2_in(v)
        v1, v2 = v.chunk(2, 1)
        v = self.gridgmlplayer2(v1) * v2
        v = self.pr2_out(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV8(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            LayerNorm2d(n_g//2),
            # nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            # nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),


        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            LayerNorm2d(n_b//2),
            # nn.GELU(),
            # MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b//2, n_b//2, kernel_size=5, padding=2, groups=n_b//2, bias=self.bias),
            nn.GELU(),
        )
        self.pr1_in = nn.Conv2d(n_c, n_g, 1, bias=self.bias)
        self.pr2_in = nn.Conv2d(n_c, n_b, 1, bias=self.bias)
        self.pr1_out = nn.Conv2d(n_g//2, n_c, 1, bias=self.bias)
        self.pr2_out = nn.Conv2d(n_b//2, n_c, 1, bias=self.bias)
        self.fc2 = nn.Conv2d(n_c*2, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.pr1_in(u)
        u1, u2 = u.chunk(2, 1)
        u = self.gridgmlplayer(u1) * u2
        u = self.pr1_out(u)
        # BlockGMLPLayer
        v = self.pr2_in(v)
        v1, v2 = v.chunk(2, 1)
        v = self.gridgmlplayer2(v1) * v2
        v = self.pr2_out(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiGmlpLayerV9(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_g//2),
            # nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            # nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),


        )
        self.dropout = nn.Dropout(self.drop)

        self.gridgmlplayer2 = nn.Sequential(
            # LayerNorm2d(n_b//2),
            # nn.GELU(),
            # MLP_intrak_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b//2, n_b//2, kernel_size=5, padding=2, groups=n_b//2, bias=self.bias),
            nn.GELU(),
        )
        self.pr1_in = nn.Conv2d(n_c, n_g, 1, bias=self.bias)
        self.pr2_in = nn.Conv2d(n_c, n_b, 1, bias=self.bias)
        self.pr1_out = nn.Conv2d(n_g//2, n_c, 1, bias=self.bias)
        self.pr2_out = nn.Conv2d(n_b//2, n_c, 1, bias=self.bias)
        self.fc2 = nn.Conv2d(n_c*2, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.pr1_in(u)
        u1, u2 = u.chunk(2, 1)
        u = self.gridgmlplayer(u1) * u2
        u = self.pr1_out(u)
        # BlockGMLPLayer
        v = self.pr2_in(v)
        v1, v2 = v.chunk(2, 1)
        v = self.gridgmlplayer2(v1) * v2
        v = self.pr2_out(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class SRMG(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            DCT2x(),
            LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            IDCT2x(),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            # MLP_interk_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_mul(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            # MLP_interk_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u) * u
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * v
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_linear_mul(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u) * u
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * v
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_linear_cross_mul(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u_ = self.gridgmlplayer(u) * v
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * u
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u_, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_linear_cross_mul2(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u) * v
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * u
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_cross_mul(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            # MLP_interk_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u = self.gridgmlplayer(u) * v
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * u
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class RMGV10_cross_mul2(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            # MLP_interk_linear(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b, n_b, kernel_size=5, padding=2, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        # self.idct = IDCT2x()
    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        # out_dir = '/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/Motion_Deblurring/results/feature'

        u, v = x.chunk(2, dim=1)
        # save_feature(os.path.join(out_dir, 'fc1_u'), u)
        # save_feature(os.path.join(out_dir, 'fc1_v'), v)
        # save_feature(os.path.join(out_dir, 'sfc1_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sfc1_v'), self.idct(v))
        u_ = self.gridgmlplayer(u) * v
        # BlockGMLPLayer
        v = self.blockgmlplayer(v) * u
        # save_feature(os.path.join(out_dir, 'grid_u'), u)
        # save_feature(os.path.join(out_dir, 'block_v'), v)
        # save_feature(os.path.join(out_dir, 'sgrid_u'), self.idct(u))
        # save_feature(os.path.join(out_dir, 'sblock_v'), self.idct(v))
        x = torch.cat([u_, v], dim=1)
        # x = torch.cat([u+v, u*v], dim=1)
        # save_feature(os.path.join(out_dir, 'x'), x)
        # save_feature(os.path.join(out_dir, 'sx'), self.idct(x))
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_SCA(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.sca = WSCA(hidden_feature, window_size=None)
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x = self.fc1(x)
        u, v = x.chunk(2, dim=1)
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # x = u * v
        x = self.sca(x)
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_skip(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.fc_skip = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                     nn.GELU(),
                                     nn.Conv2d(hidden_feature, n_c, kernel_size=1, bias=self.bias)
                                     # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                     #           groups=hidden_feature, bias=self.bias)
                                     )
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)
        x_skip = self.fc_skip(x)
        x = self.fc1(x)

        u, v = x.chunk(2, dim=1)
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u + x_skip, u*v], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV4_skip2(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0.,
                 modulator=False,window_size_dct=None):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.window_size_dct = window_size_dct
        self.drop = dropout_rate
        if window_size_dct:
            self.winp = WindowPartition(window_size_dct, shift_size=0)
            self.winr = WindowReverse(window_size_dct, shift_size=0)
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.fc_skip = nn.Sequential(
                                     nn.Conv2d(hidden_feature, n_c * 3, kernel_size=1, bias=self.bias)
                                     # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                     #           groups=hidden_feature, bias=self.bias)
                                     )
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias),
            nn.GELU(),
            MLP_intrak_linear(grid_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_g, n_g, kernel_size=3, padding=1, groups=n_g, bias=self.bias),
            nn.GELU(),

            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias),
            nn.GELU(),
            MLP_interk_linear(block_size, num_heads, bias=self.bias),
            # nn.Conv2d(n_b, n_b, kernel_size=3, padding=1, groups=n_b, bias=self.bias),
            nn.GELU(),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(n_c*3, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # h, w = x.shape[-2:]
        # if self.window_size_dct:  # and 'dct' in self.cs:
        #     x, batch_list = self.winp(x)
            # print(x.shape)

        x = self.fc1(x)
        x_skip = self.fc_skip(x)
        u_, z, v_ = x_skip.chunk(3, dim=1)
        u, v = x.chunk(2, dim=1)
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u + u_, u*v + z, v + v_], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # if self.window_size_dct: #  and 'dct' in self.cs:
        #     x = self.winr(x, h, w, batch_list)
        # x = x + shortcut
        return x
class ResidualMultiHeadMultiAxisGmlpLayerV5(nn.Module):
    """The multi-axis gated MLP block."""
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0., modulator=False):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        hidden_feature = dim * self.input_proj_factor
        self.fc1 = nn.Sequential(nn.Conv2d(dim, hidden_feature, kernel_size=1, bias=self.bias),
                                 nn.GELU(),
                                 # nn.Conv2d(hidden_feature, hidden_feature, kernel_size=3, padding=1,
                                 #           groups=hidden_feature, bias=self.bias)
                                 )

        n_c = hidden_feature // 2
        n_g = n_c * grid_gmlp_factor
        n_b = n_c * block_gmlp_factor
        self.gridgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_g, 1, bias=self.bias, groups=num_heads),
            nn.GELU(),
            MLP_intrak_linear2(grid_size, num_heads, bias=self.bias),
            nn.Conv2d(n_c * grid_gmlp_factor, n_c, 1, bias=self.bias)
        )
        self.dropout = nn.Dropout(self.drop)

        self.blockgmlplayer = nn.Sequential(
            # LayerNorm2d(n_c),
            nn.Conv2d(n_c, n_b, 1, bias=self.bias, groups=num_heads),
            nn.GELU(),
            MLP_interk_linear2(block_size, num_heads, bias=self.bias),
            nn.Conv2d(n_b, n_c, 1, bias=self.bias)
        )
        self.fc2 = nn.Conv2d(hidden_feature, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        x = self.fc1(x)
        u, v = x.chunk(2, dim=1)
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # x = x + shortcut
        return x
class MixerLocal(ResidualMultiHeadMultiAxisGmlpLayerV2):
    def __init__(self, dim, num_heads=1, block_size=[8,8], grid_size=[8,8],
                 block_gmlp_factor=2,grid_gmlp_factor=2 , input_proj_factor=2,use_bias=True,dropout_rate=0., modulator=False,
                 kernel_size=[128, 128]):
        super().__init__(dim, num_heads=num_heads, block_size=block_size, grid_size=grid_size,
                 block_gmlp_factor=block_gmlp_factor,grid_gmlp_factor=grid_gmlp_factor ,
                         input_proj_factor=input_proj_factor,use_bias=use_bias,dropout_rate=dropout_rate, modulator=modulator)
        # N, C, H, W = train_size
        self.block_size = block_size
        self.grid_size = grid_size
        # self.layernorm = LayerNorm(dim)
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.bias = use_bias
        self.drop = dropout_rate
        self.fc1 = nn.Conv2d(dim, dim * self.input_proj_factor, kernel_size=1, bias=self.bias)
        n_c = dim * self.input_proj_factor // 2
        self.gelu = nn.GELU()
        self.gridgmlplayer = GridGmlpLayer(dim=n_c, num_heads=num_heads,grid_size=self.grid_size,
                                           factor=self.grid_gmlp_factor,use_bias=self.bias,
                                           dropout_rate=self.drop, modulator=modulator)
        self.blockgmlplayer = BlockGmlpLayer(dim=n_c, num_heads=num_heads,block_size=self.block_size,
                                             factor=self.block_gmlp_factor,  use_bias=self.bias,
                                             dropout_rate=self.drop, modulator=modulator)
        self.fc2 = nn.Conv2d(dim * self.input_proj_factor, dim, 1, bias=self.bias) #  * self.input_proj_factor
        self.dropout = nn.Dropout(dropout_rate)
        self.overlap_size = (kernel_size[0]//8, kernel_size[1]//8)
        # print(self.overlap_size)
        # print(kernel_size)
        self.kernel_size = kernel_size
        self.fast_imp = False
    def forward_(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        # c = x.size(1)//2
        # u, v = torch.split(x, c, dim=1)
        u, v = x.chunk(2, dim=1)
        # GridGMLPLayer
        u = self.gridgmlplayer(u)
        # BlockGMLPLayer
        v = self.blockgmlplayer(v)
        # x = torch.cat([u, v], dim=1)
        x = torch.cat([u+v, u*v], dim=1)
        # x = u * v
        x = self.fc2(x)
        # x = x
        x = self.dropout(x)
        # x = x + shortcut
        return x
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size# (64, 64)
        stride = (k1-overlap_size[0], k2-overlap_size[1])
        num_row = (h - 1) // stride[0] + 1
        num_col = (w - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
    def forward(self, x):
        # print(self.kernel_size, self.base_size, self.train_size)
        # if self.kernel_size is None and self.base_size:
        #     if isinstance(self.base_size, int):
        #         self.base_size = (self.base_size, self.base_size)
        #     self.kernel_size = list(self.base_size)

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            x = self.grids(x)  # convert to local windows
            # print(x.shape)
            x = self.forward_(x)
            # print(x_fft.shape)
            x = self.grids_inverse(x)  # reverse

        return x
class Mask_Modulater(nn.Module):
    def __init__(self, dim,
                 kernel_size=[128, 128]):
        super().__init__()

        # print(self.overlap_size)
        # print(kernel_size)
        self.mask = nn.Parameter(torch.zeros(1, dim, kernel_size[0], kernel_size[1]))

    def forward(self, x):
        # shortcut = x
        # x = self.layernorm(x)
        x = x * self.mask
        return x

if __name__=='__main__':

    # net = ContentChannelAttention(dim=32, window_size=8, window_size_dct=128, num_heads=2, cs='channel').cuda()
    # net = Window_Local(qkv=False)

    x = torch.randn(2,32,256,256)
    x = x.cuda()
    # QM = QCircleMask_Idx(dim=32, window_size_dct=256)
    # z = QM.forward_gather(x)
    # print(z.shape)
    # z = QM.backward_scatter(z)
    # print(z.shape)
    # net = OrthoLSTM(dim=32, num_heads=1, bias=True,
    #                          window_size=8, grid_size=8,
    #                          temp_div=True, norm_dim=-1, qk_norm=False,
    #                          cs='channel_mlp', proj_out=True, temp_adj=None).cuda()
    # net = DCT_inv_quan(8).cuda()
    net = TopkAttention(dim=32, num_heads=2, bias=True, topk=32).cuda()
    # net = Sparse_act(dim=32, bias=True).cuda()
    # x1 = torch.nn.functional.normalize(x, dim=-1)
    # x1 = rearrange(x1, 'b (head c) h w -> b head c h w', head=4)
    # x2 = torch.nn.functional.normalize(rearrange(x, 'b (head c) h w -> b head c h w', head=4), dim=-1)
    # print(torch.mean(x1-x2))
    # y1 = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=32, w1=32)
    # y2, _ = window_partitionx(x, 8)
    # print(torch.mean(y1 - y2))
    y = net(x)
    # y = net.grids(x)
    # y = net.grids_inverse(y)
    # print(y.shape)
    print(torch.mean(y-x))