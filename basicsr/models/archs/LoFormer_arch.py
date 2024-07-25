

from basicsr.models.archs.arch_util import *
from basicsr.models.archs.dct_util import *
from basicsr.models.archs.up_down import *
from einops import rearrange
# import kornia
import seaborn as sns
import matplotlib.pyplot as plt

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        h, w = x.shape[-2:]
        x = check_image_size(x, self.patch_size)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x[:, :, :h, :w]).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class DCTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DCTFFN, self).__init__()
        self.dct = DCT2x()
        self.idct = IDCT2x()
        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.quant = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        h, w = x.shape[-2:]
        x = check_image_size(x, self.patch_size)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_dct = self.dct(x_patch)
        x_patch_dct = x_patch_dct * self.quant
        x_patch = self.idct(x_patch_dct)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x[:, :, :h, :w]).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, ffn='ffn', window_size=None):
        super(FeedForward, self).__init__()

        self.ffn_expansion_factor = ffn_expansion_factor

        self.ffn = ffn
        if self.ffn_expansion_factor == 0:
            hidden_features = dim
            self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    groups=dim, bias=bias)
        else:
            hidden_features = int(dim*ffn_expansion_factor)
            self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
            self.act = nn.GELU()
            self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dim = dim
        self.hidden_dim = hidden_features
    def forward(self, inp):
        x = self.project_in(inp)
        if self.ffn_expansion_factor == 0:
            x = self.act(self.dwconv(x))
        else:
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = self.act(x1) * x2
        x = self.project_out(x)
        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H*W*C*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*C
        # print("GDFN:{%.2f}"%(flops/1e9))
        return flops
class GEGLU(nn.Module):
    def __init__(self, dim, kernel_size, bias):
        super(GEGLU, self).__init__()

        self.project_in = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=kernel_size, padding=1, groups=dim * 2, bias=bias)
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, inp):
        x = self.project_in(inp)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
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

class TransformerBlock(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 window_size_dct=8,
                 num_k=8,
                 shift_size=0,
                 cs='channel',
                 norm_type=['LayerNorm', 'LayerNorm'],
                 qk_norm=False,
                 temp_adj=None,
                 ffn='ffn',
                 i=None):
        # def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # print(window_size_dct)
        self.window_size_dct = window_size_dct
        # print(window_size_dct)
        # self.out_dir = i

        self.dim = dim
        self.num_k = num_k
        self.window_size = window_size
        self.shift_size = shift_size
        self.cs = cs

        self.window_size_dct = window_size_dct
        temp_div = True  # if not qk_norm else False

        if 'FLOPs' in cs:
            self.dct = DCT2_FLOPs(window_size_dct, window_size_dct)
            self.idct = IDCT2_FLOPs(window_size_dct, window_size_dct)
        elif 'nodct' in cs:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        elif 'dct_torch' in cs:
            self.dct = DCT2x_torch()
            self.idct = IDCT2x_torch()
        else:
            self.dct = DCT2x()
            self.idct = IDCT2x()

        # self.attn = Attention_inter_wsca(dim, num_heads, bias, window_size=window_size, shift_size=shift_size, sca=False)
        if cs != 'identity':
            if norm_type[0] == 'InstanceNorm':
                norm1 = nn.InstanceNorm2d(dim)
            elif norm_type[0] == 'LayerNorm':
                norm1 = LayerNorm(dim, LayerNorm_type) # , out_dir=i)
            elif norm_type[0] == 'LayerNorm2x':
                norm1 = LayerNorm2x(dim, LayerNorm_type)
            elif norm_type[0] == 'LayerNorm2':
                norm1 = LayerNorm(dim*2, LayerNorm_type)
            elif norm_type[0] == 'LayerNorm_mu_sigma':
                norm1 = LayerNorm(dim, LayerNorm_type, True)
            elif norm_type[0] == 'BatchNorm':
                norm1 = nn.BatchNorm2d(dim)
            elif norm_type[0] == 'Softmax':
                norm1 = nn.Softmax(dim=1)
            else:
                norm1 = nn.Identity()
        else:
            norm1 = nn.Identity()
        if norm_type[1] == 'InstanceNorm':
            norm2 = nn.InstanceNorm2d(dim)
        elif norm_type[1] == 'LayerNorm':
            norm2 = LayerNorm(dim, LayerNorm_type)
        elif norm_type[1] == 'LayerNorm2':
            norm2 = LayerNorm(dim*2, LayerNorm_type)
        elif norm_type[1] == 'LayerNorm_mu_sigma':
            norm1 = LayerNorm(dim, LayerNorm_type, True)
        elif norm_type[1] == 'BatchNorm':
            norm2 = nn.BatchNorm2d(dim)
        else:
            norm2 = nn.Identity()
        self.norm1 = norm1

        self.attn = nn.Sequential(
            Attention(dim, num_heads, bias, window_size_dct=window_size_dct,
                         window_size=window_size, grid_size=num_k,
                         temp_div=temp_div, norm_dim=-1, qk_norm=qk_norm,
                         cs=cs, proj_out=True)
        )
        # self.attn = nn.Sequential(
        #     ProAttention(dim, num_heads, bias, window_size_dct=window_size_dct,
        #                   window_size=window_size, grid_size=num_k,
        #                   temp_div=temp_div, norm_dim=-1, qk_norm=qk_norm,
        #                   cs=cs, proj_out=True)
        # )
        self.norm2 = norm2
        if ffn == 'DFFN':
            self.ffn = nn.Sequential(
                DFFN(dim, ffn_expansion_factor, bias)
            )
        elif ffn == 'DCTFFN':
            self.ffn = nn.Sequential(
                DCTFFN(dim, ffn_expansion_factor, bias)
            )
        else:
            self.ffn = nn.Sequential(
                FeedForward(dim, ffn_expansion_factor, bias, ffn=ffn)
            )
        self.ffn_type = ffn

    def forward(self, x):
        # if 'nodct' in self.cs:
        #     x = self.attn(self.norm1(x)) + x
        # else:
        if 'LN_DCT' in self.cs:
            x_dct = self.dct(self.norm1(x))
            x_attn = self.attn(x_dct)
            x = self.idct(x_attn) + x
        else:
            x_dct = self.dct(x)
            x_attn = self.attn(self.norm1(x_dct))
            x_dct = x_dct + x_attn
            x = self.idct(x_dct)

        x_norm2 = self.norm2(x)
        x = x + self.ffn(x_norm2)
        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # dct
        if 'nodct' in self.cs:
            flops += 0
        else:
            flops += self.dct.flops(inp_shape)
            flops += self.idct.flops(inp_shape)
        # LN * 2
        flops += 2 * H * W * C
        for blk in self.attn:
            flops += blk.flops(inp_shape)
        for blk in self.ffn:
            flops += blk.flops(inp_shape)
        return flops


class TransformerBlock_2b(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 window_size_dct=8,
                 num_k=8,
                 norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                 qk_norm=[False, False],
                 cs=['channel', 'channel'],
                 temp_adj=None,
                 i=None,
                 ffn='ffn'):
        super().__init__()
        # print(window_size_dct)
        window_size_dct1 = None if window_size_dct < 1 else window_size_dct
        window_size_dct2 = None if window_size_dct < 1 else window_size_dct
        #     shift_size_ = [0, 0] # window_size_dct // 2] # [0, window_size_dct // 2]
        # else:
        #     window_size_dct1, window_size_dct2 = None, None
        shift_size_ = [0, 0]  # [0, window_size_dct // 2] # [0, 0]
        # print(cs, norm_type_, qk_norm)
        self.trans1 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct1, num_k=num_k,
                                       shift_size=shift_size_[0], cs=cs[0], norm_type=norm_type_[0],
                                       qk_norm=qk_norm[0], temp_adj=temp_adj, ffn=ffn
                                       )
        self.trans2 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct2, num_k=num_k,
                                       shift_size=shift_size_[1], cs=cs[1], norm_type=norm_type_[1],
                                       qk_norm=qk_norm[1], temp_adj=temp_adj, ffn=ffn
                                       )
        # self.conv_b = conv_bench(dim)
    def forward(self, x):
        x = self.trans1(x)
        x = self.trans2(x)

        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += self.trans1.flops(inp_shape)
        flops += self.trans2.flops(inp_shape)
        return flops

class TransformerBlock_2b_save(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 ffn_expansion_factor=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 window_size_dct=8,
                 num_k=8,
                 norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                 qk_norm=[True, True],
                 cs=['channel', 'channel'],
                 temp_adj=None,
                 i='',
                 ffn='ffn'):
        super().__init__()

        window_size_dct1 = window_size_dct
        window_size_dct2 = window_size_dct

        shift_size_ = [0, 0]  # [0, window_size_dct // 2] # [0, 0]
        self.trans1 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct1, num_k=num_k,
                                       shift_size=shift_size_[0], cs=cs[0], norm_type=norm_type_[0],
                                       qk_norm=qk_norm[0], temp_adj=temp_adj, ffn=ffn, i=i+'_1'
                                       )
        self.trans2 = TransformerBlock(dim, num_heads, ffn_expansion_factor,
                                       bias, LayerNorm_type, window_size, window_size_dct2, num_k=num_k,
                                       shift_size=shift_size_[1], cs=cs[1], norm_type=norm_type_[1],
                                       qk_norm=qk_norm[1], temp_adj=temp_adj, ffn=ffn, i=i+'_2'
                                       )
        # self.conv_b = conv_bench(dim)
    def forward_(self, x):
        x = self.trans1(x)
        x = self.trans2(x)

        return x
    def forward(self, x):
        # if 'nodct' in self.cs:
        #     x = self.attn(self.norm1(x)) + x
        # else:
        if 'LN_DCT' in self.cs:
            x_dct = self.dct(self.norm1(x))
            x_attn = self.attn(x_dct)
            x = self.idct(x_attn) + x
        else:
            x_dct = self.dct(x)
            x_attn = self.attn(self.norm1(x_dct))
            x_dct = x_dct + x_attn
            x = self.idct(x_dct)
        import os
        import cv2
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_dir = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/ICCV2023/rebuttal/circle/LoFT'
        os.makedirs(save_dir, exist_ok=True)
        x_feature = kornia.enhance.normalize_min_max(x, 0., 1.)
        x_feature = kornia.tensor_to_image(x_feature.cpu())
        for c in range(x_feature.shape[-1]):
            x_c = x_feature[:, :, c]
            # cv2.imwrite(os.path.join(save_dir, 'circle_'+str(c)+'.png'), x_c * 255.)
            sns_plot = plt.figure()
            sns.heatmap(x_c, cmap='RdBu_r', linewidths=0.0, vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
            # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
            out_way = os.path.join(save_dir, 'heatmap_circle_'+str(c)+'.png')
            sns_plot.savefig(out_way, dpi=700)
            plt.close()
        x_feature = rearrange(x_feature, 'h w (k c1 c2)-> (c1 h) (c2 w) k', k=1, c1=8, c2=8)
        h, w = x_feature.shape[:2]
        x_feature = np.reshape(x_feature, (h, w))
        sns_plot = plt.figure()
        sns.heatmap(x_feature, cmap='RdBu_r', linewidths=0.0, vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
        # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
        out_way = os.path.join(save_dir, 'heatmap_circle_all' + '.png')
        sns_plot.savefig(out_way, dpi=700)
        plt.close()
        x_norm2 = self.norm2(x)
        x = x + self.ffn(x_norm2)
        return x
    def flops(self, ):
        flops = self.trans1.flops()
        flops += self.trans2.flops()

        return flops
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
    #     torch 1.7.1
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        # return self.body(x)

        return rearrange(self.body(x), 'b c (h h1) (w w1) -> b (c h1 w1) h w', h1=2, w1=2)
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += H * W * C * (C//2) * (3 * 3 + 1)
        return flops
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0

        flops += H * W * C * (C * 2) * (3 * 3 + 1)
        return flops
def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x

class LoFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 enc_blk_nums=[1, 2, 3],
                 middle_blk_num=7,
                 dec_blk_nums=[3, 2, 2],
                 heads_enc=[1, 2, 4],
                 heads_mid=8,
                 window_size_enc=[8, 8, 8],  # [64, 32, 16, 8],
                 grid_size_enc=[8, 8, 8],
                 window_size_dct_enc=[0, 0, 0],
                 window_size_mid=8,
                 grid_size_mid=8,
                 window_size_dct_mid=8,
                 ffn_expansion_factor=2.66,
                 bias=True,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False, ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 up_method='upshuffle',
                 cs_e=['channel_mlp', 'channel_mlp'],
                 cs_m=['channel_mlp', 'channel_mlp'],
                 cs_d=['channel_mlp', 'channel_mlp'],
                 norm_type_=[['LayerNorm', 'LayerNorm'], ['LayerNorm', 'LayerNorm']],
                 qk_norm=[False, False],
                 train_size=None,
                 temp_adj=None,
                 return_feat=False,
                 ffn='ffn',
                 out_method=None,
                 decoder_select=False
                 ):

        super(LoFormer, self).__init__()
        self.padder_size = 8  # 32 64
        # self.inference = inference
        # if not isinstance(cs[0], list):
        #     cs = [cs for _ in range(4)]
        # print(cs)
        # window_size = [False, False, False, False]
        self.decoder_select = decoder_select
        self.train_size = train_size
        if train_size:
            self.winp = WindowPartition(train_size)
            self.winr = WindowReverse(train_size)

        print(window_size_enc, grid_size_enc, train_size)

        self.return_feat = return_feat
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.grid = True
        self.overlap_size = (32, 32)
        self.out_method = out_method
        self.out_channels = out_channels
        # self.overlap_size = (16, 16)
        print(self.overlap_size)
        self.kernel_size = [train_size, train_size]

        TransformerBlockx = TransformerBlock_2b
        if up_method == 'freq_up':
            Upsample_method = UpShuffle_freq
        else:
            Upsample_method = Upsample
            # save_root = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/cvpr2023/figs/fig7-JS/v1'
        # save_list_encoder = [os.path.join(save_root, 'encoder_level' + str(i)) for i in range(4)]
        # save_list_decoder = [os.path.join(save_root, 'decoder_level' + str(i)) for i in range(4)]

        if not self.return_feat:
            self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # self.dcts = nn.ModuleList()
        # self.idcts = nn.ModuleList()
        self.reduce_chan = nn.ModuleList()
        chan = dim
        heads_dec = heads_enc[::-1]
        window_size_dec = window_size_enc[::-1]
        grid_size_dec = grid_size_enc[::-1]
        window_size_dct_dec = window_size_dct_enc[::-1]
        # print(window_size_dct_enc)
        for j in range(len(enc_blk_nums)):
            # self.dcts.append(DCT2x())
            # self.idcts.append(IDCT2x())
            self.encoders.append(
                nn.Sequential(
                    *[TransformerBlockx(dim=chan, num_heads=heads_enc[j], ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type,
                                        window_size=window_size_enc[j], window_size_dct=window_size_dct_enc[j],
                                        num_k=grid_size_enc[j],
                                        cs=cs_e, norm_type_=norm_type_, qk_norm=qk_norm, temp_adj=temp_adj,
                                        i=None, ffn=ffn) for _ in range(enc_blk_nums[j])]
                )
            )
            self.downs.append(
                Downsample(chan)
            )
            chan = chan * 2
        # print(NAFBlock)
        # self.dct_mid = DCT2x()
        # self.idct_mid = IDCT2x()
        self.middle_blks = \
            nn.Sequential(
                *[TransformerBlockx(dim=chan, num_heads=heads_mid, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type,
                                    window_size=window_size_mid, window_size_dct=window_size_dct_mid,
                                    num_k=grid_size_mid,
                                    cs=cs_m, norm_type_=norm_type_, qk_norm=qk_norm, temp_adj=temp_adj,
                                    i=None, ffn=ffn) for _ in range(middle_blk_num)]
            )

        for j in range(len(dec_blk_nums)):
            self.ups.append(
                Upsample_method(chan)
            )

            if j < len(dec_blk_nums) - 1:
                self.reduce_chan.append(nn.Conv2d(int(chan), int(chan // 2), kernel_size=1, bias=bias))
                chan = chan // 2
            else:
                self.reduce_chan.append(nn.Identity())

            self.decoders.append(
                nn.Sequential(
                    *[TransformerBlockx(dim=chan, num_heads=heads_dec[j], ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type,
                                        window_size=window_size_dec[j], window_size_dct=window_size_dct_dec[j],
                                        num_k=grid_size_dec[j],
                                        cs=cs_d, norm_type_=norm_type_, qk_norm=qk_norm, temp_adj=temp_adj,
                                        i=None, ffn=ffn) for _ in range(dec_blk_nums[j])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################
        self.h = None
        self.w = None
        self.window_size_cnt = 2
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.dim = dim
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        self.stride = stride
        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1 * 2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :, -self.overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def cal_best(self, inp):
        # inp [n, b, c, h, w]
        n, b, c = inp.shape[:3]
        x = torch.fft.rfft2(inp)
        x_real = torch.relu(x.real)
        x_imag = torch.relu(x.imag)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x) - inp / 2.
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = kornia.geometry.center_crop3d(x, [self.out_channels, self.window_size_cnt, self.window_size_cnt])
        # print(torch.mean(x, dim=[2, 3], keepdim=True))
        # print(x.shape)
        x, idx = torch.max(torch.mean(x, dim=[-1, -2], keepdim=False), dim=0, keepdim=False)

        # print('idx: ', idx.shape)
        # idx = 0
        results = []
        for batch in range(b):
            inp_b = torch.index_select(inp, dim=1, index=torch.tensor(batch, device=x.device))
            results_batch = []
            for channel in range(c):
                inp_c = torch.index_select(inp_b, dim=2, index=torch.tensor(channel, device=x.device))
                # print('inp_c: ', inp_c.shape)
                inp_z = torch.index_select(inp_c, dim=0, index=idx[batch, channel])
                results_batch.append(inp_z)
                # print(inp_z.shape)
            result_b = torch.cat(results_batch, dim=2)
            results.append(result_b)
        result = torch.cat(results, dim=1)

        return result.squeeze(0)  # inp[idx, ...].unsqueeze(0)

    def return_output(self, x, inp_img_, add_inp=False):
        if self.out_method == 'fourier_select':
            x = self.output(x, inp_img_)
        else:
            x = self.output(x)
            if add_inp:
                x = x + inp_img_
        return x

    def forward(self, inp_img, file_name=None):
        # print(inp_img.shape)
        # inp_img = kornia.geometry.rescale(inp_img, (2, 2))
        B, C, H, W = inp_img.shape

        # x_attn, batch_list = window_partitionx(x, self.window_size_dct)
        # x_attn = self.attn(x_attn)
        # x_attn = window_reversex(x_attn, self.window_size_dct, h, w, batch_list)
        # x = x + x_attn
        if self.train_size and not self.grid:
            inp_img_, batch_list = self.winp(inp_img)
        elif self.train_size and self.grid:
            inp_img_ = self.grids(inp_img)
        else:  # self.train_size and not self.grid:
            inp_img_ = self.check_image_size(inp_img)
        # else:
        #     inp_img_ = inp_img
        h, w = inp_img_.shape[-2:]
        inp_enc_level1 = self.patch_embed(inp_img_)
        encs = []
        x = inp_enc_level1
        for encoder, down in zip(self.encoders, self.downs):
            # x = dctx(x)
            x = encoder(x)
            # x = idctx(x)
            encs.append(x)
            x = down(x)
        # x = self.dct_mid(x)
        x = self.middle_blks(x)
        # x = self.idct_mid(x)

        # outputs_tmp =
        for decoder, up, enc_skip, reduce_ch in zip(self.decoders, self.ups, encs[::-1], self.reduce_chan):
            x = up(x)
            x = torch.cat([x, enc_skip], dim=1)
            x = reduce_ch(x)
            # x = dctx(x)
            # print(x.shape)
            x = decoder(x)

            # x = idctx(x)
        if file_name is not None:
            self.save_feauure(x, file_name)
        # print(inp_img_.shape, x.shape)
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            x = x + self.skip_conv(inp_enc_level1)
            if not self.return_feat:
                x = self.return_output(x, inp_img_, False)
        ###########################
        else:
            if not self.return_feat:
                x = self.return_output(x, inp_img_, False)
        if self.train_size and not self.grid:
            x = self.winr(x, h, w, batch_list)
        elif self.train_size and self.grid:
            x = self.grids_inverse(x)
        if self.out_method == 'fourier_select':
            return x[:, :, :H, :W].contiguous()
        else:
            x = x[:, :, :H, :W].contiguous() + inp_img

            return x
    def save_feauure(self, x_feature, file_name):
        # x_feature = kornia.enhance.normalize_min_max(x_feature, 0., 1.)
        x_feature_mean = torch.mean(x_feature, dim=1)
        x_feature_mean = kornia.enhance.normalize_min_max(x_feature_mean, 0., 1.)

        x_feature_mean = kornia.tensor_to_image(x_feature_mean)

        sns_plot = plt.figure()
        sns.heatmap(x_feature_mean, cmap='RdBu_r', linewidths=0.0, vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False, cbar=True, square=True)  # Reds_r .invert_yaxis()
        # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
        out_way = os.path.join(file_name)
        sns_plot.savefig(out_way, dpi=700)
        plt.close()
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def flops(self, H, W):
        # from ptflops import get_model_complexity_info
        inp_shape = (self.inp_channels, H, W)
        C, H, W = inp_shape
        flops = 0

        flops += H * W * self.inp_channels * self.dim * 3 * 3
        inp_shape = (self.dim, H, W)
        for encoder, down in zip(self.encoders, self.downs):
            print(inp_shape)
            for blk in encoder:
                flops += blk.flops(inp_shape)
            flops += down.flops(inp_shape)
            c_, h_, w_ = inp_shape

            inp_shape = (c_*2, h_//2, w_//2)
        for blk in self.middle_blks:
            flops += blk.flops(inp_shape)

        for j, decoder, up, reduce_ch in zip(range(len(dec_blk_nums)), self.decoders, self.ups, self.reduce_chan):
            print(inp_shape)
            flops += up.flops(inp_shape)
            c_, h_, w_ = inp_shape

            if j < len(dec_blk_nums) - 1: # reduce_ch
                inp_shape = (c_ // 2, h_ * 2, w_ * 2)
                flops += H * W * self.out_channels * self.dim
            else:
                inp_shape = (c_, h_ * 2, w_ * 2)
            for blk in decoder:
                flops += blk.flops(inp_shape)
            # flops += decoder.flops(inp_shape)
        flops += H * W * self.out_channels * self.dim * 3 * 3
        # print(flops / 1e9)
        return flops


if __name__=='__main__':
    import torch
    import kornia
    import cv2
    import os

    # fuse_matrix1 = torch.ones(7, 16) * torch.linspace(1., 0., 7).view(7, 1)
    # print(fuse_matrix1)
    heads = [1, 2, 4, 8]
    window_size_dct_enc = [256, 128, 64]
    window_size_dct_mid = 32

    model_type = 'LoFormer-L' # LoFormer-B Restormer
    attn_type = 'channel_mlp'
    if model_type == 'LoFormer-S':
        # LoFormer-S  36.26 38.24
        dim = 32
        enc_blk_nums = [1, 2, 3]
        middle_blk_num = 7
        dec_blk_nums = [3, 2, 2]
    elif model_type == 'LoFormer-B':
        # LoFormer-B
        dim = 36
        enc_blk_nums = [1, 2, 6]
        middle_blk_num = 9
        dec_blk_nums = [6, 2, 2]
    elif model_type == 'LoFormer-L':
        # LoFormer-B
        dim = 48
        enc_blk_nums = [1, 2, 6]
        middle_blk_num = 9
        dec_blk_nums = [6, 2, 2]
    else:
        # Restormer
        dim = 48
        enc_blk_nums = [2, 3, 3]
        middle_blk_num = 4
        dec_blk_nums = [3, 3, 4]
    cs_e = [attn_type, attn_type]
    cs_m = [attn_type, attn_type]
    cs_d = [attn_type, attn_type]
    net = LoFormer(dim=dim, enc_blk_nums=enc_blk_nums, middle_blk_num=middle_blk_num,dec_blk_nums=dec_blk_nums,
                       cs_e=cs_e, cs_m=cs_m, cs_d=cs_d,
                       window_size_dct_enc=window_size_dct_enc, window_size_dct_mid=window_size_dct_mid, bias=True).cuda()
    # net = FourierSelectOutBlock(dim=32, out_dim=3).cuda()
    # x = torch.randn(4,4,3,256,256)
    # a = torch.randn(4, 3, 256, 256).cuda()
    # x = x.cuda()
    # z = torch.randn(1, 32, 256, 256)
    # z = z.cuda()
    # # y = net(z, x)
    # y = net.cal_best(x)
    # net.flops(256, 256)
    print('# model_restoration parameters: %.2f M' % (
                sum(param.numel() for param in net.parameters()) / 1e6))
    print("number of GFLOPs: %.2f G" % (net.flops(256, 256) / 1e9))
    # y = net.grids(x)
    # y = net.grids_inverse(y)
    # print(y.shape)
    # print(torch.mean(y-a))
    inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    # # print(params)
    # # params = float(params[:-3])
    # # macs = float(macs[:-4])
    #
    # print('FLOPs: ', macs)
    # print('params: ', params)