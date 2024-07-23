from dct_util import *
from einops import rearrange
import numbers
import math
import torch.nn.functional as F
def d4_to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def d3_to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
def d5_to_3d(x):
    # x = x.permute(0, 2, 1, 3, 4)
    return rearrange(x, 'b c s h w -> b (s h w) c')


def d3_to_5d(x, s, h, w):
    x = rearrange(x, 'b (s h w) c -> b c s h w', s=s, h=h, w=w)
    # x = x.permute(0, 2, 1, 3, 4)
    return x
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias, mu_sigma=False):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.mu_sigma = mu_sigma
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.norm_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):

        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if self.norm_bias:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        else:
            x = (x - mu) / torch.sqrt(sigma+1e-5) * self.weight
        if self.mu_sigma:
            return x, mu, sigma
        else:
            return x
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True, mu_sigma=False, out_dir=None):
        super(LayerNorm, self).__init__()
        self.mu_sigma = mu_sigma
        self.body = WithBias_LayerNorm(dim, bias, mu_sigma)
        self.out_dir = out_dir
        # if self.out_dir:
        #     self.dct = DCT2x()

    def forward(self, x):
        # if self.out_dir:
        #     save_feature(self.out_dir+'/beforeLN', x, min_max='log')
        h, w = x.shape[-2:]
        x = d4_to_3d(x)

        if self.mu_sigma:
            x, mu, sigma = self.body(x)
            return d3_to_4d(x, h, w), d3_to_4d(mu, h, w), d3_to_4d(sigma, h, w)
        else:
            x = self.body(x)
            # if self.out_dir:
            #     x = d3_to_4d(x, h, w)
            #     save_feature(self.out_dir + '/afterLN', x, min_max='log')
            #     return x
            # else:
            return d3_to_4d(x, h, w)

def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    if isinstance(padder_size, int):
        padder_size_h = padder_size
        padder_size_w = padder_size
    else:
        padder_size_h, padder_size_w = padder_size
    mod_pad_h = (padder_size_h - h % padder_size_h) % padder_size_h
    mod_pad_w = (padder_size_w - w % padder_size_w) % padder_size_w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, padding_mode='zeros'):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        self.mlp = nn.Sequential(
            nn.Linear(window_size**2, window_size**2, bias=True),
            nn.GELU(),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                stride=1, padding=1, groups=dim * 3, bias=bias, padding_mode=padding_mode)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(dim))


        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)


    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (z head c) (h1 h) (w1 w) -> z (b h1 w1) head c (h w)', z=3, head=self.num_heads,
                        h=self.window_size, w=self.window_size)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out * self.mlp(v)

        out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads, h1=Hx//self.window_size,
                        w1=Wx//self.window_size, h=self.window_size, w=self.window_size)

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
        flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        # MGate
        flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops
class FreqLCBlock(nn.Module):
    def __init__(self, dim=32,
                 num_heads=1,
                 bias=False,
                 LayerNorm_type='WithBias',
                 window_size=8,
                 cs='dct'):
        # def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FreqLCBlock, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.cs = cs

        if 'nodct' in cs:
            self.dct = nn.Identity()
            self.idct = nn.Identity()
        elif 'dct_torch' in cs:
            self.dct = DCT2x_torch()
            self.idct = IDCT2x_torch()
        else:
            self.dct = DCT2x()
            self.idct = IDCT2x()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, window_size=window_size)

    def forward(self, x):
        x_dct = self.dct(x)
        x_attn = self.attn(self.norm1(x_dct))
        x_dct = x_dct + x_attn
        x = self.idct(x_dct)

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
        # LN * 1
        flops += 1 * H * W * C
        flops += self.attn.flops(inp_shape)
        return flops

if __name__=='__main__':
    x = torch.randn(1, 32, 64, 64).cuda()
    model1 = FreqLCBlock(32, cs='dct_torch').cuda()
    model2 = FreqLCBlock(32, cs='dct').cuda()
    model3 = FreqLCBlock(32, cs='nodct').cuda()
    y1 = model1(x)
    y2 = model2(x)
    y3 = model3(x)
    print(y1.shape, y2.shape, y3.shape)
    print(model1.flops((32, 64, 64)), model2.flops((32, 64, 64)), model3.flops((32, 64, 64)))