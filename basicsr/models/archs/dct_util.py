import torch
from torch import nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch.nn import init
# import torch_dct as DCT
# DCT.idct()
# def dct1_torch(x):
#     """
#     Discrete Cosine Transform, Type I
#
#     :param x: the input signal
#     :return: the DCT-I of the signal over the last dimension
#     """
#     x_shape = x.shape
#     x = x.view(-1, x_shape[-1])
#     print(x.shape)
#     print(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1).shape)
#     x = torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), dim=1)
#     print(x.shape)
#     x = x.real
#     print(x.shape)
#     return x.view(*x_shape)
#
#
# def idct1_torch(X):
#     """
#     The inverse of DCT-I, which is just a scaled DCT-I
#
#     Our definition if idct1 is such that idct1(dct1(x)) == x
#
#     :param X: the input signal
#     :return: the inverse DCT-I of the signal over the last dimension
#     """
#     n = X.shape[-1]
#     return dct1_torch(X) / (2 * (n - 1))
def dct(x, W=None, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    # print(v.shape)
    Vc = torch.fft.fft(v, dim=1) # , onesided=False)
    # print(Vc.shape)
    if W is None:
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
    else:
        W_r, W_i = W
        W_r = W_r.to(x.device)
        W_i = W_i.to(x.device)
    V = Vc.real * W_r - Vc.imag * W_i # [:, :N // 2 + 1]

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    # print(V)
    return V

def idct(X, W=None, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    # print(X)
    if W is None:
        k = - torch.arange(N, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
    else:
        W_r, W_i = W
        W_r = W_r.to(X.device)
        W_i = W_i.to(X.device)
    # k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    # W_r = torch.cos(k)
    # W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    # V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, dim=1) # , onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    x = x.real
    return x.view(*x_shape)


def dct_2d_torch(x, N_h=None, N_w=None, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, N_w, norm=norm)
    X2 = dct(X1.transpose(-1, -2), N_h, norm=norm)
    return X2.transpose(-1, -2)


def idct_2d_torch(X, N_h=None, N_w=None, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, N_w, norm=norm)
    x2 = idct(x1.transpose(-1, -2), N_h, norm=norm)
    return x2.transpose(-1, -2)

def get_dctMatrix(m, n):
    N = n
    C_temp = np.zeros([m, n])
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)
    return torch.tensor(C_temp, dtype=torch.float)


def dct1d(feature, dctMat):
    feature = feature @ dctMat.T # dctMat @ feature  #
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


def idct1d(feature, dctMat):
    feature = feature @ dctMat # .T # dctMat.T @ feature  # .T
    return feature.contiguous()  # torch.tensor(x, device=feature.device)


# def dct2d(feature, dctMat):
#     # print(dctMat.shape, feature.shape)
#     feature = dctMat @ feature
#     # print(dctMat.shape, feature.shape)
#     feature = feature @ dctMat.T
#     return feature.contiguous()  # torch.tensor(x, device=feature.device)
def dct2d(feature, dctMat):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat)# dctMat @ feature
    # print(dctMat.shape, feature.shape)
    # feature = feature @ dctMat.T
    # print(feature.transpose(-1, -2).shape, dctMat.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat) # dctMat @ feature.transpose(-1, -2) # @ dctMat.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

# def idct2d(feature, dctMat):
#     feature = dctMat.T @ feature
#     feature = feature @ dctMat
#     return feature.contiguous()  # torch.tensor(x, device=feature.device)
def idct2d(feature, dctMat):
    feature = idct1d(feature, dctMat) # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat)
    return feature.transpose(-1, -2).contiguous() # torch.tensor(x, device=feature.device)

def dct2dx(feature, dctMat1, dctMat2):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat1) # dctMat1 @ feature
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat2) # feature @ dctMat2.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)


def idct2dx(feature, dctMat1, dctMat2):
    feature = idct1d(feature, dctMat1)  # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat2)
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

def dct1_spectral2d_torch(x):
    # b, c, s, h, w = x.shape
    # x = rearrange(x, 'b c s h w -> (b c h w) s')
    x = x.permute(0, 2, 3, 1)
    x = dct(x, 'ortho')
    x = x.permute(0, 3, 1, 2)
    # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
    return x.contiguous() # torch.tensor(x, device=feature.device)

def idct1_spectral2d_torch(x):
    # n = feature.shape[-1]
    # b, c, s, h, w = x.shape
    # x = rearrange(x, 'b c s h w -> (b c h w) s')
    x = x.permute(0, 2, 3, 1)
    x = idct(x, 'ortho')
    x = x.permute(0, 3, 1, 2)
    # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
    return x.contiguous()
# def dct1_spectral2d(x, dctMat):
#     # b, c, s, h, w = x.shape
#     # x = rearrange(x, 'b c s h w -> (b c h w) s')
#     x = x.permute(0, 2, 3, 1)
#     x = x @ dctMat.T
#     x = x.permute(0, 3, 1, 2)
#     # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
#     return x.contiguous() # torch.tensor(x, device=feature.device)
#
# def idct1_spectral2d(x, dctMat):
#     # n = feature.shape[-1]
#     # b, c, s, h, w = x.shape
#     # x = rearrange(x, 'b c s h w -> (b c h w) s')
#     x = x.permute(0, 2, 3, 1)
#     x = x @ dctMat
#     x = x.permute(0, 3, 1, 2)
#     # rearrange(x, ' (b c h w) s -> b c s h w', c=c, h=h, w=w)
#     return x.contiguous()

class SDCTx(nn.Module):
    def __init__(self, heads=1):
        super().__init__()
        self.dctMat = None
        self.heads = heads
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):

        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        self.check_dct_matrix(x.shape[-3])
        self.dctMat = self.dctMat.to(x.device)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x

class ISDCTx(nn.Module):
    def __init__(self, heads=1):
        super().__init__()
        self.heads = heads
        self.dctMat = None
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):
        # self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        self.check_dct_matrix(x.shape[-3])
        self.dctMat = self.dctMat.to(x.device)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x
class SDCT(nn.Module):
    def __init__(self, window_size=64, dynamic=False, heads=1):
        super().__init__()
        if not dynamic:
            self.dctMat = get_dctMatrix(window_size//heads, window_size//heads)
        else:
            self.dctMat = nn.Parameter(get_dctMatrix(window_size//heads, window_size//heads),
                                   requires_grad=True)
        self.heads = heads
    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = dct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
        return x

class ISDCT(nn.Module):
    def __init__(self, window_size=64, dynamic=False, heads=1):
        super().__init__()
        self.heads = heads
        if not dynamic:
            self.dctMat = get_dctMatrix(window_size//heads, window_size//heads)
        else:
            self.dctMat = nn.Parameter(get_dctMatrix(window_size//heads, window_size//heads),
                                   requires_grad=True)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        if self.heads > 1:
            x = rearrange(x, 'b (head c) h w -> b head c h w', head=self.heads)
            # x = rearrange(x, 'b (c head) h w -> b head c h w', head=self.heads)
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = idct1d(x, self.dctMat)
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if self.heads > 1:
            x = rearrange(x, 'b head c h w -> b (head c) h w')
            # x = rearrange(x, 'b head c h w -> b (c head) h w')
        return x
class DCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(DCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        return x


class IDCT1d(nn.Module):
    def __init__(self, window_size=64):
        super(IDCT1d, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)

    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        x = idct1d(x, self.dctMat)
        return x
class DCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(DCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim

    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)

    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape)-1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()


class IDCT1x(nn.Module):
    def __init__(self, dim=-1):
        super(IDCT1x, self).__init__()
        self.dctMat = None
        self.dim = dim
    def check_dct_matrix(self, d):
        if self.dctMat is None or d != self.dctMat.shape[-1]:
            self.dctMat = get_dctMatrix(d, d)
    def forward(self, x):
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        self.check_dct_matrix(x.shape[-1])

        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = idct1d(x, self.dctMat)
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()

class OrthoT1d(nn.Module):
    def __init__(self, n_feature, dim=-1, inference=False):
        super(OrthoT1d, self).__init__()
        self.inference = inference
        self.E = torch.eye(n_feature)
        self.dim = dim
        # self.v_ilst = []
        # for i in range(n_feature):
        #     self.v_ilst.append(
        #         nn.Parameter(torch.randn([n_feature, 1]),requires_grad=True)
        #     )
        self.v_ilst = nn.ParameterList(nn.Parameter(
            torch.randn([n_feature, 1]), requires_grad=True)
                                       for _ in range(n_feature))
        self.A = None
        self.niter = 0

    def get_A(self):
        E = self.E.to(self.v_ilst[0].device)
        self.A = E
        for v in self.v_ilst:
            # print('v: ', (v @ v.transpose(-2,-1)).shape)
            self.A = self.A @ (
                        E - 2. * (v @ v.transpose(-2, -1)) / torch.sum(v.abs().pow(2), dim=[-2, -1], keepdim=True))
        # return A
    def forward(self, x, inverse=False):
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        if not inverse and not self.inference:
            self.get_A()
        elif self.inference and not self.niter:
            self.get_A()
            self.niter += 1
        # A = self.get_A()
        # print(A @ A.T)
        A = self.A.to(x.device)
        if inverse:
            x = x @ A.T
        else:
            x = x @ A
        if self.dim != -1 or self.dim != len(x.shape) - 1:
            x = x.transpose(self.dim, -1)
        return x.contiguous()
class OrthoT2d(nn.Module):
    def __init__(self, h, w, inference=False):
        super(OrthoT2d, self).__init__()
        self.inference = inference
        self.Ev = torch.eye(h)
        self.Eh = torch.eye(w)

        self.v_ilst = nn.ParameterList(nn.Parameter(
            torch.randn([h, 1]), requires_grad=True)
                                       for _ in range(h))
        self.h_ilst = nn.ParameterList(nn.Parameter(
            torch.randn([w, 1]), requires_grad=True)
                                       for _ in range(w))
        # self.h_ilst = []
        # for i in range(w):
        #     self.h_ilst.append(
        #         nn.Parameter(torch.randn([w, 1]), requires_grad=True)
        #     )
        self.Av = None
        self.Ah = None
        # if self.inference:
        #     self.get_A()
    def get_A(self):
        Ev = self.Ev.to(self.v_ilst[0].device)
        Eh = self.Eh.to(self.h_ilst[0].device)
        self.Av = Ev
        self.Ah = Eh
        for v in self.v_ilst:
            # print((v @ v.T).shape)
            self.Av = self.Av @ (Ev - 2. * (v @ v.T) / v.abs().pow(2).sum())
        for h in self.h_ilst:
            # print((v @ v.T).shape)
            self.Ah = self.Ah @ (Eh - 2. * (h @ h.T) / h.abs().pow(2).sum())
        # return Av, Ah
    def forward(self, x, inverse=False):
        if not inverse and not self.inference:
            self.get_A()
        elif self.inference and not self.niter:
            self.get_A()
            self.niter += 1
        # print(A @ A.T)
        Av = self.Av.to(x.device)
        Ah = self.Ah.to(x.device)
        if inverse:
            x = x @ Ah.T
            x = x.transpose(-2, -1) @ Av.T
        else:
            x = x @ Ah
            x = x.transpose(-2, -1) @ Av
        x = x.transpose(-2, -1)
        return x.contiguous()
class DCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(DCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct2d(x, dctMat)
        return x
class DCT2_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2_torch, self).__init__()
        self.norm = norm
        self.W = None
    def forward(self, x):
        # print(x.shape)
        x = dct_2d_torch(x, norm=self.norm)
        return x
class IDCT2_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2_torch, self).__init__()
        self.norm = norm
    def forward(self, x):
        # print(x.shape)
        x = idct_2d_torch(x, norm=self.norm)
        return x
def get_dct_init(N):
    k = - torch.arange(N, dtype=torch.float)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    return [W_r, W_i]
class DCT2x_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x_torch, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm
    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
            # print('Init !!!')
        elif h != self.dctMatH[0].shape[-1] and w != self.dctMatW[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
            # print('xxxxxxxxx')
        elif h != self.dctMatH[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            # print('xxxxxxxxx')
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW[0].shape[-1]:
            self.dctMatW = get_dct_init(w)
            # print('xxxxxxxxx')
        # print(self.dctMatW[0].shape)
    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct_2d_torch(x, self.dctMatH, self.dctMatW, norm=self.norm)
        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops

class IDCT2x_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x_torch, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1] and w != self.dctMatW[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            self.dctMatW = get_dct_init(w)
        elif h != self.dctMatH[0].shape[-1]:
            self.dctMatH = get_dct_init(h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW[0].shape[-1]:
            self.dctMatW = get_dct_init(w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = idct_2d_torch(x, self.dctMatH, self.dctMatW, norm=self.norm)

        return x

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops
class DCT2_FLOPs(nn.Module):
    def __init__(self, H=256, W=256, norm='ortho'):
        super(DCT2_FLOPs, self).__init__()
        self.dcth = nn.Linear(H, H, bias=False)
        self.dctw = nn.Linear(W, W, bias=False)
    def forward(self, x):
        # print(x.shape)
        x = self.dcth(x.transpose(-2, -1))
        x = self.dctw(x.transpose(-2, -1))
        return x
class IDCT2_FLOPs(nn.Module):
    def __init__(self, H=256, W=256, norm='ortho'):
        super(IDCT2_FLOPs, self).__init__()
        self.dcth = nn.Linear(H, H, bias=False)
        self.dctw = nn.Linear(W, W, bias=False)
    def forward(self, x):
        x = self.dcth(x.transpose(-2, -1))
        x = self.dctw(x.transpose(-2, -1))
        return x
class IDCT2(nn.Module):
    def __init__(self, window_size=8, norm='ortho'):
        super(IDCT2, self).__init__()
        self.dctMat = get_dctMatrix(window_size, window_size)
        self.norm = norm
        self.window_size = window_size
    def forward(self, x):
        dctMat = self.dctMat.to(x.device)
        x = idct2d(x, dctMat)
        return x
class RFFT2(nn.Module):
    def __init__(self, norm='ortho'):
        super(RFFT2, self).__init__()
        self.norm = norm

    def forward(self, x):
        x = torch.fft.rfft2(x, norm=self.norm)

        return torch.cat([x.real, x.imag], dim=1)


class IRFFT2(nn.Module):
    def __init__(self, norm='ortho'):
        super(IRFFT2, self).__init__()
        self.norm = norm

    def forward(self, x):
        x_real, x_imag = x.chunk(2, dim=1)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x, norm=self.norm)
        # print(x.shape)
        return x
class DCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct2dx(x, dctMatW, dctMatH)

        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops

class IDCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        x = idct2dx(x, dctMatW, dctMatH)

        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops

class OrthoConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', inference=False):
        super(OrthoConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        # self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        assert M == N
        self.D_mul = M * N
        # self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        # init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        self.inference = inference
        # self.E = nn.Parameter(repeat(torch.eye(M), 'h w -> o c h w', o=out_channels, c=in_channels // groups), requires_grad=False)
        self.E = repeat(torch.eye(M), 'h w -> o c h w', o=out_channels, c=in_channels // groups)
        # self.dim = dim
        # v_ilst = []
        # for i in range(M):
        #     v_ilst.append(
        #         nn.Parameter(torch.randn([out_channels, in_channels // groups, M, 1]), requires_grad=True)
        #     )
        self.v_ilst = nn.ParameterList(nn.Parameter(
            torch.randn([out_channels, in_channels // groups, M, 1]), requires_grad=True)
                                       for _ in range(M))
        # self.A = None
        # self.v_ilst = [torch.randn([out_channels, in_channels // groups, M, 1]) for _ in range(M)]
        #
        # A = self.E  # .to(self.v_ilst[0].device)

        # print('A: ', self.A.shape)
        # for v in self.v_ilst:
        # # print('v: ', (v @ v.transpose(-2,-1)).shape)
        #     A = A @ (self.E - 2. * (v @ v.transpose(-2, -1)) / torch.sum(v.abs().pow(2), dim=[-2, -1], keepdim=True))
        #
        # self.A = nn.Parameter(A, requires_grad=True)
        self.niter = 0
        # if self.inference:
        #     self.get_A()
    def get_A(self):
        E = self.E.to(self.v_ilst[0].device)
        self.A = E  # .to(self.v_ilst[0].device)

        # print('A: ', self.A.shape)
        for v in self.v_ilst:
            # print('v: ', (v @ v.transpose(-2,-1)).shape)
            # tmp = self.A
            self.A = self.A @ (E - 2. * (v @ v.transpose(-2, -1)) / torch.sum(v.abs().pow(2), dim=[-2, -1], keepdim=True))
            # print(torch.mean(E - self.A @ self.A.transpose(-2, -1)))
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(OrthoConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, inverse=False):
        if self.inference and not self.niter:
            self.get_A()
            self.niter += 1
        else:
            self.get_A()
        # print(self.A.device, input.device)
        DoW = self.A.to(input.device)
        # print(DoW.shape)
        return self._conv_forward(input, DoW)
class Ortho_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        self.ortho = OrthoT1d(window_size**2, dim=1)
        self.dct_matrix = get_dctMatrix(window_size, window_size)
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            pad_size = window_size - 1
            self.pad_size = (pad_size // 2, pad_size - pad_size // 2)
            self.pad_sizex = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))


    def ortho_forward(self, x):
        b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size[0], self.pad_size[1], self.pad_size[0], self.pad_size[1]), mode=self.mode)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_sizex - self.window_size) // self.stride + 1, (
                    W + self.pad_sizex - self.window_size) // self.stride + 1
        x = self.ortho(x)

        return rearrange(x, 'b (h w) (k1 k2)  -> b (h w) k1 k2', h=self.window_size, w=self.window_size, k1=self.h, k2=self.w)

    def iortho_forward(self, x):
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        x = rearrange(x, 'b n k1 k2 -> b (k1 k2) n')

        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size[0]:-self.pad_size[1], self.pad_size[0]:-self.pad_size[1]].contiguous()
        x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        # out = self.project_out(x)
        return x

    def forward(self, x, ortho_forward=True):
        if ortho_forward:
            return self.ortho_forward(x)
        else:
            return self.iortho_forward(x)
# Base matrix for luma quantization
T_luma = torch.tensor([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
        ])

# Chroma quantization matrix
Q_chroma = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])
class DCT2d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False, zig_zag=None, zig_out=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        if not dct_torch:
            self.dct2d = DCT2(window_size=window_size, norm='ortho')
            self.idct2d = IDCT2(window_size=window_size, norm='ortho')
        # self.dct_matrix = get_dctMatrix(window_size, window_size)
        if zig_zag:
            self.mask = DctMaskEncoding(vec_dim=zig_zag, mask_size=window_size)
            self.zig = True
        else:
            self.zig = False
        self.zig_out = zig_out
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            pad_size = window_size - 1
            self.pad_size = (pad_size//2, pad_size - pad_size//2)
            self.pad_sizex = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))

    # def get_bound(self, x):
    #     self.dct_matrix = self.dct_matrix.to(x.device)
    #     bound = rearrange(self.dct_matrix, 'h w -> (h w)')
    #     return bound / 2.
    def dct_forward(self, x):
        b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size[0], self.pad_size[1], self.pad_size[0], self.pad_size[1]), mode=self.mode)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_sizex - self.window_size) // self.stride + 1, (
                    W + self.pad_sizex - self.window_size) // self.stride + 1
        x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.dct2d(x)
        else:
            # print(x.max(), x.min())
            x = dct_2d_torch(x, 'ortho')
            # print(x.max(), x.min())
        if self.zig:
            x_zig = self.mask.encode(x)
            # print(x.shape)
            return rearrange(x_zig, 'b (k1 k2) n -> b n k1 k2', k1=self.h, k2=self.w), rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)
        else:
            return rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)
    def idct_forward(self, x):
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)

        # x = self.idct2d(x)
        if self.zig_out:
            x = rearrange(x, 'b n k1 k2 -> b (k1 k2) n')
            x = self.mask.decode(x)

        else:
            x = rearrange(x, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.idct2d(x)
        else:
            x = idct_2d_torch(x, 'ortho')
        x = rearrange(x, 'b n h w -> b (h w) n')
        # print(self.divisor.min(), self.divisor.max())
        x = self.fold(x) / (self.divisor+1e-7)
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size[0]:-self.pad_size[1], self.pad_size[0]:-self.pad_size[1]].contiguous()
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        # out = self.project_out(x)
        return x
    
    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)
        # b, c, H, W = x.shape
        # # _, _, H, W = x.shape
        # x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        # if self.mode == 'reflect':
        #     x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        # if x.shape[-2:] != self.divisor.shape[-2:]:
        #     self.input_ones = torch.ones_like(x)
        #     self.fold = nn.Fold(output_size=x.shape[-2:], **self.fold_params)
        #     self.divisor = self.fold(self.unfold(self.input_ones))
        # if self.divisor.device != x.device:
        #     self.divisor = self.divisor.to(x.device)
        # # print('y: ', x.shape)
        # x = self.unfold(x)
        # # print('unfold: ', x.shape)
        #
        # if self.mode != 'reflect':
        #     h, w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
        #                 W + 2 * self.pad_size - self.window_size) // self.stride + 1
        # else:
        #     h, w = (H + self.pad_size - self.window_size) // self.stride + 1, (
        #             W + self.pad_size - self.window_size) // self.stride + 1
        # x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        # x = self.dct2d(x)
        # # x = self.conv2(self.conv1(x))
        # # x = self.act(x)
        # # x = self.conv3(x) + short_cut
        # x = self.idct2d(x)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=h, w=w)
        #
        # x = self.fold(x) / self.divisor
        # # print('fold: ', x.shape)
        # if self.mode == 'reflect':
        #     x = x[:, :, self.pad_size:, self.pad_size:]
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', b=b, c=c, k=1)
        # # out = self.project_out(x)
        # return x


class MIMO_DCT2d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False, zig_zag=False,
                 zig_out=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        if not dct_torch:
            self.dct2d = DCT2(window_size=window_size, norm='ortho')
            self.idct2d = IDCT2(window_size=window_size, norm='ortho')
        # self.dct_matrix = get_dctMatrix(window_size, window_size)
        if zig_zag:
            # print(window_size)
            self.mask = DctMaskEncoding(vec_dim=window_size**2, mask_size=window_size)
            self.zig = True
        else:
            self.zig = False
        self.zig_out = zig_out
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            pad_size = window_size - 1
            self.pad_size = (pad_size // 2, pad_size - pad_size // 2)
            self.pad_sizex = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128+7, 128+7]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))
        self.shape_x = None
        # self.c = dim
    # def get_bound(self, x):
    #     self.dct_matrix = self.dct_matrix.to(x.device)
    #     bound = rearrange(self.dct_matrix, 'h w -> (h w)')
    #     return bound / 2.

    def dct_forward(self, x):
        b, _, H, W = x.shape

        # _, _, H, W = x.shape
        # x = torch.chunk(x, 3, dim=1)
        # x = torch.cat(x, dim=0)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size[0], self.pad_size[1], self.pad_size[0], self.pad_size[1]), mode=self.mode)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_sizex - self.window_size) // self.stride + 1, (
                    W + self.pad_sizex - self.window_size) // self.stride + 1
        x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.dct2d(x)
        else:
            # print(x.max(), x.min())
            x = dct_2d_torch(x, 'ortho')
            # print(x.max(), x.min())
        if self.zig:
            x_zig = self.mask.encode(x)
            x_zig = rearrange(x_zig, 'b (k1 k2) n -> b n k1 k2', k1=self.h, k2=self.w)
            return x_zig
        else:
            return rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)


    def idct_forward(self, x, rgb=True):
        # print(x.shape)
        if rgb:
            x = torch.chunk(x, 3, dim=1)
            x = torch.cat(x, dim=0)
        shape_x_ = [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
        if self.shape_x is None or self.shape_x != shape_x_:
            self.shape_x = shape_x_ # [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
            # out_shape = [x.shape[-2]*2 + self.window_size, x.shape[-2]*2 + self.window_size]
        # print(self.shape_x)
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        # x = self.idct2d(x)
        # print('x: ', x.shape)
        if self.zig_out:
            x = rearrange(x, 'b n k1 k2 -> b (k1 k2) n')
            x = self.mask.decode(x)
        else:
        # print(x.shape)
            x = rearrange(x, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size)
        # print(x.shape)
        if not self.dct_torch:
            x = self.idct2d(x)
        else:
            x = idct_2d_torch(x, 'ortho')
        # print(x.shape)
        x = rearrange(x, 'b n h w -> b (h w) n')
        # print(self.divisor.min(), self.divisor.max())
        x = self.fold(x) / (self.divisor + 1e-7)
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size[0]:-self.pad_size[1], self.pad_size[0]:-self.pad_size[1]].contiguous()
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        if rgb:
            x = torch.chunk(x, 3, dim=0)
            x = torch.cat(x, dim=1)
        # out = self.project_out(x)
        return x

    def forward(self, x, dct_forward=False, rgb=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x, rgb)
class MIMO_FFT2d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False, zig_zag=False,
                 zig_out=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        if not dct_torch:
            self.dct2d = DCT2(window_size=window_size, norm='ortho')
            self.idct2d = IDCT2(window_size=window_size, norm='ortho')
        # self.dct_matrix = get_dctMatrix(window_size, window_size)
        if zig_zag:
            self.mask = DctMaskEncoding(vec_dim=window_size**2, mask_size=window_size)
            self.zig = True
        else:
            self.zig = False
        self.zig_out = zig_out
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            pad_size = window_size - 1
            self.pad_size = (pad_size // 2, pad_size - pad_size // 2)
            self.pad_sizex = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128+7, 128+7]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))
        self.shape_x = None
        # self.c = dim
    # def get_bound(self, x):
    #     self.dct_matrix = self.dct_matrix.to(x.device)
    #     bound = rearrange(self.dct_matrix, 'h w -> (h w)')
    #     return bound / 2.

    def dct_forward(self, x):
        b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size[0], self.pad_size[1], self.pad_size[0], self.pad_size[1]), mode=self.mode)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_sizex - self.window_size) // self.stride + 1, (
                    W + self.pad_sizex - self.window_size) // self.stride + 1
        x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.dct2d(x)
        else:
            # print(x.max(), x.min())
            x = dct_2d_torch(x, 'ortho')
            # print(x.max(), x.min())
        if self.zig:
            x_zig = self.mask.encode(x)
            # print(x.shape)
            return rearrange(x_zig, 'b (k1 k2) n -> b n k1 k2', k1=self.h, k2=self.w)
        else:
            return rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)

    def idct_forward(self, x):
        # print(x.shape)
        x = torch.chunk(x, 3, dim=1)
        x = torch.cat(x, dim=0)
        shape_x_ = [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
        if self.shape_x is None or self.shape_x != shape_x_:
            self.shape_x = shape_x_ # [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
            # out_shape = [x.shape[-2]*2 + self.window_size, x.shape[-2]*2 + self.window_size]
        # print(self.shape_x)
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        # x = self.idct2d(x)
        # print('x: ', x.shape)
        x_real, x_imag = x.chunk(2, dim=1)
        # print(x_real.shape)
        x_real = rearrange(x_real, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size//2+1)
        x_imag = rearrange(x_imag, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size//2+1)
        # print(x.shape)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x, norm='ortho')
        # print(x.shape)
        x = rearrange(x, 'b n h w -> b (h w) n')
        # print(x.shape)
        # print(self.divisor.min(), self.divisor.max())
        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size[0]:-self.pad_size[1], self.pad_size[0]:-self.pad_size[1]].contiguous()
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        x = torch.chunk(x, 3, dim=0)
        x = torch.cat(x, dim=1)
        # out = self.project_out(x)
        return x

    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)
class MIMO_DCT2d_fold_branchV2(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect', dct_torch=False, zig_zag=False,
                 zig_out=False):
        super().__init__()
        self.window_size = window_size
        # n = window_size ** 2
        self.dct_torch = dct_torch
        if not dct_torch:
            self.dct2d = DCT2(window_size=window_size, norm='ortho')
            self.idct2d = IDCT2(window_size=window_size, norm='ortho')
        # self.dct_matrix = get_dctMatrix(window_size, window_size)
        if zig_zag:
            self.mask = DctMaskEncoding(vec_dim=window_size**2, mask_size=window_size)
            self.zig = True
        else:
            self.zig = False
        self.zig_out = zig_out
        self.stride = stride

        self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))
        self.shape_x = None
        # self.c = dim
    # def get_bound(self, x):
    #     self.dct_matrix = self.dct_matrix.to(x.device)
    #     bound = rearrange(self.dct_matrix, 'h w -> (h w)')
    #     return bound / 2.

    def dct_forward(self, x):
        b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        self.shape_x = x.shape
        # print('y: ', x.shape)
        x = self.unfold(x)
        # print('unfold: ', x.shape)
        x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        if not self.dct_torch:
            x = self.dct2d(x)
        else:
            # print(x.max(), x.min())
            x = dct_2d_torch(x, 'ortho')
            # print(x.max(), x.min())
        if self.zig:
            x_zig = self.mask.encode(x)
            # print(x.shape)
            return rearrange(x_zig, 'b (k1 k2) n -> b n k1 k2', k1=self.h, k2=self.w)
        else:
            return rearrange(x, 'b (k1 k2) h w -> b (h w) k1 k2', k1=self.h, k2=self.w)

    def idct_forward(self, x):
        # print(x.shape)
        x = torch.chunk(x, 3, dim=1)
        x = torch.cat(x, dim=0)
        # print(x.shape)
        shape_x_ = [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
        if self.shape_x is None or self.shape_x != shape_x_:
            self.shape_x = shape_x_ # [1, 1, x.shape[-2] + self.window_size-1, x.shape[-1] + self.window_size-1]
            # out_shape = [x.shape[-2]*2 + self.window_size, x.shape[-2]*2 + self.window_size]
        # print(self.shape_x)
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        # x = self.idct2d(x)
        # print('x: ', x.shape)
        if self.zig_out:
            x = rearrange(x, 'b n k1 k2 -> b (k1 k2) n')
            x = self.mask.decode(x)
        else:
        # print(x.shape)
            x = rearrange(x, 'b (h w) k1 k2 -> b (k1 k2) h w', h=self.window_size, w=self.window_size)
        # print(x.shape)
        if not self.dct_torch:
            x = self.idct2d(x)
        else:
            x = idct_2d_torch(x, 'ortho')
        # print(x.shape)
        x = rearrange(x, 'b n h w -> b (h w) n')
        # print(self.divisor.min(), self.divisor.max())
        x = self.fold(x) / (self.divisor + 1e-7)
        # print('fold: ', x.shape)
        # x = rearrange(x, '(b c) k h w -> b (c k) h w', c=self.c, k=1)
        x = torch.chunk(x, 3, dim=0)
        x = torch.cat(x, dim=1)
        # out = self.project_out(x)
        return x

    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)
class DCT1d_fold_branch(nn.Module):
    def __init__(self, window_size=8, pad_size=0, stride=1, pad_mode='reflect'):
        super().__init__()
        self.window_size = window_size
        n = window_size ** 2
        self.dct1d = DCT1d(window_size=n)
        self.idct1d = IDCT1d(window_size=n)
        self.dct_matrix = get_dctMatrix(n, n)
        self.stride = stride
        self.mode = pad_mode
        if self.mode != 'reflect':
            self.pad_size = pad_size
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=self.pad_size // 2, stride=self.stride)
        else:
            self.pad_size = window_size - 1
            self.fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=self.stride)
        output_size = [1, 1, 128, 128]
        self.fold = nn.Fold(output_size=output_size[-2:], **self.fold_params)
        self.unfold = nn.Unfold(**self.fold_params)

        self.input_ones = torch.ones(output_size)
        self.divisor = self.fold(self.unfold(self.input_ones))
    def dct_forward(self, x):
        self.b, self.c, H, W = x.shape

        # _, _, H, W = x.shape
        x = rearrange(x, 'b (c k) h w -> (b c) k h w', k=1)
        if self.mode == 'reflect':
            x = F.pad(x, (self.pad_size, 0, self.pad_size, 0), mode=self.mode)
        self.shape_x = x.shape

        x = self.unfold(x)
        # print('unfold: ', x.shape)

        if self.mode != 'reflect':
            self.h, self.w = (H + 2 * self.pad_size - self.window_size) // self.stride + 1, (
                    W + 2 * self.pad_size - self.window_size) // self.stride + 1
        else:
            self.h, self.w = (H + self.pad_size - self.window_size) // self.stride + 1, (
                    W + self.pad_size - self.window_size) // self.stride + 1
        # x = rearrange(x, 'b (h w) n -> b n h w', h=self.window_size, w=self.window_size)
        x = self.dct1d(x)

        return rearrange(x, 'b c (k1 k2) -> b c k1 k2', k1=self.h, k2=self.w)
    def get_bound(self, x):
        self.dct_matrix = self.dct_matrix.to(x.device)

    def idct_forward(self, x):
        if self.shape_x[-2:] != self.divisor.shape[-2:]:
            h, w = self.shape_x[-2:]
            self.input_ones = torch.ones([1, 1, h, w])
            self.input_ones = self.input_ones.to(x.device)
            self.fold = nn.Fold(output_size=self.shape_x[-2:], **self.fold_params)
            self.divisor = self.fold(self.unfold(self.input_ones))
        if self.divisor.device != x.device:
            self.divisor = self.divisor.to(x.device)
        # x = rearrange(x, 'b c h w -> b c (h w)', h=self.h, w=self.w)
        x = rearrange(x, 'b c k1 k2 -> b c (k1 k2)')
        x = self.idct1d(x)

        x = self.fold(x) / self.divisor
        # print('fold: ', x.shape)
        if self.mode == 'reflect':
            x = x[:, :, self.pad_size:, self.pad_size:].contiguous()
        x = rearrange(x, '(b c) k h w -> b (c k) h w', b=self.b, c=self.c, k=1)
        # out = self.project_out(x)
        return x
    def forward(self, x, dct_forward=True):
        if dct_forward:
            return self.dct_forward(x)
        else:
            return self.idct_forward(x)

def calcDCTBase(u, v, b=4):
    c_u = 1
    c_v = 1
    if u == 0 and v == 0:
        c_u = c_v = 1. / np.sqrt(2)
    mDCTBaseMatrix = np.zeros((b, b))
    for y in range(b):
        for x in range(b):
            base = c_u * c_v * np.cos(((2 * x + 1) * u * np.pi / 16.)) * np.cos(((2 * y + 1) * v * np.pi / 16.))
            mDCTBaseMatrix[x, y] = base
    return mDCTBaseMatrix

def get_base_image(save=False):
    block_size = 8
    x = np.zeros((block_size ** 2, block_size ** 2))
    if save:
        out_root = '/home/mxt/106-48t/personal_data/mxt/exp_results/ICCV2023/figs/DCTBase'
        os.makedirs(out_root, exist_ok=True)
    for u in range(block_size):
        for v in range(block_size):
            y = calcDCTBase(u, v, block_size) * 255
            x[u * block_size:(u + 1) * block_size, v * block_size:(v + 1) * block_size] = y
            if save:
                out_path = os.path.join(out_root, str(u) + '_' + str(v) + '.png')
                # sns_plot = plt.figure()
                # sns.heatmap(y, cmap='RdBu_r', linewidths=0.0, vmin=0, vmax=255,
                #             xticklabels=False, yticklabels=False, cbar=True)
                # sns_plot.savefig(out_path, dpi=700)
                # plt.close()
                cv2.imwrite(out_path, y)
    if save:
        out_path = os.path.join(out_root, 'global.png')
        cv2.imwrite(out_path, x)
        # sns_plot = plt.figure()
        # sns.heatmap(x, cmap='RdBu_r', linewidths=0.01, vmin=0, vmax=255,
        #             xticklabels='auto', yticklabels='auto', cbar=True)  # Reds_r .invert_yaxis()
        # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
        # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-other_LN_DCT_local' + '.png')
        # sns_plot.savefig(out_path, dpi=700)
        # plt.close()
    return x

class DctMaskEncoding(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, vec_dim=32, mask_size=8):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        # print(vec_dim, mask_size)
        assert vec_dim <= mask_size*mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, dct_x, dim=None):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_x[:, :, xs, ys].contiguous()  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, dim=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device
        if dim is None:
            dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        else:
            dct_vector_coords = self.dct_vector_coords[:dim]
            dct_vectors = dct_vectors[:, :dim].contiguous()

        N, C = dct_vectors.shape[:2]
        dct_trans = torch.zeros([N, C, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, :, xs, ys] = dct_vectors.contiguous()
        # mask_rc = torch_dct.idct_2d(dct_trans, norm='ortho')  # [N, mask_size, mask_size]
        return dct_trans.contiguous()

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs

if __name__=='__main__':
    import torch
    import kornia
    import cv2
    import os
    from time import time
    # print(np.linspace(0, 1, 20))
    # x = np.linspace(0, 1, 20)
    # net = MIMO_DCT2d_fold_branch(window_size=8, stride=1, dct_torch=False, zig_zag=True, zig_out=True).cuda()
    # net2 = MIMO_DCT2d_fold_branch(window_size=8, stride=2, dct_torch=False, zig_zag=True, zig_out=True).cuda()
    # # # net = Window_Local(qkv=False)
    # # x = torch.randn(1, 32, 128, 128)
    x = torch.randn(1, 8, 96, 64).cuda()
    # x = x.cuda()
    # y = net(x, True)
    # print(y.shape)
    # # y = net(y, False)
    # # y = net(y, False)
    # y1 = y[:, :, ::2, ::2]
    # y2 = kornia.geometry.rescale(y, 0.5)
    # y1 = net2(y1, False)
    # y2 = net2(y2, False)
    # print(torch.mean(torch.abs(y1-x)))
    # print(torch.mean(torch.abs(y2-x)))
#     o1 = OrthoT1d(8, dim=1, inference=True).cuda()
#     o2 = OrthoT2d(8, 16).cuda()
#     o3 = Ortho_fold_branch().cuda()
#     o4 = OrthoConv2d(8, 8, kernel_size=7, padding=3, groups=8).cuda()
#     # gy = o3(g, True)
#     # print(gy.shape)
#     gy = o4(g, False)
#     print(gy.shape)
#     print(torch.mean(gy - g))
    # zy = o2(z, False)
    # zy = o2(zy, True)
    # y = o1(x, False)
    # y = o1(y, True)
    # print(torch.mean(y - x))
    # print(torch.mean(zy - z))
    # print(np.tanh(-2), 1-np.tanh(-2))
    start = time()
    y = dct_2d_torch(x )
    y_ = idct_2d_torch(y)
    finish = time()
    print(finish-start)
    # dct_n = DCT2(8)
    # idct_n = IDCT2(8)
    dct_x = DCT2x_torch()
    idct_x = IDCT2x_torch()
    z1 = dct_x(x)
    z1 = idct_x(z1)
    print(torch.mean(z1 - x))
    # y2 = dct_n(x)
    # y2_ = idct_n(y2)
    # y2_x = idct_n(x)
    # # print(y2.max(), y2.min())
    # print(torch.mean(y2 - y))
    # print(torch.mean(y2_ - x))
    # print(torch.mean(y2_x - y_))
    # y = idct_2d_torch(y)
    # y = idct1_spectral2d_torch(x)
    # print(y.max(), y.min())
    # y = dct1_spectral2d_torch(y)
    # y = net(x, True)
    #
    # y = net(y, False)
    # print(torch.mean(y - x))
    # net = Mestormer(dim=8).cuda()
    # # net = Window_Local(qkv=False)
    # x = torch.randn(1,3,128,128)
    # x = x.cuda()

    # dct1d = DCT1d(window_size=64)
    # idct1d = IDCT1d(window_size=64)
    # Conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64, stride=1, padding=1, padding_mode='reflect')
    # Conv2 = Conv2.cuda()
    # window_size = 8
    # pad_size = 7
    # mode = 'reflect' # 'contast' #
    # stride = 1
    # img_size = 256
    # if mode != 'reflect':
    #     fold_params = dict(kernel_size=window_size, dilation=1, padding=pad_size, stride=stride)
    #     fold_params_fold = dict(kernel_size=window_size, dilation=1, padding=pad_size, stride=stride)
    #     # fold_params_fold['stride'] = 2 * stride
    #     fold = nn.Fold(output_size=[img_size//2, img_size//2], **fold_params_fold)
    # else:
    #     fold_params = dict(kernel_size=window_size, dilation=1, padding=0, stride=stride)
    #     fold_params_fold = dict(kernel_size=window_size, dilation=1, padding=0, stride=stride)
    #     output_size = (img_size+pad_size) # // 2
    #     fold = nn.Fold(output_size=[output_size, output_size], **fold_params_fold)
    #
    # unfold = nn.Unfold(**fold_params)
    # # z = torch.randn(3, 1, 128, 128)
    # z = cv2.imread('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/target_crops/0.png')
    # z = kornia.image_to_tensor(z, keepdim=False)
    # z_ = kornia.color.rgb_to_grayscale(z/255.)
    # z = z_.cuda()
    # H, W = z.shape[-2:]
    # if mode == 'reflect':
    #     z = F.pad(z, (pad_size, 0, pad_size, 0), 'reflect')
    #
    # input_ones = torch.ones_like(z)
    # divisor = fold(unfold(input_ones))
    # y = unfold(z)
    # if mode != 'reflect':
    #     h, w = (H + 2 * pad_size - window_size) // stride + 1, (
    #                 W + 2 * pad_size - window_size) // stride + 1
    # else:
    #     h, w = (H + pad_size - window_size) // stride + 1, (
    #             W + pad_size - window_size) // stride + 1
    #
    # y = dct1d(y)
    # y = rearrange(y, 'b c (h w) -> b c h w', h=h, w=w)
    # y = Conv2(y)
    # y = torch.relu(y)
    # print(y.shape)
    # y_out = kornia.tensor_to_image(y.cpu())
    # out_dir = '/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/freq/dct_fold_sharp_ref_s2'
    # os.makedirs(out_dir, exist_ok=True)
    # for i in range(y_out.shape[-1]):
    #     cv2.imwrite(os.path.join(out_dir, str(i)+'.jpg'), y_out[:,:,i]*255)
    # y = rearrange(y, 'b c h w -> b c (h w)', h=h, w=w)
    # y = idct1d(y)
    # y = fold(y) / divisor
    # print(y.shape)
    # print(torch.sum(y-z))
    # if mode == 'reflect':
    #     y = y[:, :, pad_size:, pad_size:]
    # print(torch.sum(y - z_.cuda()))
    # y = kornia.tensor_to_image(y.cpu())
    # cv2.imwrite(os.path.join(out_dir, 'y.jpg'), y * 255)
    # y = torch.randn(8, 32, 8, 8)
    # y1 = rearrange(y, 'b (head c) (k1 h) (k2 w) -> b head (h w) c (k1 k2)', head=4, k1=8, k2=8)
    # y1 = torch.nn.functional.normalize(y1, dim=-2)
    # y1 = rearrange(y1, 'b head (h w) c (k1 k2) -> b head c (k1 h k2 w)', head=4, h=1, w=1, k1=8, k2=8)
    # y2 = rearrange(y, 'b (head c) h w -> b head c (h w)', head=4)
    # y2 = torch.nn.functional.normalize(y2, dim=-2)
    # print(torch.mean(y1-y2))
    # y, batch_list = window_partitionxy(x, 32, start=[16, 16])
    # out = window_reversexy(y, 32, 128, 128, batch_list, start=[16, 16])
    # print(torch.mean(out-x))
    # a = torch.fft.hfft2(x)
    # print(a.shape)
    # z = torch.fft.ihfft2(a)
    # print(z.shape)
    # print(torch.mean(z.real-x))
