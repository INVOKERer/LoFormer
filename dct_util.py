import torch
from torch import nn as nn
import numpy as np


def dct(x, norm=None):
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
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i # [:, :N // 2 + 1]

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    # print(V)
    return V

def idct(X, norm=None):
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
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

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


def dct_2d_torch(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d_torch(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
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


def dct2d(feature, dctMat):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat)# dctMat @ feature
    # print(dctMat.shape, feature.shape)
    # feature = feature @ dctMat.T
    # print(feature.transpose(-1, -2).shape, dctMat.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat) # dctMat @ feature.transpose(-1, -2) # @ dctMat.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)


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
    def forward(self, x):
        # print(x.shape)
        x = dct_2d_torch(x, self.norm)
        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops
class IDCT2_torch(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2_torch, self).__init__()
        self.norm = norm
    def forward(self, x):
        # print(x.shape)
        x = idct_2d_torch(x, self.norm)
        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops

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
        flops += C * H * W * H * W
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
        flops += C * H * W * H * W
        return flops

if __name__=='__main__':
    x = torch.randn(1, 8, 64, 64).cuda()
    model1 = DCT2x().cuda()
    model2 = DCT2_torch().cuda()

    y = model1(x)
    y_ = model2(x)

    print(torch.sum(y-y_))
