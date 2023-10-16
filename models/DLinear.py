import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, **kwargs):
        super(DLinear, self).__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self.T_in, self.H, self.W, self.C_in = self.input_shape
        self.T_out, self.H, self.W, self.C_out = self.target_shape

        self.padding_H = self.H % self.kernel_size
        self.padding_W = self.W % self.kernel_size

        self.unfold = nn.Unfold(kernel_size=[self.kernel_size] * 2,
                                padding=(self.padding_H, self.padding_W),
                                stride=(self.kernel_size // 2, self.kernel_size // 2))
        self.fold = nn.Fold(output_size=(self.H, self.W),
                            kernel_size=[self.kernel_size] * 2,
                            padding=(self.padding_H, self.padding_W),
                            stride=(self.kernel_size // 2, self.kernel_size // 2))

        self.w_s= {}
        for _ in range(self.C_out * self.T_out):
            self.w_s.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })

        self.w_s = nn.ParameterDict(self.w_s)

        self.w_t= {}
        for _ in range(self.C_out * self.T_out):
            self.w_t.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })

        self.w_t = nn.ParameterDict(self.w_t)


        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x):
        B, T_in, H, W, C_in = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)
        x = self.unfold(x) # [B, ..., L]

        seasonal_init, trend_init = self.decompsition(x)
        #seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        L = x.shape[-1]
        #x = x.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]
        seasonal_init = seasonal_init.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]
        trend_init = trend_init.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]

        xx = []
        yy = []
        for i in range(self.C_out * self.T_out):
            w_s, bias_s = self.w_s[f"w_{i}"], self.w_s[f"bias_{i}"]
            xx.append((seasonal_init * w_s).sum(dim=(2, 3)) + bias_s)  # [B, L, k, k]

            w_t, bias_t = self.w_t[f"w_{i}"], self.w_t[f"bias_{i}"]
            yy.append((trend_init * w_t).sum(dim=(2, 3)) + bias_t)  # [B, L, k, k]



        seasonal_init = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
        seasonal_init = seasonal_init.permute(0, 1, 3, 4, 2).reshape(B, -1, L)
        trend_init = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
        trend_init = trend_init.permute(0, 1, 3, 4, 2).reshape(B, -1, L)

        x = seasonal_init + trend_init
        x = self.fold(x) # [B, C_out * T_out, H, W]
        x = x.reshape(B, self.T_out, self.C_out, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x # [B, T, H, W, С_out]
