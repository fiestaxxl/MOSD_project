import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, **kwargs):
        super(NLinear, self).__init__()
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

        self.w = {}
        for _ in range(self.C_out * self.T_out):
            self.w.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })

        self.w = nn.ParameterDict(self.w)

    def forward(self, x):
        B, T_in, H, W, C_in = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)
        x = self.unfold(x) # [B, ..., L]
        seq = x[:,-1:,:]
        x = x - seq
        L = x.shape[-1]
        x = x.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]

        xx = []
        for i in range(self.C_out * self.T_out):
            w, bias = self.w[f"w_{i}"], self.w[f"bias_{i}"]
            xx.append((x * w).sum(dim=(2, 3)) + bias)  # [B, L, k, k]

        x = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, L)
        x = x + seq
        x = self.fold(x) # [B, C_out * T_out, H, W]
        x = x.reshape(B, self.T_out, self.C_out, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x # [B, T, H, W, С_out]