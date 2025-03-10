import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class RotaryPE(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"

        inv_freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positions = torch.arange(0, max_seq_len, dtype=torch.float)

        freq = torch.einsum('i,j->ij', positions, inv_freq)
        sin = torch.sin(freq)
        cos = torch.cos(freq)
        sin = torch.cat([sin, sin], dim=-1)
        cos = torch.cat([cos, cos], dim=-1)

        self.register_buffer('sin', sin.unsqueeze(0)) # shape: [1, max_seq_len, d_model]
        self.register_buffer('cos', cos.unsqueeze(0))

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        sin = self.sin[:, :x.size(1)]
        cos = self.cos[:, :x.size(1)]

        """
        x: [x1, x2]
        x * cos = [x1 * cos, x2 * cos]
        retate_half(x) * sin = [-x2 * sin, x1 * sin]
        output: [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        """
        x_rotated = (x * cos) + self.rotate_half(x) * sin
        return x_rotated