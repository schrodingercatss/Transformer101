import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()

        # Create a matrix for positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) # shape: [1, max_seq_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]