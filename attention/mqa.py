import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        self.k_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _transpose_score(self, x): # x shape: [B, seq_len, d_model]
        new_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        Q = self.q_proj(q) # shape: [B, seq_len, d_model]
        K = self.k_proj(k)
        V = self.v_proj(v)
        
        # split into multiple heads
        Q = self._transpose_score(Q)
        K = K.unsqueeze(1) # shape: [B, 1, seq_len, d_k]
        V = V.unsqueeze(1) # shape: [B, 1, seq_len, d_k]

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)

        context = attn_weights @ V
        context = context.permute(0, 2, 1, 3).contiguous().view(*q.size())

        output = self.o_proj(context)
        return output