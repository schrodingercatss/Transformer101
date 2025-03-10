import torch.nn as nn
import torch.nn.functional as F
import math


class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_groups

        self.q_proj = nn.Linear(d_model, d_model)

        # The key and value projections are now split into num_groups groups
        self.k_proj = nn.Linear(d_model, num_groups * self.d_k)
        self.v_proj = nn.Linear(d_model, num_groups * self.d_k)
        
        self.o_proj = nn.Linear(d_model, d_model)

    def _transpose_score(self, x): # x shape: [B, seq_len, d_model]
        new_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _reshape_kv(self, x):
        new_shape = x.size()[:-1] + (self.num_groups, self.d_k)
        x = x.view(*new_shape)
        x = x.permute(0, 2, 1, 3)
        return x.repeat_interleave(self.group_size, dim=1) # every key and value is repeated group_size times, [B, num_heads, seq_len, d_k]

    def forward(self, q, k, v, mask=None):
        Q = self.q_proj(q)
        Q = self._transpose_score(Q)

        K = self.k_proj(k)
        K = self._reshape_kv(K) # shape: [B, num_heads, seq_len, d_k]
        V = self.v_proj(v)
        V = self._reshape_kv(V)

        score = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(score, dim=-1)
        context = attn_weights @ V

        context = context.permute(0, 2, 1, 3).contiguous().view(*q.size())
        output = self.o_proj(context)
        return output