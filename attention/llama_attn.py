import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from pos_encoding.RotaryPE import RotaryPE


class LlamaAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryPE(self.d_k, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def _transpose_score(self, x): # x shape: [B, seq_len, d_model]
        new_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        B, seq_len, _ = q.size()
        Q = self._transpose_score(self.q_proj(q))
        K = self._transpose_score(self.k_proj(k))
        V = self._transpose_score(self.v_proj(v))

        Q = self.rotary_emb(Q.reshape(-1, seq_len, self.d_k)).reshape(B, self.num_heads, seq_len, self.d_k)
        K = self.rotary_emb(K.reshape(-1, seq_len, self.d_k)).reshape(B, self.num_heads, seq_len, self.d_k)
        V = self.rotary_emb(V.reshape(-1, seq_len, self.d_k)).reshape(B, self.num_heads, seq_len, self.d_k)

        score = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(score, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ V
        context = context.permute(0, 2, 1, 3).contiguous().view(B, seq_len, self.d_model)
        output = self.o_proj(context)
        return output