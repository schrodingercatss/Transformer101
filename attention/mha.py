import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _transpose_score(self, x): # x shape: [B, seq_len, d_model]
        new_shape = x.size()[:-1] + (self.num_heads, self.d_k)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        # Linear projections for Q, K, V
        Q = self._transpose_score(self.q_proj(q))
        K = self._transpose_score(self.k_proj(k))
        V = self._transpose_score(self.v_proj(v))

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to V
        context = attn_weights @ V # shape: [B, num_heads, seq_len, d_k]

        # reshape back to [B, seq_len, num_heads * d_k]
        context = context.permute(0, 2, 1, 3).contiguous().view(*q.size())
        output = self.o_proj(context)
        return output