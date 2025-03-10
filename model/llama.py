import torch.nn as nn
import torch.nn.functional as F
from attention.llama_attn import LlamaAttention


class LlamaMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * F.silu(self.up_proj(x)))
        

class LlamaDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.self_attn = LlamaAttention(d_model, num_heads, max_seq_len, dropout)
        self.ffn = LlamaMLP(d_model, 4 * d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # first normalization, then self-attention
        x = self.norm1(x)
        x = x + self.dropout(self.self_attn(x, x, x, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x        


class LlamaDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_seq_len, dropout=0.0):
        super().__init__()
    
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(d_model, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = LlamaDecoder(d_model, num_heads, num_layers, max_seq_len, dropout)


        self.norm = nn.RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, masks=None):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.decoder(x, masks)
        x = self.norm(x)
        output = self.fc(x)
        return output


# class RMSNorm(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-8):
#         super().__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(normalized_shape))
        
#     def forward(self, x):
#         rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
#         x_normalized = x / (rms + self.eps)
#         return self.alpha * x_normalized
    
