import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from attention.llama_attn import LlamaAttention
from model.llama import LlamaMLP
from model.MoE import DeepSeekMoE

class LlamaDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0, use_moe=False, moe_params=None):
        """
        use_moe: 是否使用 MoE 版 FFN
        moe_params: 当 use_moe=True 时，传入一个字典，包含如下参数：
            - hidden_dim: 专家内部隐藏层维度（通常 4*d_model）
            - num_experts: 专家数量
            - top_k: 每个 token 路由到的专家数
            - capacity_factor: 专家容量因子
        """
        super().__init__()
        self.self_attn = LlamaAttention(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_moe = use_moe
        if use_moe:
            # 使用 MoE 版 FFN
            self.ffn = DeepSeekMoE(
                d_model, 
                moe_params.get("hidden_dim", 4 * d_model), 
                moe_params.get("num_experts", 4),
                moe_params.get("top_k", 2),
                moe_params.get("capacity_factor", 1.25),
                moe_params.get("alpha", 0.1)
            )
        else:
            # 保持原始 FFN 实现
            self.ffn = LlamaMLP(d_model, 4 * d_model)
    def forward(self, x, mask=None):
        # self-attention 部分
        x = self.norm1(x)
        x = x + self.dropout(self.self_attn(x, x, x, mask))
        # FFN 部分
        x_norm = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
            aux_loss = None
        x = x + self.dropout(ffn_out)
        return x, aux_loss


class LlamaDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_seq_len, dropout=0.0, use_moe=False, moe_params=None):
        super().__init__()
        self.layers = nn.ModuleList([
            LlamaDecoderBlock(d_model, num_heads, max_seq_len, dropout, use_moe, moe_params)
            for _ in range(num_layers)
        ])
    def forward(self, x, tgt_mask=None):
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, tgt_mask)
            if aux_loss is not None:
                total_aux_loss += aux_loss
        return x, total_aux_loss
    
class LlamaModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, dropout=0.0, use_moe=False, moe_params=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = LlamaDecoder(d_model, num_heads, num_layers, max_seq_len, dropout, use_moe, moe_params)
        self.norm = nn.RMSNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, masks=None):
        x = self.embedding(x)
        x = self.dropout(x)
        x, aux_loss = self.decoder(x, masks)
        x = self.norm(x)
        output = self.fc(x)
        return output, aux_loss