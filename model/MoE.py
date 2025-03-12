import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.down_proj(self.gate_proj(x) * F.silu(self.up_proj(x)))
    
class MoEGate(nn.Module):
    def __init__(self, d_model, num_experts, top_k, capacity_factor, alpha, norm_topk_prob=False):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.alpha = alpha
        self.norm_topk_prob = norm_topk_prob

        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)

        # select top_k experts
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1) # topk_weights shape: [N, top_k], topk_indices shape: [N, top_k]
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-9
            topk_weights = topk_weights / denominator

        # 计算 load，用onehot表示，每个token在topk内可能出现多次
        one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float() # shape: [N, top_k, num_experts]
        load = one_hot.view(-1, self.num_experts).sum(dim=0) # shape: [num_experts, ]

        # 计算每个专家的容量，每个token都会被分配到topk个专家
        n_tokens = x.size(0)
        capacity = self.capacity_factor * (n_tokens * self.top_k) / self.num_experts

        # 如果某个专家的容量超过了上限，则计算惩罚
        capacity_penalty = (F.relu(load - capacity).sum() / self.num_experts) / (batch_size * seq_len)
        aux_loss = self.alpha * capacity_penalty

        # 恢复topk_idx 和 topk_weights 的形状
        topk_indices = topk_indices.view(batch_size, seq_len, self.top_k)
        topk_weights = topk_weights.view(batch_size, seq_len, self.top_k)
        return topk_indices, topk_weights, aux_loss



class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, top_k, capacity_factor, alpha, open_shared_expert=False):
        super().__init__()
        self.input_dim = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.hidden_dim = hidden_dim

        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim) for _ in range(num_experts)
        ])

        self.shared_expert = Expert(d_model, hidden_dim) if open_shared_expert else None

        self.gate = MoEGate(d_model, num_experts, top_k, capacity_factor, alpha=alpha, norm_topk_prob=True)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape


        topk_indices, topk_weights, aux_loss = self.gate(x)
        x_flat = x.view(-1, d_model)
        x_repeated = x_flat.repeat_interleave(self.top_k, dim=0) # shape: [N * top_k, d_model]

        # 初始化 expert_outputs
        expert_outputs = torch.empty_like(x_repeated)
        flat_topk_indices = topk_indices.view(-1) # shape: [N * top_k, ]

        # 对于每个专家处理被分配的 token
        for expert_id, expert in enumerate(self.experts):
            mask = (flat_topk_indices == expert_id)
            if mask.sum() == 0:
                continue
            expert_outputs[mask] = expert(x_repeated[mask])
        
        expert_outputs = expert_outputs.view(batch_size, seq_len, self.top_k, d_model)
        expert_outputs = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        if self.shared_expert is not None:
            expert_outputs += self.shared_expert(x)
        return expert_outputs, aux_loss