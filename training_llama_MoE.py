import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import wandb

def create_causal_mask(seq_len):
    """Create a causal attention mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def train_llama(model, train_dataloader, val_dataloader, max_steps, lr, device, eval_interval=2000, aux_loss_weight=1.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # 默认 reduction='mean'
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000, num_training_steps=max_steps)
    
    global_step = 0

    while global_step < max_steps:
        model.train()
        for inputs, targets in train_dataloader:
            if global_step >= max_steps:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            seq_len = inputs.size(1)
            causal_mask = create_causal_mask(seq_len).to(device)

            optimizer.zero_grad()
            # 修改处：解包模型输出，获得 output 与 aux_loss
            output, aux_loss = model(inputs, causal_mask)
            actv_norm = output.detach().norm(2).item()

            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            model_loss = loss.item()
            if aux_loss is not None:
                loss = loss + aux_loss_weight * aux_loss

            loss.backward()

            # 计算梯度范数
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.detach().norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            # 若梯度范数过大，则跳过该 batch
            if math.isnan(total_norm) or (global_step > 3000 and total_norm > 3.0):
                print(f"Step {global_step} | Skipping batch due to gradient norm {total_norm:.4f}")
                continue

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'global_step': global_step,
                'lr': current_lr,
                'train_loss': loss.item(),
                'l2_grad_norm': total_norm,
                'actv_norm': actv_norm,
                'model_loss': model_loss,
                'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
            })

            # 验证
            if global_step % eval_interval == 0:
                model.eval()
                total_val_loss = 0.0
                total_tokens = 0
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        seq_len_val = val_inputs.size(1)
                        causal_mask_val = create_causal_mask(seq_len_val).to(device)
                        # 解包验证输出（aux_loss 忽略）
                        outputs_val, _ = model(val_inputs, causal_mask_val)
                        loss_val = criterion(outputs_val.view(-1, outputs_val.size(-1)), val_targets.view(-1))
                        
                        if val_targets.dim() == 3:
                            batch_tokens = val_targets.size(0) * val_targets.size(1)
                        else:
                            batch_tokens = val_targets.numel()
                        
                        total_val_loss += loss_val.item() * batch_tokens
                        total_tokens += batch_tokens

                avg_val_loss = total_val_loss / total_tokens
                val_ppl = math.exp(avg_val_loss)
                wandb.log({
                    'global_step': global_step,
                    'val_loss': avg_val_loss,
                    'val_ppl': val_ppl,
                })
                print(f"Step {global_step} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.4f}")
                model.train()

            if global_step % 1000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                }, f'llama_step_{global_step}.pth')
                
    torch.save(model.state_dict(), f'llama_final.pth')

def generate_text(model, tokenizer, start_text, max_length, temperature=1.0, device='cpu'):
    model.eval()
    input_seq = tokenizer.encode(
        start_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

    for _ in range(max_length):
        seq_len = input_tensor.size(1)
        causal_mask = create_causal_mask(seq_len).to(device)
        # 解包生成时的输出
        output, _ = model(input_tensor, causal_mask)
        
        next_token_logits = output[0, -1, :] / temperature
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1).item()
        input_tensor = torch.cat([
            input_tensor,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)
    
    generate_tokens = input_tensor[0].tolist()
    generate_text = tokenizer.decode(generate_tokens)
    return generate_text
