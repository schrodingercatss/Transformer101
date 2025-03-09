import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup


def create_causal_mask(seq_len):
    """Create a causal attention mask to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0

import math
import wandb

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def train_transformer(model, train_dataloader, val_dataloader, max_steps, lr, device, eval_interval=1000):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # 默认 reduction='mean'
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=max_steps)
    
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
            outputs = model(inputs, causal_mask)
            
            actv_norm = outputs.detach().norm(2).item()

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()

            # grad norm
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.detach().norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            # grad norm clip
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
            })

            # 每隔一定步数进行验证
            if global_step % eval_interval == 0:
                model.eval()
                total_val_loss = 0.0
                total_tokens = 0
                with torch.no_grad():
                    for val_inputs, val_targets in val_dataloader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        seq_len_val = val_inputs.size(1)
                        causal_mask_val = create_causal_mask(seq_len_val).to(device)
                        outputs_val = model(val_inputs, causal_mask_val)
                        loss_val = criterion(outputs_val.view(-1, outputs_val.size(-1)), val_targets.view(-1))
                        
                        # 计算当前 batch 的 token 数
                        # 如果 val_targets 的 shape 是 [Batch, seq_len, dim]，则：
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
            if global_step % 5000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                }, f'transformer_step_{global_step}.pth')
                
    torch.save(model.state_dict(), f'transformer_final.pth')


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
        seq_len =  input_tensor.size(1)
        causal_mask = create_causal_mask(seq_len).to(device)

        with torch.no_grad():
            output = model(input_tensor, causal_mask)
        
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