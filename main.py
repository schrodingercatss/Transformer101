import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from torch.utils.data import DataLoader
from text_dataset import TinyStoriesDataset
from transformer import Transformer
from training import train_transformer, generate_text
from transformers import AutoTokenizer
import wandb


# Hyperparameters
seq_length = 512
batch_size = 64
d_model = 512
num_heads = 8
d_ff = d_model * 4
num_layers = 6
dropout = 0.1
# num_epochs = 20
max_steps = 100000
lr = 1e-3



# 初始化 wandb
wandb.init(project='tiny_transformer', config={
    'seq_length': seq_length,
    'batch_size': batch_size,
    'd_model': d_model,
    'num_heads': num_heads,
    'd_ff': d_ff,
    'num_layers': num_layers,
    'dropout': dropout,
    'max_steps': max_steps,
    'lr': lr,
})


if __name__ == '__main__':

    raw_dataset = load_dataset("roneneldan/TinyStories")

    train_texts = [sample["text"] for sample in raw_dataset["train"]]
    val_texts = [sample["text"] for sample in raw_dataset["validation"]]


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    train_dataset = TinyStoriesDataset(train_texts, tokenizer, seq_length)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer, seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)



    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout
    )

    train_transformer(model, train_dataloader, val_dataloader, max_steps, lr, device)
    print("Model saved to transformer.pth")

    # Generate text
    start_text = "Once upon a time, there was a little prince."
    generated_text = generate_text(model, tokenizer, start_text, max_length=128, temperature=0.1, device=device)
    print(f"Generated text:\n{generated_text}")