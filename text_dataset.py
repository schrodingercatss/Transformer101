import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset

class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        """
        texts: 文本列表，每个元素为一个字符串
        tokenizer: Huggingface tokenizer，例如使用 "bert-base-cased"
        max_length: 最大长度（包含特殊 token），超过部分会截断，不足部分则填充
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        
        # 截断：如果长度超过 max_length，则保留前 max_length 个 token
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            # 填充：如果长度不足，则填充 pad token
            pad_id = self.tokenizer.pad_token_id
            encoded = encoded + [pad_id] * (self.max_length - len(encoded))
        
        # 对于自回归生成任务，输入为序列的前 n-1 个 token，目标为后 n-1 个 token
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
        target_ids = torch.tensor(encoded[1:], dtype=torch.long)
        
        return input_ids, target_ids
