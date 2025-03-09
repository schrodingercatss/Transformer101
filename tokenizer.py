class SimpleTokenizer:
    def __init__(self, texts=None):
        self.char_to_idx = {}
        self.idx_to_char = {}

        if texts:
            self.fit(texts)
    
    def fit(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
            
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    @property
    def vocab_size(self):
        return len(self.char_to_idx)