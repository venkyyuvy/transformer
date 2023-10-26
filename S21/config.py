import torch
from dataclasses import dataclass

@dataclass
class BigramConfig:
    batch_size = 32  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?
    max_iters = 3000
    eval_interval = 300 
    learning_rate = 1e-2 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200

@dataclass
class GPTConfig:
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

bigram_config = BigramConfig()
gpt_config = GPTConfig()
