import torch
import torch.nn as nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        num_embed=32,
        block_size=8,
        num_heads=4,
        num_layers=4,
        dropout=0.2
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_embed = num_embed
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
        # each position from 0 to block_size-1 will get its embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
        self.blocks = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=self.num_embed,
                    nhead=self.num_heads,
                    dim_feedforward=4 * self.num_embed,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True
                ) for _ in range(self.num_layers)]
        )
        # we add the layer norm before the Linear layer
        self.ln_f = nn.LayerNorm(self.num_embed)
        self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B,T) tensor of integers
        # the token_emb is (B, T, C), C = NUM_EMBED
        token_emb = self.token_embedding_table(idx)
        # (T, C)
        posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

        x = token_emb + posit_emb
        # apply one head of self-attention
        x = self.blocks(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        # compute the loss
        if targets != None:
            # cross_entropy accepts inputs in a (batch_size, num_classes)
            # so we need to reformat our logits dimensions to
            # (batch_size * time, dim_vocabulary), time = block_size
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            targets = torch.reshape(targets, (B * T,))
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return id
