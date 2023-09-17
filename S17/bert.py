import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        pe.requires_grad = False
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x.size(1) = seq_len
        return self.pe[:,:x.size(1)] 
    

class Transformer(nn.Module):
    def __init__(
            self,
            n_code,
            n_heads,
            embed_size,
            inner_ff_size,
            n_embeddings,
            seq_len,
            dropout=.1):
        super().__init__()
        
        #model input
        self.embeddings = nn.Embedding(n_embeddings, embed_size)
        self.pe = PositionalEmbedding(embed_size, seq_len)
        
        self.encoders = nn.ModuleList(
        [
                nn.TransformerEncoderLayer(
                    d_model=embed_size,
                    nhead=n_heads,
                    dim_feedforward=inner_ff_size,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True
                ) for _ in range(n_code)]
        )
        
        #language model
        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings, bias=False)
                
    
    def forward(self, x):
        x = self.embeddings(x)
        x = x + self.pe(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

