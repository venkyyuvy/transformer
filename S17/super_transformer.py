
import torch
from torch import nn
from vit import PatchEmbedding, patch_size
from bert import PositionalEmbedding

patch_size=16


class SuperTransformer(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    def __init__(self,
                 model='vit',
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1, 
                 embedding_dropout:float=0.1,
                 num_classes:int=1000
                 ):
        super().__init__()
        
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches+1, embedding_dim),
            requires_grad=True)
        
        if model == "vit":
            assert img_size % patch_size == 0,\
            f"Image size must be divisible by patch size, "+\
            f"image size: {img_size}, patch size: {patch_size}."
            self.num_patches = (img_size * img_size) // patch_size**2
                     
            self.class_embedding = nn.Parameter(
                data=torch.randn(1, 1, embedding_dim),
                requires_grad=True)
        
            self.patch_embedding = PatchEmbedding(
                in_channels=in_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        
        self.transformer_encoder = nn.Sequential(
            *[
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=mlp_size,
                dropout=mlp_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])

        return x       

