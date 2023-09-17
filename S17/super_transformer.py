
import torch
from torch import nn
import pytorch_lightning as pl
from vit import PatchEmbedding
from bert import PositionalEmbedding


class SuperTransformer(pl.LightningModule):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    def __init__(self,
                 num_classes:int,
                 model='vit',
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 mlp_dropout:float=0.1, 
                 embedding_dropout:float=0.1,
                 n_embeddings: int= 128,
                 seq_len: int= 20,
                 index: int=0
                 ):
        super().__init__()
        
        self.model = model
        self.index = index
        
        if self.model == "bert":
            self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
            self.position_embedding = PositionalEmbedding(
                embedding_dim, seq_len)
        if self.model == "vit":
            assert img_size % patch_size == 0,\
            f"Image size must be divisible by patch size, "+\
            f"image size: {img_size}, patch size: {patch_size}."
            self.num_patches = (img_size * img_size) // patch_size**2
                     
            self.embeddings = PatchEmbedding(
                in_channels=in_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim)
            self.position_embedding = nn.Parameter(
                data=torch.randn(1, self.num_patches+1, embedding_dim),
                requires_grad=True)
            self.class_embedding = nn.Parameter(
                data=torch.randn(1, 1, embedding_dim),
                requires_grad=True)
            seq_len = self.num_patches + 1

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

        if self.model == "bert":
            self.word_predictor = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(
                    embedding_dim, num_classes, bias=False)
            )
        if self.model == "vit":
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=embedding_dim),
                nn.Linear(in_features=embedding_dim, 
                          out_features=num_classes)
            )
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        x = self.embeddings(x)
        if self.model == "vit":
            class_token = self.class_embedding.expand(batch_size, -1, -1)
            x = torch.cat((class_token, x), dim=1)
            x = self.position_embedding + x
        else:
            x = self.position_embedding(x) + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        if self.model == "vit":
            x = self.classifier(x[:, 0])
        elif self.model == "bert":
            x = self.word_predictor(x)

        return x       

    def training_step(self, batch, batch_idx):
        if self.model == "vit":
            X, y = batch
            y_pred = self(X)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(y_pred, y)
        elif self.model == "bert":
            masked_input = batch['input']
            masked_target = batch['target']
            
            masked_input = masked_input
            masked_target = masked_target
            output = self(masked_input)
            
            #compute the cross entropy loss 
            output_v = output.view(-1,output.shape[-1])
            target_v = masked_target.view(-1,1).squeeze()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.index)
            loss = loss_fn(output_v, target_v)
        return loss


    def configure_optimizers(self):
        if self.model == "vit":
            lr = 3e-3
            weight_decay = 0.3
        elif self.model == "bert":
            lr = 1e-4 
            weight_decay = 1e-4
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        ) 
        return optimizer
