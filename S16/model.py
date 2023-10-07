import torch
import torch.nn as nn
import math
from torch import optim 
 
import torchmetrics
import torchmetrics.text
from torch.utils.tensorboard import SummaryWriter 

from pytorch_lightning import LightningModule
from dataset import causal_mask 
from config import get_config, get_weights_file_path 

class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn. Parameter(torch.ones (1)) #alpha is a learnable parameter
        self.bias = nn. Parameter(torch.zeros(1)) #·bias is a learnable parameter
        
    def forward(self,x):
        #x: (batch, seq_len, hidden_size)
        #Keep the dimension for broadcasting
        mean = x.mean (dim = -1, keepdim = True) # (batch, seq_len, 1)
        #Keep the dimension for broadcasting
        std = x.std (dim = -1, keepdim = True) # (batch, seq_len, ∙1)
        #eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn. Dropout (dropout)
        self.linear_2= nn.Linear(d_ff, d_model) # w2 and b2
        
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout (torch.relu(self.linear_1(x))))
    
    
class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size = vocab_size
        self.embedding = nn. Embedding (vocab_size, d_model)
        
    def forward(self,x):
        #· (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x)* math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module): 
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None: 
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # Create a matrix of shape (seq_len, d_model) 
        pe = torch.zeros(seq_len, d_model) # Create a vector of shape (seq_len) 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1) 
        # Create a vector of shape (d_model) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices 
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model)) 
        # Apply cosine to odd indices 
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) 
        # Register the positional encoding as a buffer 
        self.register_buffer('pe', pe) 
        
    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model) 
        return self.dropout(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None: 
        super().__init__() 
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer): 
        return x + self.dropout(sublayer(self.norm(x))) 
    
    
class MultiHeadAttentionBlock(nn.Module): 
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None: 
        super().__init__() 
        self.d_model = d_model  # Embedding vector size 
        self.h = h # Number of heads 
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq 
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk 
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv 
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo 
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            _MASKING_VALUE = -1e9 if attention_scores.dtype == torch.float32 else -1e+4
            attention_scores.masked_fill_(mask == 0, _MASKING_VALUE)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply soft
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,  dropout: float) -> None : 
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
class DecoderBlock(nn.Module):
    def __init__(
            self,
            self_attention_block: MultiHeadAttentionBlock,
            cross_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: float 
        ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(
                x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](
            x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) -> torch.Tensor:
        #- (batch, seq_len, d_model) ---> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class LitTransformer(LightningModule):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embed: InputEmbeddings,
            tgt_embed: InputEmbeddings,
            src_pos: PositionalEncoding,
            tgt_pos: PositionalEncoding,
            projection_layer: ProjectionLayer
        ) -> None:
        super().__init__()
        self.config = get_config()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
        self.val_count = 0 
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 
        self.val_num_examples = 2
        
    def encode(self, src, src_mask):
        #- (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor
    ):
        #- (batch, -seq_len, -d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(
            tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, -seq_len, -vocab_size)
        return self.projection_layer(x)

    def training_step(self, batch, batch_idx):
        x = batch
        encoder_input = x['encoder_input'].to(self.device)
        decoder_input = x['decoder_input'].to(self.device)
        encoder_mask = x['encoder_mask'].to(self.device)
        decoder_mask = x['decoder_mask'].to(self.device)
        
        # Run the tensors through the encoder, decoder and the projection layer 
        encoder_output = self.encode(
            encoder_input, encoder_mask) # (B, seq_len, d_model) 
        decoder_output = self.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask) 
        proj_output = self.project(decoder_output) # (B, seq_len, vocab_size) 
        # Compare the output with the label 
        label = batch['label'] # (B, seg_len)


        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.trainer.datamodule.tokenizer_tgt.token_to_id('[PAD]'), 
            label_smoothing=0.1
        )
        # Compute the loss using a simple cross entropy 
        loss = self.loss_fn(
            proj_output.view(-1, self.trainer.datamodule.tokenizer_tgt.get_vocab_size()),
            label.view(-1)
        ) 
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("loss", loss.item(), prog_bar=True) 
        #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) 
        
        

        return loss
    
    # def training_step_end(self, batch, ):        
    #     # Your train step end logic goes here
    #     scale = self.scaler.get_scale()
    #     self.scaler.update()
    #     skip_lr_sched = (scale > self.sceler.get_scale())
    #     if not skip_lr_sched:
    #         self.scheduler.step()

    def validation_step(self, batch, batch_idx): 
        max_len = self.config['seq_len'] 
        
        if self.val_count == self.val_num_examples:             
            return 
        
        self.val_count += 1 
        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]

        # check that the batch size is 1 
        assert encoder_input.size(0) == 1, \
        "Batch  size must be 1 for val"

        model_out = self.greedy_decode(
            self,
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            max_len,
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0] 
        model_out_text = self.trainer.datamodule.tokenizer_tgt.decode(
            model_out.detach().cpu().numpy()) 

        self.val_source_texts.append(source_text) 
        self.val_expected.append(target_text) 
        self.val_predicted.append(model_out_text) 

        # Print the source, target and model output             
        print(f"{f'SOURCE: ':>12}{source_text}") 
        print(f"{f'TARGET: ':>12}{target_text}")
        print(f"{f'PREDICTED: ':>12}{model_out_text}") 
            
    def on_validation_epoch_end(self):
        # Evaluate the character error rate 
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate() 
        cer = metric(self.val_predicted, self.val_expected) 
        self.log("val_cer", cer, prog_bar=True)
        print(f'Val CER at end of epoch {self.trainer.current_epoch} = {cer}')

        # Compute the word error rate 
        metric = torchmetrics.text.WordErrorRate() 
        wer = metric(self.val_predicted, self.val_expected) 
        self.log("val_wer", wer, prog_bar=True)
        print(f'Val WER at end of epoch {self.trainer.current_epoch} = {wer}')

        # Compute the BLEU metric 
        metric = torchmetrics.text.BLEUScore() 
        bleu = metric(self.val_predicted, self.val_expected) 
        self.log("val_bleu", bleu, prog_bar=True)
        print(f'Val BLEU at end of epoch {self.trainer.current_epoch} = {bleu}')
            
        self.val_count = 0
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 

    def test_step(self, batch, batch_idx):
        pass
    
      
    def on_train_epoch_end(self):
        
        # Save the model at the end of every 5th epoch - to save memory
        curr_epoch = self.trainer.current_epoch + 1
        if curr_epoch % 5 == 0:
            model_filename = get_weights_file_path(
                self.config, f"{self.trainer.current_epoch:02d}") 
            torch.save(
                { 
                    'epoch': self.trainer.current_epoch, 
                    'model_state_dict': self.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(), 
                    'global_step': self.trainer.global_step
                }, 
                model_filename
            ) 
            
            
    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len): 
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step 
        encoder_output = model.encode(source, source_mask) 

        # Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(self.device) 

        while True: 
            if decoder_input.size(1) == max_len:  
                break 

            # build mask for target 
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(self.device) 

            # calculate output 
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) 

            # get next token 
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input,
                 torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device)
                ], dim = 1
            )

            if next_word == eos_idx: 
                break 

        return decoder_input.squeeze(0)
    
    def configure_optimizers(self): 
        suggested_lr = 3E-04
        
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['lr'], eps=1e-9) 
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=suggested_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs, 
            pct_start=10 / self.trainer.max_epochs,
            three_phase=True,
            div_factor=3,
            final_div_factor=10,
            anneal_strategy='linear',
            )
        scheduler_dict = {
            "scheduler": self.scheduler ,
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict} # 
    
    
    
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int=512,
    N: int=6,
    h: int=8,
    dropout: float=0.1,
    d_ff: int=256
    ) -> LitTransformer:
        
    # Create the embedding: layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N//2):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    #Create the decoder blocks
    decoder_blocks = []
    for _ in range(N//2):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    e1, e2, e3 = encoder_blocks
    d1, d2, d3 = decoder_blocks
    encoder_blocks1 = [e1, e2, e3, e3, e2, e1]
    decoder_blocks1 = [d1, d2, d3, d3, d2, d1]
   
        
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList (encoder_blocks1))
    decoder = Decoder(nn.ModuleList(decoder_blocks1))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = LitTransformer(
    encoder,
    decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            #nn.init.xavier_uniform_(p)
            nn.init.normal_(p, std=0.02)

    n_param = sum(p.numel() for p in transformer.parameters())
    print("Total Model Parameters:", n_param)

    return transformer

