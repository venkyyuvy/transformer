import torch
import torch.nn as nn
import math
from torch import optim 
from torch.utils.data import DataLoader, random_split

import os 
from pathlib import Path
 
# Huggingface datasets and tokenizers 
from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 

import torchmetrics
import torchmetrics.text
from torch.utils.tensorboard import SummaryWriter 

from pytorch_lightning import LightningModule
from dataset import BilingualDataset, causal_mask 
from config import get_weights_file_path 

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
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
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
        
    def forward(self, x) -> None:
        #- (batch, seq_len, d_model) ---> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        #- (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        #- (batch, -seq_len, -d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, -seq_len, -vocab_size)
        return self.projection_layer(x)
    
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
    ) -> Transformer:
        
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
        transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
        
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)
                nn.init.normal_(p, std=0.02)

        n_param = sum(p.numel() for p in transformer.parameters())
        print("Total Model Parameters:", n_param)

        return transformer





class LitTransformer(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config 
        self.console_width = 80
        
        # Tensorboard 
        self.writer = SummaryWriter(config['experiment_name'])  
        self.scaler = torch.cuda.amp.GradScaler()
        
        try: 
            # get the console window width 
            with os.popen('stty size', 'r') as console: 
                _, console_width = console.read().split() 
                self.console_width = int(console_width) 
        except: 
            # If we can't get the console width, use 80 as default 
            self.console_width = 80 
        
        #Validation variables
        self.val_count = 0 
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 
        self.val_num_examples = 2
        
        #Train variables
        self.train_losses =[] 
        
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        encoder_input = batch['encoder_input']
        decoder_input = batch['decoder_input']
        encoder_mask = batch['encoder_mask']
        decoder_mask = batch['decoder_mask']
        
        
        with torch.cuda.amp.autocast(enabled=True):
            
            # Run the tensors through the encoder, decoder and the projection layer 
            encoder_output = self.model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) 
            decoder_output = self.model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask) 
            proj_output = self.model.project(decoder_output) # (B, seq_len, vocab_size) 

            # Compare the output with the label 
            label = batch['label'] # (B, seg_len)

            # Compute the loss using a simple cross entropy 
            loss = self.loss_fn(
                proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            ) 
            # Calling self.log will surface up scalars for you in TensorBoard
            self.log("loss = ", loss.item(), prog_bar=True) 
            #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"}) 
            
            self.train_losses.append(loss.item())
            
            # Log the loss 
            self.writer.add_scalar('train,loss', loss.item(), self.trainer.global_step) 
            self.writer.flush() 
            
            self.optimizer.zero_grad() 
            
            # Backpropagate the loss 
            self.scaler.scale(loss).backward(retain_graph=True)
            
            #Update weights
            #self.scaler.step(self.optimizer)    
            self.optimizer.step()

        return loss
    
    def training_step_end(self, batch, batch_idx):        
        # Your train step end logic goes here
        scale = self.scaler.get_scale()
        self.scaler.update()
        skip_lr_sched = (scale > self.sceler.get_scale())
        if not skip_lr_sched:
            self.scheduler.step()
    

    def validation_step(self, batch, batch_idx):       
        max_len = self.config['seq_len'] 
        
        if self.val_count == self.val_num_examples:             
            return 
        
        self.val_count += 1 
        with torch.no_grad():             
            encoder_input = batch["encoder_input"]
            encoder_mask = batch["encoder_mask"]

            # check that the batch size is 1 
            assert encoder_input.size(0) == 1, \
            "Batch  size must be 1 for val"

            model_out = self.greedy_decode(
                self.model,
                encoder_input,
                encoder_mask,
                self.tokenizer_src,
                self.tokenizer_tgt,
                max_len,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0] 
            model_out_text = self.tokenizer_tgt.decode(
                model_out.detach().cpu().numpy()) 

            self.val_source_texts.append(source_text) 
            self.val_expected.append(target_text) 
            self.val_predicted.append(model_out_text) 

            # Print the source, target and model output             
            print('-'*self.console_width) 
            print(f"{f'SOURCE: ':>12}{source_text}") 
            print(f"{f'TARGET: ':>12}{target_text}")
            print(f"{f'PREDICTED: ':>12}{model_out_text}") 
            print('-'*self.console_width)

        
            
    def on_validation_epoch_end(self):
        writer = self.writer
        if writer:
            # Evaluate the character error rate 
            # Compute the char error rate 
            metric = torchmetrics.text.CharErrorRate() 
            cer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('Validation cer', cer, self.trainer.global_step) 
            self.log("val_cer", cer, prog_bar=True)
            print(f'Val CER at end of epoch {self.trainer.current_epoch} = {cer}')
            writer.flush() 

            # Compute the word error rate 
            metric = torchmetrics.text.WordErrorRate() 
            wer = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('Validation wer', wer, self.trainer.global_step)
            self.log("val_wer", wer, prog_bar=True)
            print(f'Val WER at end of epoch {self.trainer.current_epoch} = {wer}')
            writer.flush() 

            # Compute the BLEU metric 
            metric = torchmetrics.text.BLEUScore() 
            bleu = metric(self.val_predicted, self.val_expected) 
            writer.add_scalar('Validation BLEU', bleu, self.trainer.global_step)
            self.log("val_bleu", bleu, prog_bar=True)
            print(f'Val BLEU at end of epoch {self.trainer.current_epoch} = {bleu}')
            writer.flush() 
            
        self.val_count = 0
        self.val_source_texts = [] 
        self.val_expected = [] 
        self.val_predicted = [] 

    def test_step(self, batch, batch_idx):
        pass
    
      
    def on_train_epoch_end(self):
        # Save the model at the end of every epoch   
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        print(f'Average Training Loss at end of epoch'+
              f'{self.trainer.current_epoch} = {mean_loss}')
        
        print(f'Learning Rate at end of epoch {self.trainer.current_epoch}'+
              f'= {self.scheduler.get_last_lr()}')
        
        # Save the model at the end of every 5th epoch - to save memory
        curr_epoch = self.trainer.current_epoch + 1
        if curr_epoch % 5 == 0:
            model_filename = get_weights_file_path(
                self.config, f"{self.trainer.current_epoch:02d}") 
            torch.save({ 
                        'epoch': self.trainer.current_epoch, 
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(), 
                        'global_step': self.trainer.global_step}
                       , model_filename) 
        self.train_losses = []
            
            
    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len): 
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step 
        encoder_output = model.encode(source, source_mask) 

        # Initialize the decoder input with the sos token 
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(model.device) 

        while True: 
            if decoder_input.size(1) == max_len:  
                break 

            # build mask for target 
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(model.device) 

            # calculate output 
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) 

            # get next token 
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input,
                 torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(model.device)
                ], dim = 1
            )

            if next_word == eos_idx: 
                break 

        return decoder_input.squeeze(0)
    
    def configure_optimizers(self): 
        suggested_lr = 3E-04
        
        steps_per_epoch = len(self.train_dataloader())
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=suggested_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs, 
            pct_start=10/self.trainer.max_epochs,
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
    
    
    ####################
    # DATA RELATED HOOKS
    ####################
    
    def get_all_sentences(self, ds, lang): 
        for item in ds: 
            yield item['translation'][lang]
    
    def get_model(self, config, vocab_src_len, vocab_tgt_len): 
        model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], 
                                  config["seq_len"], d_model=config["d_model"])
        return model
    
    def get_or_build_tokenizer(self, config, ds, lang): 
        tokenizer_path = Path(config['tokenizer_file'].format(lang)) 
        if not Path.exists(tokenizer_path): 
            # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour 
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) 
            tokenizer.pre_tokenizer = Whitespace() 
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                min_frequency=2)
            tokenizer.train_from_iterator(
                self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path)) 
        else: 
            tokenizer = Tokenizer.from_file(str(tokenizer_path)) 
        return tokenizer

    def prepare_data(self):  
        ds_raw = load_dataset(
            'opus_books', 
            f"{self.config['lang_src']}-{self.config['lang_tgt']}",
            split='train'
        )  
        
        # Define a function to filter dataset as per assignment requirement
        def filter_examples(example):
            source_text = example['translation'][self.config['lang_src']]
            target_text = example['translation'][self.config['lang_tgt']]
            return len(source_text) <= 150 and len(target_text) <= len(source_text) + 10

        # Filter the dataset based on the custom filter function
        ds_raw = ds_raw.filter(filter_examples)
        
        # Build tokenizers 
        self.tokenizer_src = self.get_or_build_tokenizer(
            self.config, ds_raw, self.config['lang_src'])
        self.tokenizer_tgt = self.get_or_build_tokenizer(
            self.config, ds_raw, self.config['lang_tgt'])
        
        #Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id('[PAD]'), 
            label_smoothing=0.1
        )
        
        # Get model        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = self.get_model(
            self.config,
            self.tokenizer_src.get_vocab_size(),
            self.tokenizer_tgt.get_vocab_size())
        
        #Optimizer
      #   self.optimizer = torch.optim.Adam(
      #       self.model.parameters(), lr=self.config['lr'], eps=1e-9) 
      # 
        # Keep 90% for training, 10% for validation 
        train_ds_size = int(0.9* len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        self.train_ds = BilingualDataset(
            train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config['lang_src'],
            self.config['lang_tgt'],
            self.config['seq_len']
        )
        self.val_ds = BilingualDataset(
            val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config['lang_src'],
            self.config['lang_tgt'],
            self.config['seq_len']
        )

        # Find the maximum length of each sentence in the source and target sentence 
        max_len_src = 0 
        max_len_tgt = 0 

        for item in ds_raw: 
            src_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids
            tgt_ids = self.tokenizer_src.encode(item['translation'][self.config['lang_tgt']]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f'Maximum length of source sentence: {max_len_src}') 
        print(f'Maximum length of target sentence: {max_len_tgt}') 
  

    def setup(self, stage=None):
        pass 

    def train_dataloader(self):                   
        return DataLoader(
            self.train_ds, batch_size=self.config['batch_size'], 
            shuffle=True, collate_fn = self.collate_fn, 
            num_workers=min(os.cpu_count(), 4),
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=True, 
            num_workers=min(os.cpu_count(), 4),
            persistent_workers=True,
            pin_memory=True
        ) 
    

    
    def collate_fn(self, batch):
            encoder_input_max = max(x["encoder_str_length"] for x in batch)
            decoder_input_max = max(x["decoder_str_length"] for x in batch)
            encoder_inputs = []
            decoder_inputs = []
            encoder_mask = []
            decoder_mask = []
            label = []
            src_text = []
            tgt_text = []

            for b in batch:
                encoder_inputs.append(b["encoder_input"][:encoder_input_max])
                decoder_inputs.append(b["decoder_input"][:decoder_input_max])
                encoder_mask.append(
                (b["encoder_mask"][0, 0, :encoder_input_max])\
                    .unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
                decoder_mask.append(
                (b["decoder_mask"][0, :decoder_input_max, :decoder_input_max])\
                    .unsqueeze(0).unsqueeze(0).int())
                label.append(b["label"][:decoder_input_max])
                src_text.append(b["src_text"])
                tgt_text.append(b["tgt_text"])
            return {
              "encoder_input":torch.vstack(encoder_inputs),
              "decoder_input":torch.vstack(decoder_inputs),
              "encoder_mask": torch.vstack(encoder_mask),
              "decoder_mask": torch.vstack(decoder_mask),
              "label":torch.vstack(label),
              "src_text":src_text,
              "tgt_text":tgt_text
            }
