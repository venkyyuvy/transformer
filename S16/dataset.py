from torch.utils.data import DataLoader, random_split

import os 
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from datasets import load_dataset 
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace 

class BilingualDataset (Dataset):
    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        seq_len
    ):
        super().__init__()
        self.seq_len = seq_len
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")],dtype=torch.int64)
        
    def __len__(self): 
        return len(self.ds) 
    
    def __getitem__(self, idx): 
        src_target_pair = self.ds[idx] 
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang] 
        
        # Transform the text into tokens 
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids 
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids 
        
        # Add sos, eos and padding to each sentence 
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 # We will add <s> and </s> 
        
        # We will only add <s>, and </s> only on the label 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 
        
        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:  
            raise ValueError("Sentence is too long") 
                             
        # Add <s> and </s> token 
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(enc_input_tokens, dtype=torch.int64),  
                self.eos_token,  
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64), 
            ],
            dim=0, 
        )
        # Add only <s> token 
        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64), 
            ], 
            dim=0, 
        )
        # Add only </s> token 
        label = torch.cat(
            [
               torch.tensor(dec_input_tokens, dtype=torch.int64), 
               self.eos_token, 
               torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64), 
            ], 
            dim=0, 
        )
        
        # Double check the size of the tensors to make sure they are all seq_len long 
        assert encoder_input.size(0) == self.seq_len 
        assert decoder_input.size(0) == self.seq_len 
        assert label.size(0) == self.seq_len 
        
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token)\
                .unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)\
                .unsqueeze(0).int() & causal_mask(decoder_input.size(0)), 
            # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text, 
            "tgt_text": tgt_text, 
            "encoder_str_length": len(enc_input_tokens), # (seq_len)
            "decoder_str_length": len(dec_input_tokens) # (seq_len)
        }

def causal_mask(size):
    mask = torch.triu(
        torch.ones((1, size, size)),
        diagonal=1).type(torch.int)
    return mask == 0 


class TranslateDataset(LightningDataModule):
    def __init__(
        self,
        config,
        num_workers=2,
    ):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.batch_size = config["batch_size"]

    def get_all_sentences(self, ds, lang): 
        for item in ds: 
            yield item['translation'][lang]
    
    
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

    def setup(self, stage=None):  
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
        
        
        # Keep 90% for training, 10% for validation 
        train_ds_size = int(0.9* len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(
            ds_raw, [train_ds_size, val_ds_size])

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
  


    def train_dataloader(self):                   
        return DataLoader(
            self.train_ds, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            collate_fn = self.collate_fn, 
            # num_workers=min(os.cpu_count(), 4),
            # persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False, 
            # num_workers=min(os.cpu_count(), 4),
            # persistent_workers=True,
            pin_memory=True,
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
            encoder_mask.append((b["encoder_mask"]\
                [0, 0, :encoder_input_max])\
                .unsqueeze(0).unsqueeze(0).unsqueeze(0).int())
            decoder_mask.append((b["decoder_mask"]\
                [0, :decoder_input_max, :decoder_input_max])\
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
