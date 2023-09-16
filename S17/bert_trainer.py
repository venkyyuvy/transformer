from torch.utils.data import Dataset
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import torch
import re

from bert import Transformer
batch_size = 1024
seq_len = 20
embed_size = 128
inner_ff_size = embed_size * 4
n_heads = 8
n_code = 8
n_vocab = 40000
dropout = 0.1
n_workers = 4


class SentencesDataset(Dataset):
    #Init dataset
    def __init__(self, sentences, vocab, seq_len):
        self.sentences = sentences
        self.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        self.vocab = {e:i for i, e in enumerate(self.vocab)} 
        self.rvocab = {v:k for k,v in self.vocab.items()}
        self.seq_len = seq_len
        
        #special tags
        self.IGNORE_IDX = self.vocab['<ignore>'] #replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = self.vocab['<oov>'] #replacement tag for unknown words
        self.MASK_IDX = self.vocab['<mask>'] #replacement tag for the masked word prediction task
    
    
    #fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        #while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < self.seq_len:
            s.extend(self.get_sentence_idx(index % len(self)))
            index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:self.seq_len]
        [s.append(self.IGNORE_IDX) 
            for _ in range(self.seq_len - len(s))] #PAD ok
        
        #apply random mask
        s = [(self.MASK_IDX, w) \
            if random.random() < p_random_mask \
            else (w, self.IGNORE_IDX) \
            for w in s]
        
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    def __len__(self):
        return len(self.sentences)

    def get_sentence_idx(self, index):
        s = self.sentences[index]
        s = [self.vocab[w] if w in self.vocab else self.OUT_OF_VOCAB_IDX
            for w in s] 
        return s

datapath = './bert_dataset/'
pth = datapath + 'training.txt'
sentences = open(pth).read().lower().split('\n')

print('tokenizing sentences...')
special_chars = ',?;.:/*!+-()[]{}"\'&'
sentences = [
    re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') 
    for s in sentences]
sentences = [[w for w in s if len(w)] for s in sentences]

print('creating/loading vocab...')
pth = datapath + 'vocab.txt'
if not exists(pth):
    words = [w for s in sentences for w in s]
    vocab = Counter(words).most_common(n_vocab) #keep the N most frequent words
    vocab = [w[0] for w in vocab]
    open(pth, 'w+').write('\n'.join(vocab))
else:
    vocab = open(pth).read().split('\n')
    


class LitBERT(Transformer, pl.LightningModule):
    def __init__(self, *args):
        super().__init__(*args)

    def training_step(self, batch, batch_idx):
        masked_input = batch['input']
        masked_target = batch['target']
        
        masked_input = masked_input
        masked_target = masked_target
        output = self(masked_input)
        
        #compute the cross entropy loss 
        output_v = output.view(-1,output.shape[-1])
        target_v = masked_target.view(-1,1).squeeze()
        loss_fn = nn.CrossEntropyLoss(ignore_index=index)
        loss = loss_fn(output_v, target_v)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), 
                               lr=1e-4, 
                               weight_decay=1e-4,
                               betas=(.9,.999))
        return optimizer


# model
if __name__ == '__main__':
    dataset = SentencesDataset(sentences, vocab, seq_len)
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    index = dataset.IGNORE_IDX

    model = LitBERT(
        n_code, 
        n_heads,
        embed_size,
        inner_ff_size,
        len(dataset.vocab),
        seq_len,
        dropout)

    # train model
    trainer = pl.Trainer(
        max_steps=10_000,
    )
    trainer.fit(
        model=model,
        train_dataloaders=data_loader,
    )
    print('saving embeddings...')
    N = 3000
    np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    open('names.tsv', 'w+').write('\n'.join(s) )


    print('end')

