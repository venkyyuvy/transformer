import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def encode(text_seq: str, tokenizer: any) -> torch.Tensor:
    """
    Function to encode input text using a pre-trained tokenizer and vectorized lookups
    """
    # tokenize the input text
    tokens = tokenizer.tokenize(text_seq)
    # convert the tokens to their corresponding ids
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    return token_indices
# load model from checkpoint
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

# example to decode sequence
# enc_sec = m.generate(idx=torch.zeros((1,1), dtype=torch.long),
# max_new_tokens=20)[0].tolist()
# print(decode(vocab=vocab, enc_sec=enc_sec))

# raw data


# data spliting
# n = int(0.9 * len(data))  # first 90% will be train, rest val
# self.train_data = data[:n]
# self.val_data = data[n:]
#
# no notion of lines or sentences. 
# entire corpus is considered as one single document.
# sampling happens between 0 and n-seq_len
# next word of the sequence is considered as the target
class ERATranscriptDataset(Dataset):
    def __init__(
            self,
            path_do_data: str="./gpt_dataset/english.txt",
            seq_len: int=20):
        data_raw = open(path_do_data, encoding="utf-8").read()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = tokenizer.vocab_size
        self.data = encode(text_seq=data_raw, tokenizer=tokenizer)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        data = self.data[idx: idx + self.seq_len]
        target = self.data[idx + 1: idx + self.seq_len + 1]
        return data, target

batch_size = 32
dataset = ERATranscriptDataset()
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
