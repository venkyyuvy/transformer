
# Translation using transformers

English to French translation using Opus dataset. Optimizing the training time of transformer

- One cycle policy
- Mixed precision training (Mixed-16)
- dynamic max_sequence length for batches
- cycle encoder and decoder layers
- Removes all English sentences with more than 150 "tokens"
- Removes all french sentences where len(fench_sentences) > len(english_sentrnce) + 10


Achieved a training loss of 1.5 with 30 epoches.
