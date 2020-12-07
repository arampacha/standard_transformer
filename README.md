# Standard transformer
> PyTorch implementation of transformer as presented in "Attention is all you need" paper. Done using fastai and nbdev.


Some useful text may appear here in future

## Install

to be done...

## How to use

Causal language modelling:

```python
from standard_transformer.models import TransformerLM

x = torch.randint(256, (1, 64)) # bs = 1, seq_len = 64
model = TransformerLM(256, 64, max_seq_len=64, causal=True)
out = model(x)
out.shape
```




    torch.Size([1, 64, 256])


