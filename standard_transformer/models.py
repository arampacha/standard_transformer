# AUTOGENERATED! DO NOT EDIT! File to edit: 02_models.ipynb (unless otherwise specified).

__all__ = ['top_p_filter', 'top_k_filter', 'TransformerLM', 'TransformerEncDec']

# Cell
from fastai.basics import *
from .layers import *

# Cell
# generative helpers
# credit https://github.com/huggingface/transformers/blob/a0c62d249303a68f5336e3f9a96ecf9241d7abbe/src/transformers/generation_logits_process.py
def top_p_filter(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # if min_tokens_to_keep > 1:
    #         # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
    #         sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def top_k_filter(logits, top_k=20):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

_sampler = {
    'top_k':top_k_filter,
    'top_p':top_p_filter,
    'gready':lambda x: x.argmax(-1)
}

# Cell
class TransformerLM(Module):
    """
    Basic Transformer for language modelling
    Parameters:
        * vocab_sz: int
        * d_model: int - inner dimension of the model
        * n_layers: int (default: 6)
        * heads: int (default: 8)
        * d_ff: int - inner dimension of the pointwise FeedForward net, if None defaults to 4*d_model
        * attn_dropout: float - attention dropout
        * ff_dropout: float - feed-forward dropout
        * emb_dropout: float - embedding dropout
        * causal: bool (default: True) - if True does causal masking automatically
        * max_seq_len: int (default: 512)
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * prenorm: bool - wether to use PreNorm or PostNorm
        * attn_bias: bool - wether to allow biases in attention projection layers
        * pad_idx: int - padding token id, required for autogeneration of padding mask
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
        * axial_shape: tuple - required if 'axial' positional encoding are used, should be factors of
                max_seq_len
        * axial_emb_dims: tuple - [optional] axial embedding components, should sum to d_model
    Inputs:
        * x - input ids, shape [bs, sl]
        * mask - optional boolean mask, shape [bs, sl]
    Returns:
        * logits - target token logits, shape [bs, sl, vocab_sz]
    """
    def __init__(self,
                 vocab_sz,
                 d_model,
                 n_layers=6,
                 heads=8,
                 d_ff=None,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 emb_dropout=0.1,
                 tie_weights=True,
                 causal=True,
                 pos_enc='absolute',
                 max_seq_len=512,
                 axial_shape=None,
                 axial_emb_dims=None,
                 pad_idx=None,
                 prenorm=False,
                 attn_bias=True):
        store_attr('max_seq_len, n_layers, pad_idx')
        self.emb = TransformerEmbedding(vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                        axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        self.tfmr = TransformerEncoder(d_model, n_layers, heads, causal=causal, d_ff=d_ff,
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                       prenorm=prenorm, attn_bias=attn_bias, final_norm=nn.LayerNorm)
        self.proj = nn.Linear(d_model, vocab_sz)
        if tie_weights: self.proj.weight = self.emb.emb.weight

    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.tfmr(x, mask=mask)
        return self.proj(x)

    #TODO maybe refactor
    @torch.no_grad()
    def generate(self, inp,
                max_len=50,
                temperature=1.,
                method = 'top_k',
                top_k = 20,
                top_p = 0.9,
                early_stopping=False, #need eos_idx to work
                eos_idx=None):
        self.to(inp.device) #TODO test for potential problems
        self.eval()
        thresh = top_k if method=='top_k' else top_p
        sampler = _sampler[method]
        inp = expand_dim1(inp)
        b, t = inp.shape
        out = inp
        for _ in range(max_len):
            x = out[:, -self.max_seq_len:]

            logits = self(x)[:, -1, :]
            if method == 'greedy':
                sample = sampler(logits)
            else:
                filtered_logits = sampler(logits)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if early_stopping and (sample == eos_idx).all():
                break
        # out = out[:, t:]
        return out

    def store_attention(self, layer_ids=None):
        #defaults to storing attention for all layers
        layer_ids = default(layer_ids, list(range(self.n_layers)))
        for module in self.children():
            if issubclass(type(module), (TransformerEncoder, TransformerDecoder)):
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
    def get_attention_matrix(self):
        res = []
        for m in self.modules():
            if issubclass(type(m), (Attention)):
                attention = getattr(m, 'attention', None)
                if attention is not None:
                    res.append(attention)
                # reset stored attention
                m.attention = None
                m.store_attention = False
        return res

# Cell
#TODO test weight tying
# Note on weight tying: it's done like here in fastai AWD_LSTM model
# Lucidrains does it with custom MatrixMultiply module https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L106
#TODO: update docstrings
class TransformerEncDec(Module):
    """
    Basic Transformer Encoder-Decoder model
    Parameters:
        * enc_vocab_sz: int - source vocab size
        * dec_vocab_sz: int - target vocab size
        * d_model: int - inner dimension of the model
        * n_enc_layers: int (default: 6)
        * n_dec_layers: int (default: 6)
        * heads: int (default: 8)
        * d_ff: int - inner dimension of the pointwise FeedForward net, if None defaults to 4*d_model
        * attn_dropout: float - attention dropout
        * ff_dropout: float - feed-forward dropout
        * emb_dropout: float - embedding dropout
        * max_seq_len: int (default: 512)
        * prenorm: bool - whether to use PreNorm or PostNorm
        * attn_bias: bool - whether to allow biases in attention projection layers
        * pad_idx: int - padding token id, if pad_idx is provided, and no mask/context_mask are passed to
                forward method will be used to generate padding masks
        * tie_weights: bool - if True target embedding weights are used for computation output projection
        * shared_emb: bool - if True encoder and decoder will use shared embedding layer
        * pos_enc: str from {'absolute', 'fixed', 'axial'} - type of positional encoding to use
        * axial_shape: tuple - required if 'axial' positional encoding are used, should be factors of
                max_seq_len
        * axial_emb_dims: tuple - [optional] axial embedding components, should sum to d_model
    Inputs:
        * src - source input ids, shape [bs, src_sl]
        * tgt - target input ids, shape [bs, tgt_sl]
        * src_mask - optional boolean source mask, shape [bs, src_sl]
        * tgt_mask - optional boolean target mask, shape [bs, tgt_sl]
    Returns:
        * logits - target token logits, shape [bs, tgt_sl, tgt_vocab_sz]
    """
    def __init__(self,
                 enc_vocab_sz,
                 dec_vocab_sz,
                 d_model,
                 n_enc_layers=6,
                 n_dec_layers=6,
                 heads=8,
                 d_ff=None,
                 pad_idx=None,
                 tie_weights=True,
                 shared_emb = False,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 emb_dropout=0.1,
                 prenorm=False,
                 attn_bias=True,
                 comb_attn=False,
                 pos_enc='absolute',
                 max_seq_len=512,
                 axial_shape=None,
                 axial_emb_dims=None):
        store_attr('max_seq_len, n_enc_layers, n_dec_layers, pad_idx')
        self.enc_emb = TransformerEmbedding(enc_vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                            axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)
        if shared_emb:
            assert (enc_vocab_sz == dec_vocab_sz), "Encoder and decoder vocab size doesn't match"
            self.dec_emb = self.emc_emb
        else:
            self.dec_emb = TransformerEmbedding(dec_vocab_sz, d_model, max_seq_len, dropout=emb_dropout, pos_enc=pos_enc,
                                                axial_shape=axial_shape, axial_emb_dims=axial_emb_dims)

        self.encoder = TransformerEncoder(d_model, n_enc_layers, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                          prenorm=prenorm, attn_bias=attn_bias, final_norm=nn.LayerNorm, causal=False)
        self.decoder = TransformerDecoder(d_model, n_dec_layers, heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                          prenorm=prenorm, comb_attn=comb_attn, attn_bias=attn_bias, final_norm=nn.LayerNorm)
        self.proj = nn.Linear(d_model, dec_vocab_sz)
        if tie_weights: self.proj.weight = self.dec_emb.emb.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_mask = default(src_mask, self.get_padding_mask(src))
        tgt_mask = default(tgt_mask, self.get_padding_mask(tgt))
        enc = self.encoder(self.enc_emb(src), mask=src_mask)
        out = self.decoder(self.dec_emb(tgt), context=enc, mask=tgt_mask, context_mask=src_mask)
        return self.proj(out)

    def get_padding_mask(self, x):
        if self.pad_idx is None: return None
        return (x != self.pad_idx)

    #TODO add beam search and refactor
    @torch.no_grad()
    def generate(self, src,
                src_mask=None,
                max_len=50,
                temperature=1.,
                method = 'top_k',
                top_k = 20,
                top_p = 0.9,
                early_stopping=False,
                bos_idx=2, # TODO change to match future usecases
                eos_idx=None):
        self.to(src.device) #TODO test for potential problems
        self.eval()
        thresh = top_k if method=='top_k' else top_p
        sampler = _sampler[method]
        src = expand_dim1(src)
        bs = src.size(0)
        inp = src.new_full((bs, 1), bos_idx) #start with bos tokens
        src_mask = default(src_mask, self.get_padding_mask(src))
        enc = self.encoder(self.enc_emb(src), mask = src_mask)
        out = inp
        for _ in range(max_len):
            x = out[:, -self.max_seq_len:]
            dec = self.decoder(self.dec_emb(out), context=enc)
            logits = self.proj(dec)[:, -1, :]
            if method == 'greedy':
                sample = sampler(logits)
            else:
                filtered_logits = sampler(logits, thresh)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if (early_stopping and
                ((sample == eos_idx).all() or
                (sample == self.pad_idx).all())):
                break
        #TODO mb output cleanup
        return out

    def store_attention(self, layer_ids=None, store_encoder=False, store_decoder=True):
        #defaults to storing attention for all layers
        layer_ids = default(layer_ids, list(range(self.n_enc_layers)))
        for module in self.children():
            if issubclass(type(module), TransformerEncoder) and store_encoder:
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
            elif issubclass(type(module), TransformerDecoder) and store_decoder:
                for i, l in enumerate(module.layers):
                    if i in layer_ids:
                        for m in l.modules():
                            if issubclass(type(m), (Attention)):
                                m.store_attention = True
    #TODO mb separate encoder and decoder attention
    def get_attention_matrix(self, get_encoder=False, get_decoder=True):
        res = []
        if get_encoder:
            for m in self.encoder.modules():
                if issubclass(type(m), (Attention)):
                    attention = getattr(m, 'attention', None)
                    if attention is not None:
                        res.append(attention)
                    # reset stored attention
                    m.attention = None
                    m.store_attention = False
        if get_decoder:
            for m in self.decoder.modules():
                if issubclass(type(m), (Attention)):
                    attention = getattr(m, 'attention', None)
                    if attention is not None:
                        res.append(attention)
                    # reset stored attention
                    m.attention = None
                    m.store_attention = False
        return res