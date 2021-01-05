# AUTOGENERATED! DO NOT EDIT! File to edit: 01_layers.ipynb (unless otherwise specified).

__all__ = ['exists', 'default', 'expand_dim1', 'Residual', 'PostNorm', 'PreNorm', 'FeedForward', 'MASK_VAL',
           'Attention', 'AdditiveAttention', 'AttnInProj', 'ScaledDotProdAttention', 'Attention',
           'TransformerEncoderBlock', 'TransformerEncoder', 'TransformerDecoderBlock', 'TransformerDecoderBlockV2',
           'TransformerDecoder', 'AbsolutePositionalEmbedding', 'FixedPositionalEmbedding', 'TransformerEmbedding']

# Cell
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial, reduce
from inspect import isfunction
from operator import mul
from fastai.basics import *
from einops import rearrange, repeat
try:
    from axial_positional_embedding import AxialPositionalEmbedding, AxialPositionalEmbeddingImage
except ImportError as e:
    print(e)

# Cell

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def expand_dim1(x):
    if len(x.shape) == 1:
        return x[None, :]
    else: return x

# Cell
class Residual(nn.Module):
    """Add skip-connection: out = x + sublayer(x)"""
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer
    def forward(self, x, *args, **kwargs):
        return self.sublayer(x, *args, **kwargs) + x

class PostNorm(nn.Module):
    """Adds LayerNorm after sublayer"""
    def __init__(self, dim, sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.sublayer(x, *args, **kwargs)
        return self.norm(x)

class PreNorm(nn.Module):
    """Adds LayerNorm before sublayer"""
    def __init__(self, dim, sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.sublayer(x, *args, **kwargs)

# Cell
class FeedForward(nn.Module):
    """
    Simple positional feed-forward module with GELU activation function.
    If d_ff is None defaults to 4*d_model
    """
    def __init__(self, dim, d_ff=None, dropout=0.):
        super().__init__()
        d_ff = default(d_ff, 4*dim)
        self.layers = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
            nn.Dropout(dropout)
        )
        self._init()
    def forward(self, x):
        return self.layers(x)
    def _init(self):
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

# Cell
MASK_VAL = -5e4

# Cell
class Attention(nn.Module):
    """Standard attention module"""
    def __init__(self,
                 dim,
                 n_heads = 8,
                 causal = False,
                 mask = None,
                 dropout=0.1,
                 bias=True):
        super().__init__()
        self.causal = causal
        self.store_attention = False
        self.mask = mask #??
        self.n_heads = n_heads
        self.scale = (dim//n_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(dim, dim)

        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, _, h, device = *x.shape, self.n_heads, x.device
        kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        # boolean input_mask is False at positions not to attend to
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask
        # classic dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q*self.scale, k)
        # might need to tune MASK_VAL for fp16 to work
        if exists(input_mask):
            dots.masked_fill_(~input_mask, MASK_VAL)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones((i, j), device = device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, MASK_VAL)
            del mask

        attn = F.softmax(dots, -1)
        if self.store_attention: # and not self.training
            self.attention = attn.detach().cpu()
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        #out = self.dropout(out) # option for more dropout here
        return out

    def _init(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        if getattr(self.to_q, 'bias', None) is not None:
            nn.init.constant_(self.to_q.bias, 0)
        if getattr(self.to_kv, 'bias', None) is not None:
            nn.init.constant_(self.to_kv.bias, 0)
        nn.init.constant_(self.to_out.bias, 0)

# Cell
class AdditiveAttention(nn.Module):
    """Additive attention combining self and cross attention"""
    def __init__(self,
                 dim,
                 n_heads = 8,
                 causal = False,
                 mask = None,
                 dropout=0.1,
                 bias=True):
        super().__init__()
        self.causal = causal
        self.store_attention = False
        self.mask = mask #??
        self.n_heads = n_heads
        self.scale = (dim//n_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias = bias)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(dim, dim)

        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None, store_attention=False):
        b, n, d, h, device = *x.shape, self.n_heads, x.device
        context = default(context, torch.empty(b, 0, d, dtype=x.dtype, device=device))
        kv_input = torch.cat([x, context], dim=-2)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))

        # boolean input_mask is False at positions not to attend to
        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            self_mask = q_mask[:, None, :, None] * q_mask[:, None, None, :]
            if context.size(-2) != 0:
                k_mask = default(context_mask, lambda: torch.ones((b, context.shape[-2]), device = device).bool())
                cross_mask = q_mask[:, None, :, None] * k_mask[:, None, None, :]
            else: cross_mask = torch.empty(0, dtype=self_mask.dtype, device=device)
            input_mask = torch.cat([self_mask, cross_mask], dim=-1)
        # classic scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q * self.scale, k)
        # might need to tune MASK_VAL for fp16 to work
        if exists(input_mask):
            dots.masked_fill_(~input_mask, MASK_VAL)
            del input_mask

        if self.causal:
            i, j = torch.triu_indices(n, n, 1)
            dots[:,:,i,j] = MASK_VAL

        attn = F.softmax(dots, -1)
        if self.store_attention: # and not self.training
            self.attention = attn.detach().cpu()
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

    def _init(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        if getattr(self.to_q, 'bias', None) is not None:
            nn.init.constant_(self.to_q.bias, 0)
        if getattr(self.to_kv, 'bias', None) is not None:
            nn.init.constant_(self.to_kv.bias, 0)
        nn.init.constant_(self.to_out.bias, 0)

# Cell
class AttnInProj(nn.Module):
    """Computes q, k, v from input x and [optional] context"""
    def __init__(self, d_model:int, bias:bool=False):
        super().__init__()
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_kv = nn.Linear(d_model, 2*d_model, bias=bias)
    def forward(self, x, context=None):
        context = default(context, x)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, -1)
        return q, k, v

# Cell
#TODO make sure store_attention works
class ScaledDotProdAttention(Module):

    def __init__(self, d_model, n_heads, causal=False, dropout=0., store_attention:bool=False):
        store_attr()
        self.scale = (d_model//n_heads)**-0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        device = q.device
        bs, sl, d, cl = *q.size(), k.size(1)

        q = q.view(bs, sl, self.n_heads, -1).transpose(1, 2)
        k = k.view(bs, cl, self.n_heads, -1).transpose(1, 2)
        v = v.view(bs, cl, self.n_heads, -1).transpose(1, 2)
        # classic dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q*self.scale, k)

        if exists(attn_mask):
            dots.masked_fill_(~input_mask, MASK_VAL)
            del input_mask
        if self.causal:
            i, j = torch.triu_indices(sl, sl, 1)
            dots[:,:,i,j] = MASK_VAL

        attn = F.softmax(dots, -1)
        if self.store_attention: self.attention = attn.detach().cpu()

        attn = self.dropout(attn)
        out = torch.einsum('bhij, bhjd -> bihd', attn, v)
        return out.contiguous().view(bs, sl, -1)

# Cell
class Attention(nn.Module):
    """
    Standard attention module using scaled dot-product attention
    """
    def __init__(self,
                 d_model:int,
                 n_heads:int = 8,
                 causal:bool = False,
                 mask:Tensor = None,
                 dropout:float=0.1,
                 out_dropout:float=None,
                 bias:bool=False,
                 store_attention:bool=False):
        super().__init__()
        store_attr('causal, mask, n_heads, bias')
        out_dropout = default(out_dropout, dropout)
        self.in_proj = AttnInProj(d_model, bias=bias)
        self.attn = ScaledDotProdAttention(d_model, n_heads, causal=causal,
                                           dropout=dropout, store_attention=store_attention)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(out_dropout)
        self._init()

    def forward(self, x, context = None, mask = None, context_mask = None):
        q, k, v = self.in_proj(x, context)

        attn_mask = self._make_input_mask(mask, context_mask, x, context)
        out = self.attn(q, k, v, attn_mask)

        out = self.out_proj(out)
        return self.dropout(out)

    def _init(self):
        [nn.init.xavier_uniform_(w) for w in self.parameters() if w.dim()>1]
        if self.bias:
            [nn.init.constant_(b, 0) for b in self.parameters() if b.dim()==1]

    def _make_input_mask(self, mask, context_mask, x, context):
        if any(map(exists, (mask, context_mask))):
            b, n, _, device = *x.size(), x.device
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, context.shape[-2]), device = device).bool())

            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            return q_mask * k_mask
        else: return None #input_mask is None if both mask and context_mask are None

# Cell
class TransformerEncoderBlock(nn.Module):
    """
    Bacis transformer encoder block. Consists of multi-head attention and positional feedforward layers
    """
    def __init__(self, dim, n_heads = 8, causal = False, mask = None,
                 attn_dropout=0.1, attn_bias=True, ff_dropout=0.1, d_ff=None,
                 prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(dim, Attention(dim, n_heads=n_heads, causal=causal, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(dim, FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(dim, Residual(Attention(dim, n_heads=n_heads, causal=causal, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(dim, Residual(FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))

    def forward(self, x, mask=None): #? more args
        out = self.attn(x, mask=mask)
        out = F.dropout(out, p=self.attn_dropout, training=self.training)
        out = self.ff(out)
        return out

# Cell
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=6, n_heads=8, causal=False, d_ff=None, attn_dropout=0.1, attn_bias=True,
                ff_dropout=0.1, prenorm=False, final_norm=None):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerEncoderBlock(dim, n_heads, causal=causal, d_ff=d_ff,
                                    attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, attn_bias=attn_bias))
        self.norm = None if final_norm is None else final_norm(dim)
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

# Cell
class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, n_heads = 8, mask = None, d_ff=None,
                 attn_dropout=0.1, ff_dropout=0.1, attn_bias=True,
                 prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(dim, Attention(dim, n_heads=n_heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.cross = Residual(PreNorm(dim, Attention(dim, n_heads=n_heads, causal=False, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(dim, FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(dim, Residual(Attention(dim, n_heads=n_heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.cross = PostNorm(dim, Residual(Attention(dim, n_heads=n_heads, causal=False, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(dim, Residual(FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))

    def forward(self, x, context, mask=None, context_mask=None):
        out = self.attn(x, mask=mask)
        out = F.dropout(out, p=self.attn_dropout, training=self.training)
        out = self.cross(out, context, mask=mask, context_mask=context_mask)
        out = F.dropout(out, p=self.attn_dropout, training=self.training)
        out = self.ff(out)
        return out

# Cell
class TransformerDecoderBlockV2(nn.Module):
    def __init__(self, dim, n_heads = 8, mask = None, d_ff=None,
                 attn_dropout=0.1, ff_dropout=0.1, attn_bias=True,
                 prenorm=False):
        super().__init__()
        self.attn_dropout = attn_dropout # mb separate argument attn_post_dropout
        if prenorm:
            self.attn = Residual(PreNorm(dim, DecoderAttention(dim, n_heads=n_heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.ff = Residual(PreNorm(dim, FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))
        else:
            self.attn = PostNorm(dim, Residual(DecoderAttention(dim, n_heads=n_heads, causal=True, dropout=attn_dropout, bias=attn_bias)))
            self.ff = PostNorm(dim, Residual(FeedForward(dim, d_ff=d_ff, dropout=ff_dropout)))

    def forward(self, x, context, mask=None, context_mask=None):
        out = self.attn(x, context, mask=mask, context_mask=context_mask)
        out = F.dropout(out, p=self.attn_dropout, training=self.training)
        out = self.ff(out)
        return out

# Cell
class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth=6, n_heads=8, d_ff=None, attn_dropout=0.1, ff_dropout=0.1,
                 prenorm=False, comb_attn=False, attn_bias=True, final_norm=None):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        block = TransformerDecoderBlockV2 if comb_attn else TransformerDecoderBlock
        for _ in range(depth):
            self.layers.append(block(dim, n_heads, d_ff=d_ff, attn_dropout=attn_dropout, ff_dropout=ff_dropout, prenorm=prenorm, attn_bias=attn_bias))
        self.norm = None if final_norm is None else final_norm(dim)
    def forward(self, x, context, mask=None, context_mask=None):
        for layer in self.layers:
            x = layer(x, context, mask, context_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

# Cell
class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

class TransformerEmbedding(nn.Module):
    """
    Combines token embedings with positional encodings
    pos_enc: str from {'absolute', 'fixed', 'axial'}
    """
    def __init__(self, emb_sz, dim, max_seq_len=512, dropout=0., pos_enc='absolute',
                 axial_shape=None, axial_emb_dims=None):
        super().__init__()
        self.scale = dim**0.5
        self.emb = nn.Embedding(emb_sz, dim)
        if pos_enc == 'absolute':
            self.pos_enc = AbsolutePositionalEmbedding(dim, max_seq_len)
        elif pos_enc == 'fixed':
            self.pos_enc = FixedPositionalEmbedding(dim)
        elif pos_enc == 'axial':
            assert axial_shape is not None
            assert reduce(mul, axial_shape) == max_seq_len
            axial_emb_dims = default(axial_emb_dims, get_axial_dims(dim, len(axial_shape)))
            self.pos_enc = AxialPositionalEmbedding(dim, axial_shape, axial_emb_dims)
        self.dropout = nn.Dropout(dropout)
        self._init()
    def forward(self, x):
        x = self.emb(x)
        x *= self.scale
        x += self.pos_enc(x)
        return self.dropout(x)
    def _init(self):
        nn.init.trunc_normal_(self.emb.weight, std=1/self.scale)
        # 0.02 works worse then std=1 for pe, trying d_emb**-0.5
#         if hasattr(self.pos_enc, 'emb'):
#             nn.init.trunc_normal_(self.pos_enc.emb.weight, std=1/self.scale)