"""
Copied (and then modified) from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/102024_ScaleUp1B/c0078066-c8c9-49c8-868a-ff4d4f32e615.txt
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    T: int = 1024
    length_factor: int = 3  # for cross attention between digits and embeddings; == max_digits_per_token
    sliding_window_size: int | None = None  # only for digits
    use_digits: bool = False


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

        if config.sliding_window_size is not None:
            def sliding_window_causal(b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                window_mask = q_idx - kv_idx <= config.sliding_window_size 
                return causal_mask & window_mask
            self.block_mask = create_block_mask(sliding_window_causal, B=None, H=None, 
                                                Q_LEN=config.T, KV_LEN=config.T)
        else:
            def causal(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            self.block_mask = create_block_mask(causal, B=None, H=None, 
                                                Q_LEN=config.T, KV_LEN=config.T)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = flex_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            block_mask=self.block_mask,
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.length_factor = config.length_factor  # e.g., 3 or 5
        assert self.n_embd % self.n_head == 0
        
        # Query projection for x_q
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # Key and Value projections for x_kv
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init
        self.rotary = Rotary(self.head_dim)

        # Define the sliding window mask function
        def sliding_window_mask(b, h, q_idx, kv_idx):
            window_start = kv_idx * self.length_factor
            window_end = (kv_idx + 1) * self.length_factor
            return (q_idx >= window_start) & (q_idx < window_end)

        # Create and store the block mask at initialization using known sequence length
        self.block_mask = create_block_mask(sliding_window_mask, B=None, H=None, 
                                          Q_LEN=config.T * self.length_factor, 
                                          KV_LEN=config.T)

    def forward(self, x_q, x_kv):
        B_q, T_q, C = x_q.size()
        B_kv, T_kv, C = x_kv.size()
        assert B_q == B_kv, f"Batch sizes must match: {B_q} vs {B_kv}"
        assert C == self.n_embd, f"Input dimension {C} doesn't match model {self.n_embd}"
        assert T_q == T_kv * self.length_factor, f"Query length {T_q} must be {self.length_factor}x KV length {T_kv}"

        # Project queries from x_q
        q = self.c_q(x_q).view(B_q, T_q, self.n_head, self.head_dim)
        # Project keys and values from x_kv
        k = self.c_k(x_kv).view(B_kv, T_kv, self.n_head, self.head_dim)
        v = self.c_v(x_kv).view(B_kv, T_kv, self.n_head, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # Cross attention with sliding window mask
        y = flex_attention(
            q.transpose(1, 2),  # [B, nh, T_q, hd]
            k.transpose(1, 2),  # [B, nh, T_kv, hd]
            v.transpose(1, 2),  # [B, nh, T_kv, hd]
            block_mask=self.block_mask
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C)
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        config.sliding_window_size = None  # only for digits -> when using CausalSelfAttention isolated
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            dte = nn.Embedding(config.vocab_size, config.n_embd) if config.use_digits else None,
            digit_attn = CausalSelfAttention(config) if config.use_digits else None,
            cross_attn = CrossAttention(config) if config.use_digits else None,
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, digits=None):
        if self.config.use_digits:
            assert digits is not None, "Digits must be provided"
        # forward the GPT model itself
        we = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if self.config.use_digits:
            de = self.transformer.dte(digits)
            de = self.transformer.digit_attn(de)
            x = self.cross_attn(x_q=we, x_kv=de)
        else:
            x = we
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        logits = logits.float() # use tf32/fp32 for logits

        return logits
