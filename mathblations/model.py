"""
Copied (and then modified) from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/102024_ScaleUp1B/c0078066-c8c9-49c8-868a-ff4d4f32e615.txt
"""

import copy
from dataclasses import dataclass
from typing import Literal

import einops
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
    use_digits: bool = False
    k_gt_q: bool = True  # at input: digits to tokens -> k>q; at output: tokens to digits -> q>k
    n_layer_output: int = 0  # extra layers for moving from tokens to digits at output
    output_type: Literal["sequential", "cross_attention"] = "sequential"
    digit_mixin_method: Literal["cross_attn", "concat"] = "cross_attn"


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

    def __init__(self, config):
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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.length_factor = config.length_factor  # e.g., 3 or 5
        assert self.n_embd % self.n_head == 0
        
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.rotary = Rotary(self.head_dim)

        # Define the sliding window mask function
        if self.config.k_gt_q:
            def digit_to_token_mask(b, h, q_idx, kv_idx):
                return q_idx == (kv_idx // self.length_factor)
        else:
            def digit_to_token_mask(b, h, q_idx, kv_idx):
                return kv_idx == (q_idx // self.length_factor)

        # Create and store the block mask at initialization using known sequence length
        q_len = (config.T-1) * (1 if config.k_gt_q else self.length_factor)
        kv_len = (config.T-1) * (self.length_factor if config.k_gt_q else 1)
        self.block_mask = create_block_mask(
            digit_to_token_mask, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len
        )

    def forward(self, x_q, x_kv):
        B_q, T_q, C = x_q.size()
        B_kv, T_kv, C = x_kv.size()
        assert B_q == B_kv, f"Batch sizes must match: {B_q} vs {B_kv}"
        assert C == self.n_embd, f"Input dimension {C} doesn't match model {self.n_embd}"
        if self.config.k_gt_q:
            assert T_kv == T_q * self.length_factor, f"KV length {T_kv} must be {self.length_factor}x Q length {T_q}"
        else:
            assert T_kv * self.length_factor == T_q, f"Q length {T_q} must be {self.length_factor}x KV length {T_kv}"

        # Project queries from x_q
        q = self.c_q(x_q).view(B_q, T_q, self.n_head, self.head_dim)
        # Project keys and values from x_kv
        k = self.c_k(x_kv).view(B_kv, T_kv, self.n_head, self.head_dim)
        v = self.c_v(x_kv).view(B_kv, T_kv, self.n_head, self.head_dim)

        (cos_q, sin_q), (cos_k, sin_k) = self.rotary(q), self.rotary(k)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = apply_rotary_emb(q, cos_q, sin_q), apply_rotary_emb(k, cos_k, sin_k)

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
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class TokensToDigitsSequential(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attention_layers = nn.ModuleList([
            CausalSelfAttention(config) for _ in range(config.n_layer_output)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.repeat(x, f"... seq dim-> ... (seq {self.config.length_factor}) dim")
        for layer in self.attention_layers:
            x = x + layer(F.rms_norm(x, (x.size(-1),)))
        return x


class TokensToDigitsCrossAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.k_gt_q = False
        self.digit_attention_layers = nn.ModuleList([
            CausalSelfAttention(self.config) for _ in range(config.n_layer_output - 1)
        ])
        self.token_attention_layers = nn.ModuleList([
            CausalSelfAttention(self.config) for _ in range(config.n_layer_output - 1)
        ])
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(self.config) for _ in range(config.n_layer_output)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = einops.repeat(
            torch.empty_like(x).fill_(13).float(),  # 13 is the digit padding (not the token padding, which is 12)
            f"... seq dim-> ... (seq {self.config.length_factor}) dim",
        )
        for i in range(self.config.n_layer_output - 1):
            xd = xd + self.cross_attention_layers[i](
                x_q=F.rms_norm(xd, (xd.size(-1),)),
                x_kv=F.rms_norm(x, (x.size(-1),)),
            )
            x = x + self.token_attention_layers[i](F.rms_norm(x, (x.size(-1),)))
            xd = xd + self.digit_attention_layers[i](F.rms_norm(xd, (xd.size(-1),)))
        xd = xd + self.cross_attention_layers[-1](
            x_q=F.rms_norm(xd, (xd.size(-1),)),
            x_kv=F.rms_norm(x, (x.size(-1),)),
        )
        return xd


class DigitMixinCrossAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.digit_attn = CausalSelfAttention(config)
        self.cross_attn = CrossAttention(config)
    
    def forward(self, we: torch.Tensor, de: torch.Tensor):
            de = de + self.transformer.digit_attn(F.rms_norm(de, (de.size(-1),)))
            return self.transformer.cross_attn(
                x_q=F.rms_norm(we, (we.size(-1),)),
                x_kv=F.rms_norm(de, (de.size(-1),)),
            )


class DigitMixinConcat(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd * (1 + config.length_factor), config.n_embd)
        self.config = config
    
    def forward(self, we: torch.Tensor, de: torch.Tensor):
        B_toks, S_toks = de.shape
        B_digits, S_digits = we.shape
        assert B_toks == B_digits
        assert S_digits // S_toks == self.config.length_factor
        de = de.view(B_toks, S_toks, 1)
        we = we.view(B_toks, S_toks, -1)
        x = torch.cat([de, we], dim=-1)
        return self.fc(x)


class DigitMixinNoMixin(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block = Block(config)
    
    def forward(self, x, *args):
        return self.block(x)


def make_digit_mixin(config: GPTConfig) -> DigitMixinCrossAttention | DigitMixinConcat | DigitMixinNoMixin:
    if not config.use_digits:
        return DigitMixinNoMixin(config)
    
    if config.digit_mixin_method == "cross_attn":
        return DigitMixinCrossAttention(config)
    
    return DigitMixinConcat(config)


# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            dte = nn.Embedding(14, config.n_embd) if config.use_digits else nn.Identity(),  # 10 digits + pad & op & eq
            digit_mixin = make_digit_mixin(config),
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        # Any output layer -- tokens-to-digits, or Blocks to make up for parameters
        if config.use_digits and config.n_layer_output > 0:
            self.out_layer = (
                TokensToDigitsSequential(config)
                if config.output_type == "sequential"
                else TokensToDigitsCrossAttention(config)
            )
        else:
            self.out_layer = Block(config) if config.n_layer_output > 0 else nn.Identity()
        
        # LM head with tied weights
        if config.use_digits and config.n_layer_output > 0:
            self.lm_head = nn.Linear(config.n_embd, 14, bias=False)
            self.transformer.dte.weight = self.lm_head.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, digits=None):
        if self.config.use_digits:
            assert digits is not None, "Digits must be provided"
        # forward the GPT model itself
        we = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        # Digit embeddings
        de = self.transformer.dte(digits)
        x = self.transformer.digit_mixin(we, de)

        # Model backend
        for block in self.transformer.h:
            x = block(x)
        
        # Output layer
        x = self.out_layer(x)

        # Decode logits
        x = F.rms_norm(x, (x.size(-1),))
        logits = self.lm_head(x).float() # use tf32/fp32 for logits

        return logits
