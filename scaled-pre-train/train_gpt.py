"""
Modified from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/021425_GPT2MediumOptCoeffs/1baa66b2-bff7-4850-aced-d63885ffb4b6.txt
"""

import argparse
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import time
import random
import functools
import json
from typing import Literal
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess as sp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import einops
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

from data_creation import make_embedding, tokens_to_bytes, pull_from_left, pull_from_right
from data_download import download
import wandb
import safetensors.torch
from huggingface_hub import upload_file, create_repo

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model
@dataclass
class ByteHyperparameters:
    bytes_per_token: int = 16
    vocab_size: int = 458
    byte_mixin_method: Literal["cross_attn", "concat", "noop"] = "noop"
    byte_mixout_method: Literal["noop", "copy", "split"] = "noop"
    use_byte_self_attn: bool = False
    padding_in: Literal["left", "right"] = "left"
    padding_out: Literal["left", "right"] = "left"
    pull_in: bool = True
    pull_out: bool = True
    add_padded_and_pulled: bool = False
    sliding_window_tokens: int = 8
    n_layer_out: int = 1


@dataclass
class ModelDims:
    model_dim: int = 768
    byte_dim: int = 768
    token_dim: int = 768
    expansion_factor: float = 4.0


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class CrossAttention(nn.Module):
    """
    Only project bytes_per_token bytes into their one corresponding token
    --> causality or blocks are irrelevant

    But do add rotary embeddings
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len_q: int, max_seq_len_kv: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.q_w = nn.Parameter(torch.empty(hdim, dim).uniform_(-bound, bound))
        self.kv_w = nn.Parameter(torch.empty(2, hdim, dim).uniform_(-bound, bound))
        self.lambda_factor = nn.Parameter(torch.tensor(0.5))
        self.rotary_q = Rotary(head_dim, max_seq_len_q)
        self.rotary_k = Rotary(head_dim, max_seq_len_kv)
        self.c_proj = CastedLinear(hdim, dim)  # No zero init because there won't be a residual!!! TODO: check if a residaul makes sense
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, xq: Tensor, xkv: Tensor):
        Bq, Tq = xq.size(0), xq.size(1)
        Bkv, Tkv = xkv.size(0), xkv.size(1)
        assert Bq == Bkv == 1, "Must use batch size = 1 for FlexAttention"
        k, v = F.linear(xkv, self.kv_w.flatten(end_dim=1).type_as(xkv)).view(Bq, Tkv, 2 * self.num_heads, self.head_dim).chunk(2, dim=-2)
        q = F.linear(xq, self.q_w.type_as(xq)).view(Bq, Tq, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary_q(q), self.rotary_k(k)
        v = self.lambda_factor * v

        # Because we always attend from n chars to 1 token, we can re-shape, use BMM, and save use the attention mask
        chars_per_token = Tkv // Tq
        q = q.transpose(1, 2).unsqueeze(3)  # einops.rearrange(q, "b tq h d -> b h tq 1 d")
        k = k.view(k.shape[0], k.shape[2], -1, chars_per_token, k.shape[3])  # einops.rearrange(k, "b (t c) h d -> b h t c d", c=chars_per_token)
        v = v.view(v.shape[0], v.shape[2], -1, chars_per_token, v.shape[3])

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        y = torch.matmul(attn_weights, v)
        y = y.squeeze(3).transpose(1, 2)  # einops.rearrange(y, "b h tq 1 d -> b tq h d")

        y = y.contiguous().view(Bq, Tq, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        hdim = next_multiple_of_n(int(expansion_factor * dim), n=128)
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dims: ModelDims, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dims.model_dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dims.model_dim, dims.expansion_factor)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


class FlexibleEmbedding(nn.Module):
    def __init__(self, dims: ModelDims, vocab_size, byte_params: ByteHyperparameters):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dims.token_dim if byte_params.byte_mixin_method != "noop" else dims.model_dim)
        self.embed_bytes = nn.Embedding(byte_params.vocab_size, dims.byte_dim) if byte_params.byte_mixin_method != "noop" else nn.Identity()

        if byte_params.byte_mixin_method == "noop":
            self.forward = self._forward_tokens
        elif not byte_params.pull_in:
            self.forward = self._forward_bytes_padded
        elif not byte_params.add_padded_and_pulled:
            self.forward = self._forward_bytes_pulled
        else:
            self.forward = self._forward_bytes_padded_and_pulled

    def _forward_tokens(
            self,
            tokens: Tensor,
            byte_tensor: Tensor | None,
            byte_tensor_pulled: Tensor | None,
    ) -> tuple[Tensor, None]:
        return norm(self.embed_tokens(tokens)), None
    
    def _forward_bytes_padded(
            self,
            tokens: Tensor,
            byte_tensor: Tensor,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor))
        return token_embs, byte_embs


    def _forward_bytes_pulled(
            self,
            tokens: Tensor,
            byte_tensor: Tensor | None,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor_pulled))
        return token_embs, byte_embs

    def _forward_bytes_padded_and_pulled(
            self,
            tokens: Tensor,
            byte_tensor: Tensor,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor) + self.embed_bytes(byte_tensor_pulled))
        return token_embs, byte_embs


class ByteSelfAttn(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.byte_params = byte_params
        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=max(1, dim//128),
            max_seq_len=max_seq_len * byte_params.bytes_per_token,
            head_dim=128,
        ) if byte_params.use_byte_self_attn else nn.Identity()

        def sliding_window_mask(b, h, q_idx, kv_idx):
            causality = q_idx >= kv_idx
            sliding_window = q_idx - kv_idx < byte_params.sliding_window_tokens * byte_params.bytes_per_token
            return causality & sliding_window
        
        T = max_seq_len * byte_params.bytes_per_token
        self.block_mask = create_block_mask(
            mask_mod=sliding_window_mask,
            B=None,
            H=None,
            Q_LEN=T,
            KV_LEN=T,
        ) if byte_params.use_byte_self_attn else None
    
    def forward(self, byte_embs: Tensor) -> Tensor:
        if self.byte_params.use_byte_self_attn:
            byte_embs = byte_embs + self.attention(byte_embs, None, self.block_mask)
        return byte_embs


class ByteMixinNoop(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.attention = self.mixin = nn.Identity()

    def forward(self, x, *args):
        return x


class ByteMixinConcat(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.byte_params = byte_params
        self.attention = ByteSelfAttn(dims.byte_dim, max_seq_len, byte_params) if byte_params.use_byte_self_attn else nn.Identity()
        self.mixin = CastedLinear(dims.token_dim + dims.byte_dim * byte_params.bytes_per_token, dims.model_dim)

    def forward(self, tok_embs: Tensor, byte_embs: Tensor) -> Tensor:
        if self.byte_params.use_byte_self_attn:
            byte_embs = self.attention(byte_embs)
        byte_embs = einops.rearrange(byte_embs, "B (S bpt) D -> B S (bpt D)", bpt=self.byte_params.bytes_per_token)
        return norm(self.mixin(torch.cat([tok_embs, byte_embs], dim=-1)))


class ByteMixinCrossAttn(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        assert dims.byte_dim == dims.token_dim == dims.model_dim
        self.byte_params = byte_params
        self.attention = ByteSelfAttn(dims.byte_dim, max_seq_len, byte_params) if byte_params.use_byte_self_attn else nn.Identity()
        self.mixin = CrossAttention(
            dim=dims.model_dim,
            num_heads=dims.model_dim//128,
            max_seq_len_kv=max_seq_len * byte_params.bytes_per_token,
            max_seq_len_q=max_seq_len,
            head_dim=128,
        )
    
    def forward(self, token_embs: Tensor, byte_embs: Tensor) -> Tensor:
        byte_embs = self.attention(byte_embs)
        return self.mixin(xq=token_embs, xkv=byte_embs)


class ByteMixin(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        if byte_params.byte_mixin_method == "noop":
            self.mixin = ByteMixinNoop(dims, max_seq_len, byte_params)
        elif byte_params.byte_mixin_method == "cross_attn":
            self.mixin = ByteMixinCrossAttn(dims, max_seq_len, byte_params)
        elif byte_params.byte_mixin_method == "concat":
            self.mixin = ByteMixinConcat(dims, max_seq_len, byte_params)
        else:
            raise RuntimeError(f"Invalid byte mixin method: {byte_params.byte_mixin_method}")
    
    def forward(self, tok_embs: Tensor, byte_embs: Tensor) -> Tensor:
        return self.mixin(tok_embs, byte_embs)


class ByteMixoutCopy(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            ByteSelfAttn(dims.model_dim, max_seq_len, byte_params)  # use model dim at output
            for _ in range(byte_params.n_layer_out)
        ])
        self.bpt = byte_params.bytes_per_token

    def forward(self, x: Tensor) -> Tensor:
        x = einops.repeat(x, "... T D-> ... (T bpt) D", bpt=self.bpt)
        for layer in self.attention_layers:
            x = x + layer(norm(x))
        return x


class ByteMixoutSplit(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            ByteSelfAttn(dims.model_dim, max_seq_len, byte_params)  # use model dim at output
            for _ in range(byte_params.n_layer_out)
        ])
        self.bpt = byte_params.bytes_per_token

    def forward(self, x: Tensor) -> Tensor:
        x = einops.rearrange(x, "... T (bpt D) -> ... (T bpt) D", bpt=self.bpt)
        for layer in self.attention_layers:
            x = x + layer(norm(x))
        return x


class ByteMixoutNoop(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class ByteMixout(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.mixout = {
            "noop": ByteMixoutNoop,
            "copy": ByteMixoutCopy,
            "split": ByteMixoutSplit,
        }[byte_params.byte_mixout_method](dims, max_seq_len, byte_params)

    def forward(self, x: Tensor) -> Tensor:
        return self.mixout(x)


# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_heads: int,
            model_dims: ModelDims,
            max_seq_len: int,
            byte_params: ByteHyperparameters,
    ):
        super().__init__()
        self.embed = FlexibleEmbedding(dims=model_dims, vocab_size=vocab_size, byte_params=byte_params)
        self.byte_mixin = ByteMixin(
            dims=model_dims, max_seq_len=max_seq_len, byte_params=byte_params
        )
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dims.model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dims, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.byte_mixout = ByteMixout(
            dims=model_dims, max_seq_len=max_seq_len, byte_params=byte_params
        )
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        lm_head_in_dim = model_dims.model_dim if byte_params.byte_mixout_method != "split" else model_dims.model_dim // byte_params.bytes_per_token
        lm_head_out_dim = vocab_size if byte_params.byte_mixout_method == "noop" else byte_params.vocab_size
        self.lm_head = CastedLinear(lm_head_in_dim, next_multiple_of_n(lm_head_out_dim, n=128))
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        T = max_seq_len
        self.block_mask = create_block_mask(
            mask_mod=causal_mask,
            B=None,
            H=None,
            Q_LEN=T,
            KV_LEN=T,
        )

    def forward(
            self,
            toks_in: Tensor,
            bytes_padded_in: Tensor | None,
            bytes_pulled_in: Tensor | None,
            target_seq: Tensor,  # bytes or tokens
    ):
        ve = [value_embed(toks_in) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        xt, xb = self.embed(tokens=toks_in, byte_tensor=bytes_padded_in, byte_tensor_pulled=bytes_pulled_in)
        x = x0 = self.byte_mixin(xt, xb)

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, self.block_mask)
            if i < n:
                skip_connections.append(x)

        x = self.byte_mixout(x)
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1).long())
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path, dtype: torch.dtype = torch.uint16):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, f"magic number mismatch in the data .bin file: {header[0]}"
    assert header[1] == 1, f"unsupported version, expected 1 but got {header[1]}"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=dtype, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
    return tokens


def load_data_shard(file_iter):
    while True:
        try:
            file = next(file_iter)
            dtype = torch.int32 if "bytes/" in file.name else torch.uint16
            return _load_data_shard(file, dtype=dtype).to(torch.int32)
        except AssertionError:
            pass


@torch.no_grad()
def distributed_data_generator(
        filename_patterns: str | list[str],
        seq_len: int,
        batch_size: int,
        rank : int,
        world_size : int,
        byte_params: ByteHyperparameters,
        vocab_size: int = 50257,
        device: torch.device = "cpu",
        seed: int = 12345,
):
    bpt = byte_params.bytes_per_token
    # Make the byte embeddings
    if (byte_params.byte_mixin_method != "noop" and byte_params.padding_in == "left") or (byte_params.byte_mixin_method != "noop" and byte_params.padding_out == "left"):
        ttb_left_pad = make_embedding(f"ttb_{bpt}_left_pad.json", vocab_size).to(device)
    else:
        ttb_left_pad = None
    if (byte_params.byte_mixin_method != "noop" and byte_params.padding_in == "right") or (byte_params.byte_mixin_method != "noop" and byte_params.padding_out == "right"):
        ttb_right_pad = make_embedding(f"ttb_{bpt}_right_pad.json", vocab_size).to(device)
    else:
        ttb_right_pad = None

    ttb_in = ttb_left_pad if byte_params.padding_in == "left" else ttb_right_pad
    ttb_out = ttb_left_pad if byte_params.padding_out == "left" else ttb_right_pad

    pull_kwargs = dict(bytes_per_token=bpt, pad_byte=456, eot_byte=457)
    pull_in = functools.partial(pull_from_left, **pull_kwargs) if byte_params.padding_in == "left" else functools.partial(pull_from_right, **pull_kwargs)
    pull_out = functools.partial(pull_from_left, **pull_kwargs) if byte_params.padding_out == "left" else functools.partial(pull_from_right, **pull_kwargs)

    # Options for producing the data
    # Big list of functions to avoid branching, and to make it a bit less confusing for me
    # The difference is the last Ts and Fs. The mean:
    # ...(byte_in)(pull_in)_(byte_out)(pull_out)
    bpt = byte_params.bytes_per_token
    def _create_data_from_toks_TT_TT(toks: Tensor):
        bytes_padded_in = tokens_to_bytes(toks, ttb_in)
        bytes_pulled_in = pull_in(bytes_padded_in)
        bytes_padded_out = tokens_to_bytes(toks, ttb_out)
        bytes_pulled_out = pull_out(bytes_padded_out)

        toks_in = toks[:, :-1].contiguous()
        bytes_padded_in = bytes_padded_in[:, :-bpt].contiguous()
        bytes_pulled_in = bytes_pulled_in[:, :-bpt].contiguous()
        targets = bytes_pulled_out[:, bpt:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_TF_TT(toks: Tensor):
        bytes_padded_in = tokens_to_bytes(toks, ttb_in)
        bytes_pulled_in = None
        bytes_padded_out = tokens_to_bytes(toks, ttb_out)
        bytes_pulled_out = pull_out(bytes_padded_out)

        toks_in = toks[:, :-1].contiguous()
        bytes_padded_in = bytes_padded_in[:, :-bpt].contiguous()
        targets = bytes_pulled_out[:, bpt:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_TT_TF(toks: Tensor):
        bytes_padded_in = tokens_to_bytes(toks, ttb_in)
        bytes_pulled_in = pull_in(bytes_padded_in)
        bytes_padded_out = tokens_to_bytes(toks, ttb_out)
        
        toks_in = toks[:, :-1].contiguous()
        bytes_padded_in = bytes_padded_in[:, :-bpt].contiguous()
        bytes_pulled_in = bytes_pulled_in[:, :-bpt].contiguous()
        targets = bytes_padded_out[:, bpt:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_TT_FF(toks: Tensor):
        bytes_padded_in = tokens_to_bytes(toks, ttb_in)
        bytes_pulled_in = pull_in(bytes_padded_in)
        
        toks_in = toks[:, :-1].contiguous()
        bytes_padded_in = bytes_padded_in[:, :-bpt].contiguous()
        bytes_pulled_in = bytes_pulled_in[:, :-bpt].contiguous()
        targets = toks[:, 1:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_FF_TT(toks: Tensor):
        bytes_padded_in = None
        bytes_pulled_in = None
        bytes_padded_out = tokens_to_bytes(toks, ttb_out)
        bytes_pulled_out = pull_out(bytes_padded_out)

        toks_in = toks[:, :-1].contiguous()
        targets = bytes_pulled_out[:, bpt:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_FF_TF(toks: Tensor):
        bytes_padded_in = None
        bytes_pulled_in = None
        bytes_padded_out = tokens_to_bytes(toks, ttb_out)
        
        toks_in = toks[:, :-1].contiguous()
        targets = bytes_padded_out[:, bpt:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_TF_FF(toks: Tensor):
        bytes_padded_in = tokens_to_bytes(toks, ttb_in)
        bytes_pulled_in = None

        toks_in = toks[:, :-1].contiguous()
        bytes_padded_in = bytes_padded_in[:, :-bpt].contiguous()
        targets = toks[:, 1:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    def _create_data_from_toks_FF_FF(toks: Tensor):
        bytes_padded_in = None
        bytes_pulled_in = None
        
        toks_in = toks[:, :-1].contiguous()
        targets = toks[:, 1:].contiguous()
        return toks_in, bytes_padded_in, bytes_pulled_in, targets

    create_data_from_toks = {
        # (byte_in, pull_in, byte_out, pull_out): function
        (True, True, True, True): _create_data_from_toks_TT_TT,
        (True, False, True, True): _create_data_from_toks_TF_TT,
        (True, True, True, False): _create_data_from_toks_TT_TF,
        (True, True, False, False): _create_data_from_toks_TT_FF,
        (False, False, True, True): _create_data_from_toks_FF_TT,
        (False, False, True, False): _create_data_from_toks_FF_TF,
        (True, False, False, False): _create_data_from_toks_TF_FF,
        (False, False, False, False): _create_data_from_toks_FF_FF,
    }[
        (
            byte_params.byte_mixin_method != "noop",
            byte_params.pull_in,
            byte_params.byte_mixout_method != "noop",
            byte_params.pull_out,
        )
    ]

    # Find and prepare the files
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    files = sorted(Path.cwd().glob(filename_patterns[0]))
    for filename_pattern in filename_patterns[1:]:
        files.extend(sorted(Path.cwd().glob(filename_pattern)))

    random.seed(seed)  # ensure that all shards are shuffled the same way
    random.shuffle(files)

    assert batch_size % world_size == 0
    local_seq_len = seq_len + 1  # +1 because I split into inputs and targets
    local_batch_size = (batch_size * local_seq_len) // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    data, pos = load_data_shard(file_iter), 0
    while True:
        if pos + batch_size * local_seq_len + 1 >= len(data):
            newdata, pos = load_data_shard(file_iter), 0
            data = torch.cat([data, newdata])
        tokens = data[pos + rank * local_batch_size:][:local_batch_size].view(-1, local_seq_len).to(device)
        pos += batch_size * local_seq_len
        yield create_data_from_toks(tokens)


# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # data
    train_files: str | list[str] | tuple[str, ...] = ("data/tokens/train/*.bin", "fineweb100B/fineweb_train_*.bin") # input .bin to train on
    val_files_fw: str | list[str] | tuple[str, ...] = "fineweb100B/fineweb_val_*.bin" # input .bin to eval fineweb validation loss on
    val_files_fm: str | list[str] | tuple[str, ...] = "data/tokens/val/*.bin" # input .bin to eval finemath validation loss on
    val_tokens_fw: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    val_tokens_fm: int = 1024*1024
    seq_len: int = 1024  # Sequence length
    batch_size_train: int = 64  # Batch size per device
    batch_size_val: int = 64  # Batch size per device
    # optimization
    num_iterations: int = int(50_271 * 2) - 50 # number of iterations to run
    cooldown_frac: float = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size: int = 50257
    expansion_factor: float = 4.0 # expansion factor for MLP
    padding_in: Literal["left", "right"] = "left"
    padding_out: Literal["left", "right"] = "right"
    pull_in: bool = True
    pull_out: bool = True
    add_padded_and_pulled: bool = True
    byte_mixin_method: Literal["noop", "cross_attn", "concat"] = "noop"
    byte_mixout_method: Literal["noop", "copy", "split"] = "noop"
    sliding_window_tokens: int = 8
    n_layer_out: int = 1
    bytes_per_token: int = 16
    # Model dims
    model_dim: int = 1024
    byte_dim: int = 1024
    token_dim: int = 1024
    # evaluation and logging
    val_loss_every: int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint_every: int = 0  # if 0, won't save; otherwise, save every nth step>0, where n is this arg
    # other
    seed: int | None = None
    wandb_project: str | None = None
    # results
    num_params: int | None = None
    final_val_loss_fw: float | None = None
    min_val_loss_fw: float | None = None
    final_val_loss_fm: float | None = None
    min_val_loss_fm: float | None = None
    step_avg_train_time: float | None = None
    val_losses_fw: list[float] | None = None
    val_losses_fm: list[float] | None = None
    peak_mem_alloc_mb: int | None = None
    peak_mem_reserved_mb: int | None = None


def download_data():
    if not os.path.exists("fineweb100B") or len(os.listdir("fineweb100B")) < 1029:  # 1028 train, 1 val
        sp.run(["bash", "fineweb100B.sh"])
    download(tokens_or_bytes="tokens")


def make_name(args: Hyperparameters) -> str:
    name = "MoT"
    name += f"_pad-{'l' if args.padding_in == 'left' else 'r'}{'l' if args.padding_out == 'left' else 'r'}"
    name += f"_pull-{'y' if args.pull_in else 'n'}{'y' if args.pull_out else 'n'}"
    name += "_add" if args.add_padded_and_pulled else ""
    name += f"_bpt-{args.bytes_per_token}"
    name += f"_how-{args.byte_mixin_method}-{args.byte_mixout_method}"
    name += f"_nlo-{args.n_layer_out}" if args.byte_mixout_method != "noop" else ""
    name += f"_BTMdim-{args.byte_dim}-{args.token_dim}-{args.model_dim}"
    name += f"_niter-{args.num_iterations}"
    name += f"_{args.seed}"
    return name


def get_args() -> Hyperparameters:
    parser = argparse.ArgumentParser()

    # Train args
    parser.add_argument(
        "--num-iterations", type=int, default=int(50_271 * 2) - 50,  # all tokens, but 50 fewer batches because I fucked up the calculation (expected seq_len=1024, but I need 1025 to have inputs and outputs at 1024...)
        help="",
    )
    parser.add_argument(
        "--cooldown-frac", type=float, default=0.4,
        help="",
    )
    parser.add_argument(
        "--seq-len", type=int, default=1024,
        help="",
    )
    parser.add_argument(
        "--batch-size-train", type=int, default=64,
        help="Per device batch size, default=64",
    )
    parser.add_argument(
        "--batch-size-val", type=int, default=32,
        help="Per device batch size, default=32",
    )
    parser.add_argument(
        "--val-loss-every", type=int, default=125,
        help="",
    )
    parser.add_argument(
        "--save-checkpoint-every", type=int, default=0,
        help="If >0, save model every nth step>0, where n is this arg. default=0"
    )
    # Byte Args
    parser.add_argument(
        "--bytes-per-token", type=int, choices=[16, 18, 20], default=16,
        help="",
    )
    parser.add_argument(
        "--byte-mixin-method", choices=["cross_attn", "concat", "noop"], default="noop",
        help="",
    )
    parser.add_argument(
        "--byte-mixout-method", choices=["noop", "copy", "split"], default="noop",
        help="",
    )
    parser.add_argument(
        "--padding-in", choices=["left", "right"], default="left",
        help="",
    )
    parser.add_argument(
        "--padding-out", choices=["left", "right"], default="right",
        help="",
    )
    parser.add_argument(
        "--pull-in", action="store_true",
        help="",
    )
    parser.add_argument(
        "--pull-out", action="store_true",
        help="",
    )
    parser.add_argument(
        "--add-padded-and-pulled", action="store_true",
        help="",
    )
    parser.add_argument(
        "--use-byte-self-attn", action="store_true",
        help="",
    )
    parser.add_argument(
        "--n-layer-out", type=int, default=1,
        help="",
    )
    parser.add_argument(
        "--sliding-window-tokens", type=int, default=8,
        help="",
    )
    # Model dims
    parser.add_argument(
        "--model-dim", type=int, default=1024,
        help="",
    )
    parser.add_argument(
        "--byte-dim", type=int, default=1024,
        help="",
    )
    parser.add_argument(
        "--token-dim", type=int, default=1024,
        help="",
    )
    parser.add_argument(
        "--expansion-factor", type=float, default=4.0,
        help="",
    )
    # Other
    parser.add_argument(
        "--seed", type=int, default=None,
        help="The random seed. If None, not manually set. Default: None"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None,
        help="",
    )

    args = parser.parse_args()
    hps = Hyperparameters(
        num_iterations=args.num_iterations,
        cooldown_frac=args.cooldown_frac,
        seq_len=args.seq_len,
        batch_size_train=args.batch_size_train,
        batch_size_val=args.batch_size_val,
        val_loss_every=args.val_loss_every,
        save_checkpoint_every=args.save_checkpoint_every,
        add_padded_and_pulled=args.add_padded_and_pulled,
        bytes_per_token=args.bytes_per_token,
        byte_mixin_method=args.byte_mixin_method,
        byte_mixout_method=args.byte_mixout_method,
        padding_in=args.padding_in,
        padding_out=args.padding_out,
        pull_in=args.pull_in if args.byte_mixin_method != "noop" else False,
        pull_out=args.pull_out if args.byte_mixout_method != "noop" else False,
        sliding_window_tokens=args.sliding_window_tokens,
        n_layer_out=args.n_layer_out,
        model_dim=args.model_dim,
        byte_dim=args.byte_dim,
        token_dim=args.token_dim,
        expansion_factor=args.expansion_factor,
        seed=args.seed,
        wandb_project=args.wandb_project,
    )
    hps.train_files = list(hps.train_files)
    if hps.byte_mixout_method == "split":
        assert hps.model_dim % hps.bytes_per_token == 0, f"model_dim ({hps.model_dim}) must be a multiple of bytes_per_token ({hps.bytes_per_token})"
    return hps
def main():
    args = get_args()

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8 # this code is designed for 8xH100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.

    if master_process and args.wandb_project and args.num_iterations > 0:
        wandb.init(project=args.wandb_project, name=make_name(args), config=args, save_code=True)
    if master_process:
        hf_token = os.getenv("HF_TOKEN")
        assert hf_token is not None, "Please set the HF_TOKEN environment variable."
        download_data()
        val_losses_fw = []
        val_losses_fm = []

    # begin logging
    logfile = None
    if master_process:
        run_id = make_name(args)
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)
        if args.save_checkpoint_every > 0:
            create_repo(repo_id=run_id, token=hf_token, exist_ok=True)
    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s, flush=True)
                print(s, file=f)

    print0("\n\n" + repr(args) + "\n\n", console=True)

    if args.seed:
        torch.manual_seed(args.seed)
        print0(f"Set seed to {args.seed}")

    # begin by printing this file (the Python code)
    print0(code)
    print0("="*100)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("="*100)

    ########################################
    #    Construct model and optimizer     #
    ########################################
    byte_params = ByteHyperparameters(
        padding_in=args.padding_in,
        padding_out=args.padding_out,
        pull_in=args.pull_in,
        pull_out=args.pull_out,
        add_padded_and_pulled=args.add_padded_and_pulled,
        byte_mixin_method=args.byte_mixin_method,
        byte_mixout_method=args.byte_mixout_method,
        bytes_per_token=args.bytes_per_token,
        n_layer_out=args.n_layer_out,
    )
    model_dims = ModelDims(
        model_dim=args.model_dim,
        byte_dim=args.byte_dim,
        token_dim=args.token_dim,
        expansion_factor=args.expansion_factor,
    )
    model: nn.Module = GPT(
        vocab_size=args.vocab_size,
        num_layers=16,
        num_heads=8,
        max_seq_len=args.seq_len,
        model_dims=model_dims,
        byte_params=byte_params,
    ).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    args.num_params = sum(p.numel() for p in model.parameters())
    print0(f"\n\nNumber of parameters: {args.num_params:_}\n", console=True)
    if args.num_iterations <= 0:
        dist.destroy_process_group()
        return

    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    hidden_matrix_params.extend([p for p in model.byte_mixout.parameters() if p.ndim >= 2])
    embed_params = [p for n, p in model.named_parameters() if "embed" in n if p.ndim >= 2]
    if args.byte_mixin_method == "concat":
        embed_params.extend([p for p in model.byte_mixin.mixin.mixin.parameters() if p.ndim >= 2])
        hidden_matrix_params.extend([p for p in model.byte_mixin.mixin.attention.parameters() if p.ndim >= 2])
    elif args.byte_mixin_method == "cross_attn":
        hidden_matrix_params.extend([p for p in model.byte_mixin.parameters() if p.ndim >= 2])
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.1/1024**0.5), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            return (1 - x) / args.cooldown_frac

    # attention window size schedule: linearly increase
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    def get_window_size_blocks(step: int):
        x = step / args.num_iterations # progress in training
        assert 0 <= x <= 1
        # Linearly increase the block-wise sliding window size over training 128 -> 1792
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)

    model: nn.Module = torch.compile(model, dynamic=False)

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(
        filename_patterns=args.train_files,
        seq_len=args.seq_len,
        batch_size=world_size * args.batch_size_train,
        rank=rank,
        world_size=world_size,
        byte_params=byte_params,
        device=device,
        seed=args.seed,
    )
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = torch.tensor(0.0, device="cuda")
    print0("Beginning training...", console=True)
    # begin training
    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()

            # fineweb validation
            val_steps_fw = 0
            val_loader_fw = distributed_data_generator(
                filename_patterns=args.val_files_fw,
                seq_len=args.seq_len,
                batch_size=world_size * args.batch_size_val,
                rank=rank,
                world_size=world_size,
                byte_params=byte_params,
                device=device,
                seed=args.seed,
            )
            val_loss_fw = 0
            with torch.no_grad():
                while True:
                    try:
                        toks_in, bytes_padded_in, bytes_pulled_in, targets = next(val_loader_fw)
                        val_loss_fw += model(toks_in, bytes_padded_in, bytes_pulled_in, targets)
                        val_steps_fw += 1
                    except (StopIteration, RuntimeError) as e:
                        break
            val_loss_fw /= val_steps_fw
            del val_loader_fw
            dist.all_reduce(val_loss_fw, op=dist.ReduceOp.AVG)

            # finemath validation
            val_steps_fm = 0
            val_loader_fm = distributed_data_generator(
                filename_patterns=args.val_files_fm,
                seq_len=args.seq_len,
                batch_size=world_size * args.batch_size_val,
                rank=rank,
                world_size=world_size,
                byte_params=byte_params,
                device=device,
                seed=args.seed,
            )
            val_loss_fm = 0
            with torch.no_grad():
                while True:
                    try:
                        toks_in, bytes_padded_in, bytes_pulled_in, targets = next(val_loader_fm)
                        val_loss_fm += model(toks_in, bytes_padded_in, bytes_pulled_in, targets)
                        val_steps_fm += 1
                    except (StopIteration, RuntimeError):
                        break
            val_loss_fm /= val_steps_fm
            del val_loader_fm
            dist.all_reduce(val_loss_fm, op=dist.ReduceOp.AVG)

            # print the results
            print0(f"step:{step}/{train_steps} val_loss_fw:{val_loss_fw:.4f} val_loss_fm:{val_loss_fm:.4f} steps_fw:{val_steps_fw} steps_fm:{val_steps_fm} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            if master_process and args.wandb_project:
                wandb.log({"val/loss_fw": val_loss_fw, "val/loss_fm": val_loss_fm, "val/train_time": training_time_ms,  "val/step_avg_time": training_time_ms/max(step, 1)})
            if master_process:
                val_losses_fw.append(float(val_loss_fw))
                val_losses_fm.append(float(val_loss_fm))
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if master_process and step > 0 and args.save_checkpoint_every > 0 and (step % args.save_checkpoint_every == 0 or last_step):
            t0 = time.perf_counter()
            safetensors.torch.save_model(model, run_id + ".safetensors", metadata={str(k): str(v) for k, v in vars(args).items()})
            upload_file(
                path_or_fileobj=run_id + ".safetensors",
                path_in_repo="model.safetensors",
                repo_id=run_id,
                revision="main" if last_step else f"step-{step}",
                token=hf_token,
            )
            print(f"Saved checkpoint at step {step} in {int(time.perf_counter()-t0)} seconds")
        if last_step:
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        toks_in, bytes_padded_in, bytes_pulled_in, targets = next(train_loader)
        loss = model(toks_in, bytes_padded_in, bytes_pulled_in, targets)
        loss.backward()
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
        if master_process and args.wandb_project:
            wandb.log({"train/loss": loss.item(), "train/train_time": approx_training_time_ms})

    peak_mem_alloc_mb = torch.cuda.max_memory_allocated() // 1024 // 1024
    peak_mem_reserved_mb = torch.cuda.max_memory_reserved() // 1024 // 1024
    print0(f"peak memory allocated: {peak_mem_alloc_mb} MiB reserved: {peak_mem_reserved_mb} MiB", console=True)
    if master_process:
        os.makedirs("results", exist_ok=True)
        args.final_val_loss_fw = val_losses_fw[-1]
        args.min_val_loss_fw = min(val_losses_fw)
        args.final_val_loss_fm = val_losses_fm[-1]
        args.min_val_loss_fm = min(val_losses_fm)
        args.step_avg_train_time = float(approx_training_time_ms / max(step+1, 1))
        args.val_losses_fw = val_losses_fw
        args.val_losses_fm = val_losses_fm
        args.peak_mem_alloc_mb = peak_mem_alloc_mb
        args.peak_mem_reserved_mb = peak_mem_reserved_mb
        if os.path.exists("results/results.json"):
            with open("results/results.json", "r") as f:
                results: list = json.loads(f.read())
                results.append(vars(args))
        else:
            results = [vars(args)]
        with open("results/results.json", "w") as f:
            f.write(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
