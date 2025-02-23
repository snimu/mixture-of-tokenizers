# Devlog

The devlog for my nanogpt speedrun attempt.

## 2025-02-23

- With CrossAttention (implemented as BMM), but no self-attention on the chars:
  - the per-step speed is cs 7ms slower than the original (~108ms vs ~101ms in final step)
  - At the same time, the final loss is 3.3092 vs 3.2799 for the original (run once)
- When including the self-attention on the chars:
  - the per-step speed is now ~523ms... much slower!
  - the final loss is 3.3087; the tiniest bit better than without char-self-attention.

So what's wrong? I have a few ideas:

- The lack of residual connections to the token embeddings might be a problem
- Using SelfAttention on the chars (16x sequence length) might be bad with the extreme sequence lengths that nanogpt is using
  - Before, I had to reduce the sequence length in the validation set; otherwise, int32 indexing doesn't work anymore, and Triton doesn't support int64

Here's the current forward pass:

```python
    def forward(self, input_seq: Tensor, input_char_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_block_masks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = norm(self.token_embed(input_seq)[None]) # use of norm here by @Grad62304977

        # Incorporate byte-level info into tokens
        xc = norm(self.char_embed(input_char_seq))
        if self.use_mot_self_attn:
            char_bm = self.create_mot_self_attn_mask(input_char_seq, chars_per_token=self.chars_per_token)
            xc = xc + self.char_self_attn(xc, None, char_bm)
        x = self.mot_cross_attn(xq=x, xkv=xc)
        # Project into model dim
        # Consider the result of this the actual embedding
        # we have just used a more complex embedding model than an FC layer
        # Therefore, x0 is determined here
        x = x0 = norm(self.up_proj(x))

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss
```

What are some solutions?

- How about we change `x = self.mot_cross_attn(xq=x, xkv=xc)` to `x = x + self.mot_cross_attn(xq=x, xkv=xc)`?
  - actually, this but with learned weights
- to reduce the char-sequence length, we could reshape the 16 chars per token into 4x4, then sum
  - how does the model know the order of the characters?
  - maybe add a learned vector to the char-embeds before the reshape?
  - so: (b, 16\*t, d) -> (b, 16, d, t) &rarr; add (16,) parameter to last dim &rarr; (b, 16, d, t) &rarr; (b, t, 4, 4, d) &rarr; sum(dim=2) &rarr; (b, t, 4, d) &rarr; (b, 4\*t, d)
- Reduce the sliding window size
  - but wouldn't that make the final perf even worse?
- use [mamba](https://github.com/state-spaces/mamba) on the chars?

But first, test some assumptions:

1. Train again with char-dim of 768 like the model-dim &rarr; does loss improve?
2. Train with larger and smaller sliding window sizes

**Changing sliding window size.**

Repetition: results with sliding window size of 16 (default)

- Per-step time: 523ms
- Final loss: 3.3087

Results with sliding window size of 4:

- Per-step time: 525ms
  - &rarr; this doesn't even save any time...
  - Probably because the block size is 128.
  - But wait! It's 16 tokens, so 256 chars! That cannot be the reason.
  - Does FlexAttention simply suck?
  - Let's wait for the larger sliding window size before jumping to conclusions.
- Final loss: 3.3115
  - At least that's lower, so this part makes sense.

Results with sliding window size of 64:

- Per-step time: 525ms
  - &rarr; why the fuck does this not make any difference?
  - This makes it seem to me like there is some problem with the block creation.
  - I hope it is; that would explain the huge slowdown.
- Final loss: 3.3104
  - Better than before, but worse than the MoT-less baseline.
  - Problem might be from the lack of cross-document attention blocking.

**Next steps:**

- [x] Use [mamba](https://github.com/state-spaces/mamba) on the chars
- [x] Increase char-dim to 768

**Mamba.**

There is some kind of import error:

```bash
Traceback (most recent call last):
  File "/root/mixture-of-tokenizers/modded-nanogpt/train_gpt.py", line 28, in <module>
    from mamba_ssm import Mamba2
  File "/usr/local/lib/python3.10/dist-packages/mamba_ssm/init.py", line 3, in <module>
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
  File "/usr/local/lib/python3.10/dist-packages/mamba_ssm/ops/selective_scan_interface.py", line 18, in <module>
    import selective_scan_cuda
ImportError: /usr/local/lib/python3.10/dist-packages/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: ZN3c107WarningC1ESt7variantIJNS011UserWarningENS0_18DeprecationWarningEEERKNS_14SourceLocationESsb
```

I'm not dealing with that shit. Let's move on.

**Char-dim.**

Dim 768, no self-attn on chars:

- Per-step time: 134ms
- Final loss: 3.3063

Dim 768, self-attn on chars:

- Per-step time: 586ms
- Final loss: 3.

**Next steps:**

- [x] No char-dim; just use model dim
- [x] Weighted residual from token embedding to mixed token and byte embeddings; hold up no, this already happens in the first transformer layer
- [x] Create sliding-window mask once outside the model, just pass it.
