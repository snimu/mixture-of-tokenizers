import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset

import transformers
from transformers import AutoConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from rotary_embedding_torch import RotaryEmbedding
from datasets import load_dataset
from typing import List, Optional, Tuple, Union, Literal
from dataclasses import dataclass
from huggingface_hub import login
from dataclasses import dataclass
from tqdm import tqdm
import logging
import time
import math
import numpy as np



SEQ_LEN = 512
HF_ACCESS_TOKEN = "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"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ["HUGGINGFACE_TOKEN"] = HF_ACCESS_TOKEN
os.environ["HF_TOKEN"] = HF_ACCESS_TOKEN

login(token=HF_ACCESS_TOKEN)



@dataclass
class ModelArgs:
    version: Literal["no_residual", "one_residual", "two_residual"]
    n_heads: int = 32
    dim: int = 2048
    intermediate_dim: int = 8192
    head_dim: int = 64
    norm_eps: float = 1e-5

    
class TokenMixByCharStreamingDataset(IterableDataset):
    def __init__(self, stream_dataset, num_char_positions=8):
        self.stream_dataset = stream_dataset
        self.max_char = num_char_positions
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=HF_ACCESS_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.leading_space_ind = ord(list(self.tokenizer.tokenize("hi there")[1])[0])

    def chr_tokenize(self, x):
        ind = ord(x)
        if ind <= 127:  # ascii
            return ind
        if ind == self.leading_space_ind:  # leading character for token, e.g. "hi _there _you", "hi Ġthere Ġyou"
            return 128
        if ind == self.tokenizer.bos_token_id:  # BOS
            return 129
        if ind == self.tokenizer.eos_token_id:  # EOS/PAD
            return 130
        else:  # unicode past 128
            return 131

    def get_tokens(self, input_text):
        tokens = self.tokenizer.tokenize(input_text)
        bpe_tokens = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=SEQ_LEN, return_tensors="pt").input_ids.squeeze(0)
        char_tokens = []
        if bpe_tokens[0] == self.tokenizer.bos_token_id:  # BOS
            char_tokens.append([129])
        for token in tokens:
            char_tokens.append([self.chr_tokenize(x) for x in list(token)])
        return char_tokens, bpe_tokens

    def create_char_matrix(self, char_tokens, seq_len=SEQ_LEN):
        # ONE EOW TOKEN IS 130 THEN 2
        # Initialize character matrices with 2: EOS/PAD
        mat = torch.zeros(seq_len, self.max_char) + 2
        for row, sublist in enumerate(char_tokens):
            # Break if no more tokens
            if row >= seq_len:
                break
            ind = 0
            for char in sublist:
                # If more characters max_char, truncate
                if ind >= self.max_char:
                    break
                mat[row][ind] = char
                ind += 1
            if ind < self.max_char:
                mat[row][ind] = 130
        return mat

    def process_single_input(self, input_text):
        char_tokens, bpe_tokens = self.get_tokens(input_text)
        # Ensure all tensors have the correct size
        bpe_tokens = bpe_tokens[:SEQ_LEN]
        char_mat = self.create_char_matrix(char_tokens, seq_len=SEQ_LEN)
        char_mat = char_mat.long()
        # t tokens, txn where n is max number of chars. each elem in txn is a char index from 132
        return bpe_tokens, char_mat


    def __iter__(self):
        for sample in self.stream_dataset:
            text = sample['text']
            char_tokens, bpe_tokens = self.get_tokens(text)

            # Ensure all tensors have the correct size
            bpe_tokens = bpe_tokens[:SEQ_LEN]
            char_mat = self.create_char_matrix(char_tokens, seq_len=SEQ_LEN)
            char_mat = char_mat.long()
            # t tokens, txn where n is max number of chars. each elem in txn is a char index from 132
            yield bpe_tokens, char_mat


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    # SwiGLU
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.intermediate_dim, bias=False)
        self.w2 = nn.Linear(args.intermediate_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.intermediate_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class TokenMixByCharBMM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim

        self.bmm_dim = args.n_heads * args.head_dim

        self.wq = nn.Linear(args.dim, self.bmm_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.bmm_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.bmm_dim, bias=False)
        self.wo = nn.Linear(self.bmm_dim, args.dim, bias=False)

        self.attention_scores = None

        # number of past token character representations blocks (c_v) each token attends to
        self.window_size = 8

    def swa_transform_explicit(self, x):
        b, t, c_v, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        padding = torch.zeros(b, self.window_size-1, c_v, d, device=device)
        x_pad = torch.cat([padding, x], dim=1)
        x_reshape = x_pad.reshape(b, t+self.window_size-1, c_v*d)
        x_unfold = x_reshape.unfold(1, self.window_size, 1)
        return x_unfold.transpose(-1,-2).reshape(b, t, c_v *self.window_size, d)

    def swa_transform(self, x):
        b, t, c_v, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = torch.cat([torch.zeros(b, self.window_size-1, c_v, d, device=device), x], dim=1)
        x = x.reshape(b, t+self.window_size-1, c_v*d)
        x = x.unfold(1, self.window_size, 1)
        return x.transpose(-1,-2).reshape(b, t, c_v *self.window_size, d)

    def forward(
        self,
        x, # normal tok embeddings b t d
        chars, # char embeddings b t c_v d
        rotary_emb_fn,
    ):
        batch_size, seq_len = x.shape[0], x.shape[1]
        c_v = chars.shape[2] # c_v: how many characters are allowed to represent each token

        xq = self.wq(x)
        xk, xv = self.wk(chars), self.wv(chars) # b t c_v d_bmm
        xk, xv = self.swa_transform(xk), self.swa_transform(xv) # b t (c_v*window) d_mm

        # Reshape to separate heads
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, c_v * self.window_size, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, c_v * self.window_size, self.n_heads, self.head_dim)

        # Apply RoPE
        # Reshape for RoPE: (b, t, h, d) -> (b, h, t, d)
        xq = xq.permute(0, 2, 1, 3)
        #xk = xk.permute(0, 3, 1, 2, 4)  # b h t (c_v*window) d_head
        xk = xk.permute(0, 3, 2, 1, 4) # b h (c_v*window) t d_head

        xq = rotary_emb_fn.rotate_queries_or_keys(xq) # b h t d_head
        xk = rotary_emb_fn.rotate_queries_or_keys(xk) # b h (c_v*window) t d_head

        xk = xk.permute(0, 1, 3, 2, 4) # (b h (c_v*window) t d_head) -> b h t (c_v*window) d_head

        # Compute attention scores
        qk = xq.unsqueeze(-2) @ xk.transpose(-1, -2) # (b h t 1 d_head) @ (b h t d_head (c_v*window)) -> (b h t 1 c_v*window)
        qk = qk / self.head_dim**.5
        qk = torch.nn.functional.softmax(qk, dim=-1)

        self.attention_scores = qk.squeeze(-2)

        # (b t c_v*window h d_head) -> (b h t (c_v*window) d_head)
        xv = xv.permute(0, 3, 1, 2, 4)

        # (b h t 1 c_v*window) @ (b h t c_v*window d_head) -> (b h t 1 d_head)
        qkv = qk @ xv
        qkv = qkv.squeeze(-2) # b h t d_head
        qkv = qkv.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim) # b t d_bmm

        return self.wo(qkv)  # b t d


class TokenMixByCharBMMBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.tok_attention = TokenMixByCharBMM(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.char_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args
        self.version = args.version

        # Version-specific parameters
        if self.version in ["two_residual", "no_residual"]:
            self.lambda_tok = torch.nn.Parameter(torch.ones(1))
            self.lambda_char = torch.nn.Parameter(torch.ones(1))

        if self.version == "two_residual":
            self.register_buffer('current_step', torch.tensor(0))

    def get_residual_scale(self):
        # scale up to 1 over the course of scaling_period training steps
        scaling_period = 5000
        return min(self.current_step.item() / scaling_period, 1.0)

    def forward(
        self,
        toks, # tok embeddings b t d
        chars, # char embeddings b t c_v d
        rotary_emb_fn
    ) -> torch.Tensor:

        if self.version == "no_residual":
            h = self.tok_attention.forward(self.attention_norm(toks), self.char_norm(chars), rotary_emb_fn)

        if self.version == "one_residual":
            h = self.tok_attention.forward(self.attention_norm(toks), self.char_norm(chars), rotary_emb_fn) + toks

        if self.version == "two_residual":
            h = self.tok_attention.forward(self.attention_norm(toks), self.char_norm(chars), rotary_emb_fn) + self.lambda_tok * toks + self.lambda_char * chars.mean(dim=-2)

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out





class CustomLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config, model_args):
        super().__init__(config)

        # Load the original LLaMA model
        self.model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name_or_path, token=HF_ACCESS_TOKEN)
        self.base_config = self.model.config

        # Create character embeddings
        self.char_embeddings = nn.Embedding(
            config.char_vocab_size,
            self.base_config.hidden_size
        )

        self.max_char = 8

        self.dataset_helper = TokenMixByCharStreamingDataset(None, num_char_positions=8)

        model_args.n_heads = self.base_config.num_attention_heads
        model_args.dim = self.base_config.hidden_size
        model_args.intermediate_dim = self.base_config.intermediate_size
        model_args.head_dim = self.base_config.head_dim
        model_args.norm_eps = self.base_config.rms_norm_eps
        # Insert TokenMixByCharBMMBlock
        self.char_token_mixer = TokenMixByCharBMMBlock(model_args)

        # Initialize weights
        # self.post_init()

        self.rotary_emb_fn = RotaryEmbedding(dim = self.base_config.head_dim // 2)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        char_ids: Optional[torch.LongTensor] = None,  # Shape: [batch_size, seq_len, char_len]
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Get token embeddings from the model's embed_tokens
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Get character embeddings
        # char_ids shape: [batch_size, seq_len, char_len]
        char_embeds = self.char_embeddings(char_ids)  # [batch_size, seq_len, char_len, hidden_size]

        # Pass through TokenMixByCharBMMBlock
        mixed_embeddings = self.char_token_mixer(
            inputs_embeds,
            char_embeds,
            self.rotary_emb_fn
        )

        # Get hidden states from transformer
        hidden_states = self.model.model(
            inputs_embeds=mixed_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state

        # Pass through the LM head to get logits
        logits = self.model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # We're not using past_key_values here
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,  # We're not returning attention weights
        )


    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        dataset_helper: TokenMixByCharStreamingDataset = None,
        greedy = False,
        monitor_probs = False
    ):
        if dataset_helper is None:
            dataset_helper = TokenMixByCharStreamingDataset(None)

        print("Processing input text:", input_text)
        # Process the input text
        input_ids, char_matrix = dataset_helper.process_single_input(input_text)

        # Find the actual length of the input (excluding padding)
        actual_length = (input_ids != dataset_helper.tokenizer.pad_token_id).sum()

        # Keep only the actual content plus some padding
        max_input_length = SEQ_LEN // 2  # Leave room for generation
        input_ids = input_ids[:min(actual_length, max_input_length)]
        char_matrix = char_matrix[:min(actual_length, max_input_length)]


        # Create attention mask for actual tokens and ensure it's Long
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Add batch dimension, ensure correct type, and move to device
        input_ids = input_ids.long().unsqueeze(0).to(self.device)
        char_matrix = char_matrix.long().unsqueeze(0).to(self.device)
        attention_mask = attention_mask.long().unsqueeze(0).to(self.device)

        # Track generated sequence
        current_token_ids = input_ids.clone()
        current_char_matrix = char_matrix.clone()
        current_attention_mask = attention_mask.clone()

        for i in range(max_new_tokens):
            # Forward pass with attention mask
            outputs = self.forward(
                input_ids=current_token_ids,
                char_ids=current_char_matrix,
                attention_mask=current_attention_mask,
            )

            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            if greedy:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)


            # monitor top probs
            if monitor_probs:
                if i < 5:  # Monitor first few tokens
                    top_probs, top_indices = torch.topk(probs, 5)
                    print(f"Top 5 token probabilities:")
                    for prob, idx in zip(top_probs[0], top_indices[0]):
                        print(f"Token: {dataset_helper.tokenizer.decode([idx])}, Prob: {prob:.4f}")


            # Get character matrix for just the new token
            next_token_text = dataset_helper.tokenizer.decode(next_token[0])

            char_tokens, _ = dataset_helper.get_tokens(next_token_text)

            if char_tokens:  # Make sure we have character tokens
                next_char_matrix = dataset_helper.create_char_matrix([char_tokens[-1]], seq_len=1).long()  # Ensure long
                next_char_matrix = next_char_matrix.unsqueeze(0).to(self.device)

                # Create attention mask value for new token (1 since it's a generated token)
                next_attention = torch.ones((1, 1), dtype=torch.long, device=self.device)

                # Update sequences
                current_token_ids = torch.cat([current_token_ids, next_token], dim=1)
                current_char_matrix = torch.cat([current_char_matrix, next_char_matrix], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, next_attention], dim=1)

            else:
                print("Warning: No char tokens generated!")

            # Check for EOS
            if next_token[0, 0].item() == self.model.config.eos_token_id:
                # print("EOS token generated, stopping")
                break

            # Check sequence length
            if current_token_ids.size(1) >= SEQ_LEN:
                print("Max sequence length reached, stopping")
                break

        # Decode the generated sequence
        generated_text = dataset_helper.tokenizer.decode(current_token_ids[0], skip_special_tokens=False)
        print("\nFinal generated text:", repr(generated_text))

        return generated_text


def run_inference(model_version, input_text):

    valid_versions = ["one_residual", "two_residual", "no_residual"]
    
    if model_version == "one_residual":
        hf_model_path = "nickcdryan/tokenmix-V1-one-residual"
    elif model_version == "two_residual":
        hf_model_path = "nickcdryan/tokenmix-V1-two-residual"
    elif model_version == "no_residual":
        hf_model_path = "nickcdryan/tokenmix-V1-no-residual"
    else:
        raise ValueError(f"Invalid model version: {model_version}. Please use one of: {', '.join(valid_versions)}")

    # Load the config first
    config = AutoConfig.from_pretrained(hf_model_path)
    
    model_args = ModelArgs(version=model_version)
    
    model = CustomLlamaModel.from_pretrained(
        hf_model_path,
        model_args,
        config=config,
        ignore_mismatched_sizes=True,
        # Prevent loading base model weights
        _fast_init=True,
    )
    
    model.to(device);
    
    dataset_helper = TokenMixByCharStreamingDataset(None)
    generated_text = model.generate(
        input_text = input_text,
        max_new_tokens=100,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        dataset_helper=dataset_helper,
        greedy=False
    )

    return generated_text


if __name__ == "__main__":
    print ("Start")
    run_inference("one_residual", "The capital of France is")



