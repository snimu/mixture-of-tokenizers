# /// script
# requires-python = "==3.12"
# dependencies = [
#   "torch",
#   "transformers",
#   "huggingface_hub[cli]",
#   "tiktoken",
#   "datasets",
#   "psutil",
# ]
# ///
import time
import argparse
import json
import random
import os
import multiprocessing as mp
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError
from time import perf_counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import psutil
import numpy as np
import torch
from torch import nn
import tiktoken
from datasets import load_dataset, arrow_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HTTPError


#####################################################################
###### GIVEN BATCH OF TOKENS, CREATE CORRESPONDING BYTES BATCH ######
#####################################################################


def load_ttb(filename: str) -> dict[int, list[int]]:
    with open(f"embeddings/{filename}", "r") as f:
        text = f.read()
    ttb = json.loads(text)
    ttb = {int(k): [int(x) for x in v] for k, v in ttb.items()}
    return ttb


def make_embedding(filename: str, vocab_size: int) -> nn.Embedding:
    dim = int(filename.split("_")[1])
    emb = nn.Embedding(vocab_size, dim)
    ttb = load_ttb(filename)
    for idx in ttb:
        emb.weight.data[idx] = torch.tensor(ttb[idx])
    emb.weight.requires_grad = False
    return emb


def tokens_to_bytes(tokens: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
    with torch.no_grad():
        byte_tensor = emb(tokens).to(torch.int64)
    if tokens.ndim == 2:
        return byte_tensor.view(byte_tensor.shape[0], -1)  # einops.rearrange(byte_tensor, "b n c -> b (n c)")
    else:
        return byte_tensor.view(-1).unsqueeze(0)  # einops.rearrange(byte_tensor, "n c -> (n c)")[None]


@torch.compile
def pull_from_right(
        byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> torch.Tensor:
    B, T = byte_tensor.size()
    T_reduced = T // bytes_per_token 
    
    # Reshape to token-level representation
    byte_tensor = byte_tensor.view(B, T_reduced, bytes_per_token)
    
    # Create masks for non-padding bytes and EOT tokens
    non_pad_mask = byte_tensor != pad_byte  # Shape: (B, T_reduced, bytes_per_token)
    is_eot_token = torch.all(byte_tensor == eot_byte, dim=2)  # Shape: (B, T_reduced)
    
    # Initialize output tensor
    pulled_tensor = torch.full_like(byte_tensor, pad_byte)
    
    # Process each batch (we still need this loop as batches can have different EOT patterns)
    for batch_idx in range(B):
        # Calculate valid bytes per token and their cumulative positions
        valid_bytes_per_token = non_pad_mask[batch_idx].sum(dim=1)
        cum_positions = torch.zeros(T_reduced + 1, dtype=torch.long, device=byte_tensor.device)
        cum_positions[1:] = torch.cumsum(valid_bytes_per_token, dim=0)
        
        # Extract all valid bytes at once using a list comprehension and single cat operation
        valid_bytes_list = [byte_tensor[batch_idx, t][non_pad_mask[batch_idx, t]] 
                           for t in range(T_reduced) if valid_bytes_per_token[t] > 0]
        
        # Skip if no valid bytes in this batch
        if not valid_bytes_list:
            continue
            
        all_valid_bytes = torch.cat(valid_bytes_list)
        
        # Find EOT positions
        eot_positions = is_eot_token[batch_idx].nonzero(as_tuple=True)[0]
        
        # For each token, find its next EOT token using searchsorted (vectorized)
        if len(eot_positions) > 0:
            token_indices = torch.arange(T_reduced, device=byte_tensor.device)
            next_eot_idx = torch.searchsorted(eot_positions, token_indices)
            
            # Convert searchsorted indices to actual EOT positions
            next_eot = torch.full((T_reduced,), T_reduced, device=byte_tensor.device)
            valid_mask = next_eot_idx < len(eot_positions)
            if valid_mask.any():
                next_eot[valid_mask] = eot_positions[next_eot_idx[valid_mask]]
        else:
            # No EOT tokens in this batch
            next_eot = torch.full((T_reduced,), T_reduced, device=byte_tensor.device)
        
        # Process each token (minimal operations in this loop now)
        for token_idx in range(T_reduced):
            # For EOT tokens, keep original bytes
            if is_eot_token[batch_idx, token_idx]:
                start_idx = cum_positions[token_idx].item()
                end_idx = cum_positions[token_idx + 1].item()
                if start_idx < end_idx:
                    token_bytes = all_valid_bytes[start_idx:end_idx]
                    pulled_tensor[batch_idx, token_idx, :len(token_bytes)] = token_bytes
                continue
            
            # For non-EOT tokens, pull bytes up to the next EOT
            start_idx = cum_positions[token_idx].item()
            next_eot_pos = next_eot[token_idx].item()
            end_idx = cum_positions[next_eot_pos].item()
            
            bytes_to_pull = min(bytes_per_token, end_idx - start_idx)
            
            if bytes_to_pull > 0 and start_idx < len(all_valid_bytes):
                pulled_tensor[batch_idx, token_idx, :bytes_to_pull] = all_valid_bytes[start_idx:start_idx + bytes_to_pull]
    
    # Reshape back to original dimensions
    return pulled_tensor.view(B, T)


@torch.compile
def pull_from_left(
        byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> torch.Tensor:
    B, T = byte_tensor.size()
    T_reduced = T // bytes_per_token 
    
    # Reshape to token-level representation
    byte_tensor = byte_tensor.view(B, T_reduced, bytes_per_token)
    
    # Create masks for non-padding bytes and EOT tokens
    non_pad_mask = byte_tensor != pad_byte  # Shape: (B, T_reduced, bytes_per_token)
    is_eot_token = torch.all(byte_tensor == eot_byte, dim=2)  # Shape: (B, T_reduced)
    
    # Initialize output tensor
    pulled_tensor = torch.full_like(byte_tensor, pad_byte)
    
    # Process each batch
    for batch_idx in range(B):
        # Calculate valid bytes per token and their cumulative positions
        valid_bytes_per_token = non_pad_mask[batch_idx].sum(dim=1)
        cum_positions = torch.zeros(T_reduced + 1, dtype=torch.long, device=byte_tensor.device)
        cum_positions[1:] = torch.cumsum(valid_bytes_per_token, dim=0)
        
        # Extract all valid bytes at once using a list comprehension and single cat operation
        valid_bytes_list = [byte_tensor[batch_idx, t][non_pad_mask[batch_idx, t]] 
                           for t in range(T_reduced) if valid_bytes_per_token[t] > 0]
        
        # Skip if no valid bytes in this batch
        if not valid_bytes_list:
            continue
            
        all_valid_bytes = torch.cat(valid_bytes_list)
        
        # Find EOT positions
        eot_positions = is_eot_token[batch_idx].nonzero(as_tuple=True)[0]
        
        # Pre-compute previous EOT indices for all token positions
        token_indices = torch.arange(T_reduced, device=byte_tensor.device)
        if len(eot_positions) > 0:
            # Use a trick with searchsorted to find the previous EOT
            prev_eot_idx = torch.searchsorted(eot_positions, token_indices, right=True) - 1
            
            # Convert to actual positions and handle no-previous-EOT case
            prev_eot = torch.full((T_reduced,), -1, device=byte_tensor.device)
            valid_mask = prev_eot_idx >= 0
            if valid_mask.any():
                prev_eot[valid_mask] = eot_positions[prev_eot_idx[valid_mask]]
        else:
            # No EOT tokens in this batch
            prev_eot = torch.full((T_reduced,), -1, device=byte_tensor.device)
        
        # Process each token
        for token_idx in range(T_reduced):
            current_start = cum_positions[token_idx].item()
            current_end = cum_positions[token_idx + 1].item()
            
            # If this is an EOT token, only use its own bytes
            if is_eot_token[batch_idx, token_idx]:
                valid_bytes = current_end - current_start
                if valid_bytes > 0:
                    # Place bytes right-aligned (left-padded)
                    pulled_tensor[batch_idx, token_idx, -valid_bytes:] = all_valid_bytes[current_start:current_end]
                continue
            
            # Determine where to start pulling bytes from
            prev_eot_pos = prev_eot[token_idx].item()
            if prev_eot_pos >= 0:
                # Start from token after the previous EOT
                pull_start = cum_positions[prev_eot_pos + 1].item()
            else:
                # No previous EOT, start from beginning
                pull_start = 0
            
            # Determine the byte range to pull
            total_bytes = current_end - pull_start
            
            if total_bytes <= bytes_per_token:
                # If we have fewer bytes than capacity, use all of them
                bytes_to_use = all_valid_bytes[pull_start:current_end]
                # Place bytes right-aligned (left-padded)
                pulled_tensor[batch_idx, token_idx, -len(bytes_to_use):] = bytes_to_use
            else:
                # If we have more bytes than capacity, take the rightmost bytes
                bytes_to_use = all_valid_bytes[current_end - bytes_per_token:current_end]
                # Place all bytes (fills the token completely)
                pulled_tensor[batch_idx, token_idx, -bytes_per_token:] = bytes_to_use
    
    # Reshape back to original dimensions
    return pulled_tensor.view(B, T)


def create_batch(
        tokens: torch.Tensor,
        bytes_per_token: int,
        pad_byte: int,
        eot_byte: int,
        tokens_to_bytes_right_pad: nn.Embedding,
        tokens_to_bytes_left_pad: nn.Embedding,
) -> torch.Tensor:
    B, T = tokens.size()
    byte_tensor_left_padded = tokens_to_bytes(tokens, tokens_to_bytes_left_pad)
    byte_tensor_pulled_from_left = pull_from_left(byte_tensor_left_padded, bytes_per_token, pad_byte, eot_byte)
    byte_tensor_right_padded = tokens_to_bytes(tokens, tokens_to_bytes_right_pad)
    byte_tensor_pulled_from_right = pull_from_right(byte_tensor_right_padded, bytes_per_token, pad_byte, eot_byte)
    full_tensor = torch.cat([
            tokens.unsqueeze(-1),
            byte_tensor_left_padded.view(B, T, bytes_per_token),
            byte_tensor_pulled_from_left.view(B, T, bytes_per_token),
            byte_tensor_right_padded.view(B, T, bytes_per_token),
            byte_tensor_pulled_from_right.view(B, T, bytes_per_token),
        ],
        dim=-1,
    )
    return full_tensor


################################################
###### CREATE CONSCIOUSLY CREATED BATCHES ######
################################################

# The goal:
#   - Download all samples from finemath
#   - Tokenize them & create_batch them
#   - Filter out any sample with sequence-length > seq_len (e.g. 1024)
#   - Assuming fineweb-edu-100BT is already downloaded & tokenized, use a dataloader to fill the finemath samples up to seq_len (with a eot token between)
#   - Turn the rest of the fineweb-edu-100BT tokens into their own batches with create_batch


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, shuffle: bool = False):
    files = sorted(Path.cwd().glob(filename_pattern))
    if shuffle:
        random.shuffle(files)
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    while True:
        try:
            yield _load_data_shard(next(file_iter))
        except (RuntimeError, StopIteration):
            break


def upload_with_backoff(api: HfApi, batch: torch.Tensor, filename: str, repo_id: str):
    save_file(f"data/{filename}", batch)
    sleep_time = 10
    for i in range(5):
        try:
            api.upload_file(path_or_fileobj=f"data/{filename}", path_in_repo=filename, repo_id=repo_id, repo_type="dataset")
            break
        except (HTTPError, ConnectionError, ProtocolError) as e:
            print(f"Upload failed with error {e}. Retrying in 10 seconds...")
            if i < 5:
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                raise e
    os.remove(f"data/{filename}")  # Delete the file after uploading it


def save_file(path: str, data: torch.Tensor):
    # When saving:
    with open(path, "wb") as f:
        header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
        header[0] = 20240520
        header[1] = 1
        header[2] = data.numel() # number of tokens after the 256*4 bytes of header
        # construct the data (numpy array of tokens)
        toks_np = np.array(data, dtype=np.int32)
        # write to file
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(toks_np.tobytes())


def verify_data(path: str, data: torch.Tensor, B, T, bytes_per_token):
    save_file(path, data)

    with Path(path).open("rb", buffering=0) as f:
        tokens = torch.empty((B, T, 1 + bytes_per_token * 4), dtype=torch.int32, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
    assert data.shape == tokens.shape, f"{data.shape=} vs {tokens.shape=}"
    assert (data == tokens).all(), f"{len(torch.where(data != tokens)[0])} tokens mismatch"
    os.remove(path)


def optional_print(s: str, verbose: bool = True, **kwargs):
    if verbose:
        print(s, **kwargs)


def create_finemath_data(
        datafile: str,
        skip_val_batches: bool = False,
        count_batches: bool = False,
        verbose: bool = True,
        B: int = 1024,
        T: int = 1024,
        bytes_per_token: int = 16,
        pad_byte: int = 456,
        eot_byte: int = 457,
        vocab_size: int = 50257,
        num_fm_val_batches: int = 1,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
):
    token=os.getenv("HF_TOKEN")
    assert token is not None, "Please set the HF_TOKEN environment variable."

    optional_print(f"\n{B=} {T=} {bytes_per_token=} {pad_byte=} {eot_byte=} {vocab_size=} {num_fm_val_batches=}\n", verbose)

    os.makedirs("data", exist_ok=True)

    eot_token = vocab_size - 1
    optional_print("Creating tokens-to-bytes-embeddings...", verbose)
    tokens_to_bytes_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    tokens_to_bytes_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)
    optional_print("Setting up tiktoken encoding...", verbose)
    encoding = tiktoken.encoding_for_model("gpt-2")
    optional_print("Setting up fineweb dataloader...", verbose)
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin", shuffle=True)  # shuffle so that different threads still use different data (at least random data, possibly with overlap)
    tokens_fw = next(dl)

    # Download, tokenize, and save the finemath data, and fill it up to T with random fineweb samples
    optional_print("Setting up HF API...", verbose)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    batch = []
    num_fm_tokens_train = 0
    num_fw_tokens_train = 0
    num_fm_tokens_val = 0

    with open(datafile, "r") as f:
        texts = json.loads(f.read())

    if count_batches:
        optional_print("Counting batches...", verbose)
        num_batches = len(texts) // B  # One entry in the batch per line
        optional_print(f"Number of batches: {num_batches}", verbose)
        return

    optional_print("Starting data creation...", verbose)
    is_batch_start = True
    batch_num = 0
    executor = ThreadPoolExecutor(max_workers=1)
    futures = []
    t0 = perf_counter()
    t0_global = perf_counter()
    for text in texts:
        is_val_batch = batch_num < num_fm_val_batches
        if is_val_batch and skip_val_batches:
            continue
        if is_batch_start and is_val_batch:
            optional_print(f"finemath val batch {batch_num}...", verbose, end="", flush=True)
        elif is_batch_start:
            optional_print(f"finemath train batch {batch_num - num_fm_val_batches}...", verbose, end="", flush=True)
        if is_val_batch:
            filename = f"val_batch_finemath_{batch_num}.bin"
        else:
            filename = f"train_batch_fm_{batch_num - num_fm_val_batches}.bin"

        tokens_fm = torch.tensor(encoding.encode(text, disallowed_special=()), dtype=torch.int32)
        tokens_fm = tokens_fm[:T]

        # The sample will be filled to T with a random fineweb slice;
        # There has to be an EOT token between them.
        # Exception: first N batches, which will be the finemath-validation set
        if is_val_batch:
            tokens = torch.empty((T,), dtype=torch.int32).fill_(eot_token)
            tokens[:len(tokens_fm)] = tokens_fm
            batch.append(tokens.tolist())
            num_fm_tokens_val += len(torch.where(tokens != eot_token))
        else:
            num_fm_tokens_train += len(tokens_fm)
            if len(tokens_fm) < T:  # Append a single eot token to separate finemath from fineweb
                tokens_fm = torch.cat([tokens_fm, torch.empty((1,), dtype=torch.int32).fill_(eot_token)])
            if len(tokens_fm) < T:  # Only a single eot token was appended, fill up the rest with fineweb
                num_tokens_missing = T - len(tokens_fm)  # 0 <= num_tokens_missing <= T, see condition above
                while len(tokens_fw) < num_tokens_missing:
                    tokens_fw = torch.cat([tokens_fw, next(dl)])

                fillup_tokens, tokens_fw = tokens_fw[:num_tokens_missing], tokens_fw[num_tokens_missing:]
                tokens_fm = torch.cat([tokens_fm, fillup_tokens.to(tokens_fm.dtype)])
                num_fw_tokens_train += len(fillup_tokens)
            batch.append(tokens_fm.tolist())

        # Save every B samples; a.k.a. every batch
        if len(batch) == B:
            for future in futures:
                future.result()
            futures = []
            batch = create_batch(
                tokens=torch.tensor(batch, dtype=torch.int32),
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            if batch_num % 100 == 0:
                verify_data(f"data/{filename}", batch, B, T, bytes_per_token)
            futures.append(executor.submit(upload_with_backoff, api, batch, filename, repo_id))
            time_taken_step = perf_counter() - t0
            time_taken_global = perf_counter() - t0_global
            t0 = perf_counter()
            optional_print(f"{(batch_num+1)*B*T:_} tokens done in {round(time_taken_step):_}s ({round(time_taken_global):_}s total)", verbose)
            batch = []
            is_batch_start = True
            batch_num += 1
        else:
            is_batch_start = False
    

def create_fineweb_data(
        from_batch: int = 0,
        to_batch: int = -1,
        skip_val_batches: bool = False,
        count_batches: bool = False,
        verbose: bool = True,
        B: int = 1024,
        T: int = 1024,
        bytes_per_token: int = 16,
        pad_byte: int = 456,
        eot_byte: int = 457,
        vocab_size: int = 50257,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
):
    token=os.getenv("HF_TOKEN")
    assert token is not None, "Please set the HF_TOKEN environment variable."

    optional_print("FINEWEB DATA CREATION", verbose)
    optional_print(f"\n{B=} {T=} {bytes_per_token=} {pad_byte=} {eot_byte=} {vocab_size=}\n", verbose)
    os.makedirs("data", exist_ok=True)

    optional_print("Creating tokens-to-bytes-embeddings...", verbose)
    tokens_to_bytes_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    tokens_to_bytes_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)
    optional_print("Setting up fineweb dataloader...", verbose)
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin", shuffle=False)
    if count_batches:
        optional_print("Counting batches...", verbose)
        num_tokens = 0
        for tokens in dl:
            num_tokens += len(tokens)
        num_batches = num_tokens // (B*T)
        optional_print(f"Number of batches: {num_batches}", verbose)
        return
            
    tokens_fw = next(dl)

    batch_num = 0
    futures = []
    executor = ThreadPoolExecutor(max_workers=1)
    t0 = perf_counter()
    t0_global = perf_counter()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    num_fw_tokens_train = 0
    num_fw_tokens_val = 0

    # Now, turn the rest of the fineweb-edu-100BT tokens into their own batches with create_batch
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        if len(tokens_fw) < B*T:
            continue
        for i in range(0, len(tokens_fw) // B*T + 1, B*T):
            if len(tokens_fw[i:]) < B*T:
                break
            batch_num += 1
            if batch_num < from_batch:  # Skip non-val-batches before the from_batch
                continue
            if batch_num > to_batch:  # Skip non-val-batches after the to_batch
                break
            filename = f"train_batch_fw_{batch_num}.bin"
            if os.path.exists(f"data/{filename}"):
                optional_print(f"Skipping {filename} because it already exists...", verbose)
                continue
            for future in futures:
                future.result()
            futures = []
            batch = tokens_fw[i:i+B*T].view(B, T).to(torch.int32)
            batch = create_batch(
                tokens=batch,
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            if batch_num % 100 == 0:
                verify_data(f"data/{filename}", batch, B, T, bytes_per_token)
            futures.append(executor.submit(upload_with_backoff, api, batch, filename, repo_id))
            time_taken_step = perf_counter() - t0
            time_taken_global = perf_counter() - t0_global
            t0 = perf_counter()
            optional_print(f"{(batch_num+1)*B*T:_} tokens done in {round(time_taken_step):_}s ({round(time_taken_global):_}s total)", verbose)
            num_fw_tokens_train += B*T

    # For fineweb, just use the validation set by karpathy
    if not skip_val_batches:
        dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
        tokens_fw = None
        batch_num = 0
        for new_tokens in dl:
            tokens_fw = torch.cat([tokens_fw, new_tokens]) if tokens_fw else new_tokens
            if len(tokens_fw) < B*T:
                continue
            for i in range(0, len(tokens_fw) // B*T + 1, B*T):
                if len(tokens_fw[i:]) < B*T:
                    break
                if len(futures) == 5:
                    for future in futures:
                        future.result()
                    futures = []
                batch = tokens_fw[i:i+B*T].view(B, T).to(torch.int32)
                batch = create_batch(
                    tokens=batch,
                    bytes_per_token=bytes_per_token,
                    pad_byte=pad_byte,
                    eot_byte=eot_byte,
                    tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                    tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
                )
                filename = f"val_batch_fineweb_{batch_num}.bin"
                if batch_num % 100 == 0:
                    verify_data(f"data/{filename}", batch, B, T, bytes_per_token)
                futures.append(executor.submit(upload_with_backoff, api, batch, filename, repo_id))
                num_fw_tokens_val += B*T
                batch_num += 1
    # Wait for all uploads to finish
    for future in futures:
        future.result()
    futures = []
    executor.shutdown()

    optional_print(f"fineweb: {num_fw_tokens_train=}", verbose)
    optional_print(f"fineweb: {num_fw_tokens_val=}", verbose)


#####################
###### TESTING ######
#####################


def _print_batch():
    B, T = 1024, 1024
    tokens = torch.randint(0, 50256, (B, T), dtype=torch.int32)
    eot_positions = torch.rand(B, T)
    tokens = torch.where(eot_positions > 0.8, 50256, tokens)
    bytes_per_token = 16
    pad_byte = 456
    eot_byte = 457
    vocab_size = 50257
    bytes_to_tokens_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)
    bytes_to_tokens_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    byte_tensor_left_pad = tokens_to_bytes(tokens, bytes_to_tokens_left_pad)
    byte_tensor_right_pad = tokens_to_bytes(tokens, bytes_to_tokens_right_pad)
    batch = create_batch(tokens, bytes_per_token, pad_byte, eot_byte, bytes_to_tokens_right_pad, bytes_to_tokens_left_pad)
    print(f"{tokens.shape=}\n{batch.shape=}\n\nTOKENS")
    print(tokens)
    print("\n\nBYTES LEFT PAD")
    print(byte_tensor_left_pad.view(B, T, bytes_per_token))
    print("\n\nBYTES RIGHT PAD")
    print(byte_tensor_right_pad.view(B, T, bytes_per_token))
    print("\n\nBATCH-TOKENS")
    print(batch[:, :, 0])
    print("\n\nBATCH-BYTES LEFT PAD")
    print(batch[:, :, 1:17])
    print("\n\nBATCH-BYTES LEFT PULLED")
    print(batch[:, :, 17:33])
    print("\n\nBATCH-BYTES RIGHT PAD")
    print(batch[:, :, 33:49])
    print("\n\nBATCH-BYTES RIGHT PULLED")
    print(batch[:, :, 49:65])
    print("\n\n")


def main():
    # Finemath: 6542 batches (at B=1024, T=1024 --> 6,859,784,192 tokens)
    # Fineweb: 85067 batches (at B=1024, T=1024 --> 89,199,214,592 tokens)
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-batch", type=int, default=0)
    parser.add_argument("--to-batch", type=int, default=-1)
    parser.add_argument("--skip-fm-val-batches", action="store_true")
    parser.add_argument("--skip-fw-val-batches", action="store_true")
    parser.add_argument("--no-fm", action="store_true")
    parser.add_argument("--no-fw", action="store_true")
    parser.add_argument("--nproc", type=int, default=-1)
    parser.add_argument("--count-batches", action="store_true")
    args = parser.parse_args()

    B = 1024

    if args.count_batches:
        assert args.nproc == 1, "--count-batches only works with --nproc=1"

    # Prepare finemath data
    if not args.no_fm:
        data: arrow_dataset.Dataset = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=8)
        data.sort("text")
        texts = list(data["text"])
        # I will split the texts into batches.
        # I don't want them to be alphabetical,
        # but they must be repeatable (from-batch & to-batch should always do the same thing).
        # So shuffle them with a given seed.
        random.seed(123456)
        random.shuffle(texts)
        args.to_batch = args.to_batch if args.to_batch > 0 else len(texts)

    # Don't fuck with multiprocessing if I don't have to.
    if args.nproc == 1:
        if not args.no_fm:
            filename = "finemath-4plus.txt"
            with open(filename, "w") as f:
                f.write(json.dumps(texts[B*args.from_batch:B*args.to_batch]))
            create_finemath_data(filename, args.skip_fm_val_batches, args.count_batches)
        if not args.no_fw:
            create_fineweb_data(args.from_batch, args.to_batch, args.skip_fw_val_batches, args.count_batches)
    else:
        assert args.to_batch > 0  # TODO: just set this to the correct value for fineweb & finemath
        nproc = (psutil.cpu_count(logical=True) - 2)
        nproc = min(args.nproc, nproc) if args.nproc > 1 else nproc  # Set nproc to -1 to get the maximum out

        # TODO: treat val-batches independently
        interval, remainder = divmod(abs(args.from_batch - args.to_batch), nproc)
        from_to  = [(args.from_batch + i*interval, args.from_batch + (i+1)*interval) for i in range(nproc)]

        if not args.no_fm:
            # Split the data into chunks & save it -> can be used in different threads
            text_chunks = [texts[B*from_to[i][0]:B*from_to[i][1]] for i in range(nproc)]
            text_chunk_names = [f"finemath-4plus-{i}.txt" for i in range(nproc)]
            for i, text in enumerate(text_chunks):
                filename = f"finemath-4plus-{i}.txt"
                if not Path(filename).exists():
                    with open(filename, "w") as f:
                        f.write(json.dumps(text))

            # datafile, skip_val_batches, count_batches, verbose
            args = [(text_chunk_names[i], args.skip_fm_val_batches, False, (i==0)) for i in range(nproc)]
            with mp.Pool(nproc) as pool:
                pool.starmap(create_finemath_data, args)

            if remainder > 0:
                remainder_start = args.from_batch + nproc*interval
                remainder_end = remainder_start + remainder
                create_finemath_data(remainder_start, remainder_end, args.skip_fm_val_batches, args.count_batches)
        if not args.no_fw:
            # from_batch, to_batch, skip_val_batches, count_batches, verbose
            args = [(from_to[i][0], from_to[i][1], True, False, (i==0)) for i in range(nproc)]
            with mp.Pool(nproc) as pool:
                pool.starmap(create_fineweb_data, args)

            if remainder > 0:
                remainder_start = args.from_batch + nproc*interval
                remainder_end = remainder_start + remainder
                create_fineweb_data(remainder_start, remainder_end, True, args.count_batches)


if __name__ == "__main__":
    main()
