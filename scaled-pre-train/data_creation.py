# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "tiktoken",
#   "datasets",
#   "huggingface_hub[cli]",
#   "transformers",
#   "psutil",
#   "tqdm",
# ]
# ///

import time
import argparse
import random
import psutil
import json
import os
import functools
import concurrent.futures
import multiprocessing as mp
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError
from time import perf_counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import tiktoken
from datasets import load_dataset, arrow_dataset
from huggingface_hub import HfApi, hf_hub_download
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
    header = torch.from_file(str(file), False, 3, dtype=torch.int32) # header is 3 numbers of int32
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
        random.seed(12345)
        random.shuffle(files)
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    while True:
        try:
            yield _load_data_shard(next(file_iter))
        except AssertionError:
            continue
        except StopIteration:
            break


def upload_with_backoff(api: HfApi, batch: torch.Tensor, filename: str, repo_id: str, path_in_repo: str = "bytes"):
    save_file(f"data/{filename}", batch)
    sleep_time = 10
    for i in range(5):
        try:
            api.upload_file(
                path_or_fileobj=f"data/{filename}",
                path_in_repo=f"{path_in_repo}/{filename}",
                repo_id=repo_id,
                repo_type="dataset",
            )
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


def load_file(path: str) -> torch.Tensor:  # Thank you Gemini Pro 2.5 :)
    file_path = Path(path) # Use Path object for robustness

    # Read header using torch.from_file on the path
    try:
        # Reads first 256 * 4 bytes = 1024 bytes
        header = torch.from_file(str(file_path), shared=False, size=256, dtype=torch.int32)
    except RuntimeError as e:
        print(f"Error reading header from {file_path}: {e}")
        raise e

    # Verify header (use constants or variables for clarity if preferred)
    magic_number = 20240520
    version = 1
    assert header[0] == magic_number, f"magic number mismatch in data file {file_path} (expected {magic_number}, got {header[0]})"
    assert header[1] == version, f"unsupported version in data file {file_path} (expected {version}, got {header[1]})"

    # Get number of elements (saved as int32)
    num_elements = int(header[2])

    # Open file handle to read the data part
    with file_path.open("rb") as f:
        # Seek past the header (256 int32 values * 4 bytes/int32)
        f.seek(256 * 4)

        # Create tensor with the correct dtype used during saving (int32)
        # pin_memory=True is optional, depends if you move to GPU immediately
        data_tensor = torch.empty(num_elements, dtype=torch.int32, pin_memory=True)

        # Use readinto for efficiency - read directly into the tensor's buffer
        # We need to view the numpy array as bytes for readinto
        buffer = data_tensor.numpy().view(np.byte)
        nbytes_read = f.readinto(buffer)

        # Verify number of bytes read
        expected_bytes = num_elements * data_tensor.element_size() # element_size for int32 is 4
        assert nbytes_read == expected_bytes, \
            f"number of bytes read ({nbytes_read}) does not match header claim ({expected_bytes} bytes for {num_elements} int32 elements) in {file_path}"

    return data_tensor


def verify_data(path: str, data: torch.Tensor, B, T, bytes_per_token):
    save_file(path, data)

    with Path(path).open("rb", buffering=0) as f:
        tokens = torch.empty((B, T, 1 + bytes_per_token * 4), dtype=torch.int32, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
    assert data.shape == tokens.shape, f"{data.shape=} vs {tokens.shape=}"
    assert (data == tokens).all(), f"{len(torch.where(data != tokens)[0])} tokens mismatch"
    os.remove(path)


def download_tokens(
        B: int = 1024,
        T: int = 1024,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
):
    hf_token=os.getenv("HF_TOKEN")
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
    download = functools.partial(hf_hub_download, local_dir="data", repo_type="dataset", token=hf_token)
    
    allfiles = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    trainfiles = sorted([f for f in allfiles if f.startswith("tokens/train/")])
    valfiles = sorted([f for f in allfiles if f.startswith("tokens/val/")])

    assert trainfiles and valfiles, "No train or val files found in the repo. Did you run create_and_upload_data()?"

    # Download train batches
    with mp.Pool(processes=min(len(trainfiles), psutil.cpu_count()-2)) as pool:
        pool.starmap(download, [(repo_id, f) for f in trainfiles])

    # Now, split them into individual batches
    # Deterministic because the files are sorted
    batch_num = 0
    buffer = None
    for filename in trainfiles:
        filename = filename.split("/")[-1]
        loaded = load_file(f"data/tokens/train/{filename}").view(-1, T)
        buffer = loaded if buffer is None else torch.cat([buffer, loaded])
        while len(buffer) >= B:
            batch = buffer[:B]
            buffer = buffer[B:]
            save_file(f"data/fm_toks_train_batch_{batch_num}.bin", batch)
            batch_num += 1
        os.remove(f"data/tokens/train/{filename}")
    
    # Download the single val batch & move it to the right place
    download(repo_id, valfiles[0])
    os.rename(f"data/{valfiles[0]}", f"data/{valfiles[0].split('/')[-1]}")
    
    return len(trainfiles)


def tokenize_finemath(
        B: int = 1024,
        T: int = 1024,
        vocab_size: int = 50257,
        num_fm_val_batches: int = 1,
        overlap: int = 128,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
) -> tuple[int, bool]:  # (num_train_batches, pre_existed)
    hf_token=os.getenv("HF_TOKEN")
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)

    print(f"\n{B=} {T=} {vocab_size=} {num_fm_val_batches=}\n")
    t0 = perf_counter()
    os.makedirs("data", exist_ok=True)

    print("Checking for existing finemath train batches on HF...")
    allfiles = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    trainfiles = [f for f in allfiles if f.startswith("tokens/train/")]
    on_hf = len(trainfiles) > 0
    print("Checking for existing finemath train batches on device...")
    existing = list(Path.cwd().glob("data/fm_toks_train_batch*.bin"))
    on_device = len(existing) > 0
    if on_hf and not on_device:
        print("Found finemath train batches on HF. Downloading them to device...")
        download_tokens(repo_id=repo_id)
    if on_hf or on_device:
        return len(existing), on_hf

    print("Nothing found. Creating new finemath train batches...")
    eot_token = vocab_size - 1
    encoding = tiktoken.encoding_for_model("gpt-2")
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin", shuffle=True)
    tokens_fw = next(dl)
    buffer = []
    print("Loading finemath train batches from HF...")
    data: arrow_dataset.Dataset = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=psutil.cpu_count())
    num_fw_tokens_train = 0
    num_fm_tokens_train = 0
    num_fm_tokens_val = 0
    batch_num = 0
    futures = []

    print("Starting loop...")
    for _ in tqdm(range(len(data) // B), total=len(data) // B):
        texts = data[batch_num * B : (batch_num + 1) * B]["text"]
        tokens = encoding.encode_batch(texts, disallowed_special=())
        if len(tokens) + len(buffer) < B:
            break
        for i in range(len(tokens)):
            while len(tokens[i]) > T:
                sample, tokens[i] = tokens[i][:T], tokens[i][T-overlap:]
                buffer.append(sample)
                num_fm_tokens_train += len(sample)

            if len(tokens[i]) == T:
                buffer.append(tokens[i])
                num_fm_tokens_train += len(tokens[i])
                continue

            if len(tokens[i]) < T:
                num_missing = T - len(tokens[i])
                if batch_num < num_fm_val_batches:
                    buffer.append(tokens[i] + [eot_token] * num_missing)
                    num_fm_tokens_val += len(tokens[i])
                else:
                    while len(tokens_fw) < num_missing:
                        tokens_fw = torch.cat([tokens_fw, next(dl)])
                    fillup_tokens, tokens_fw = tokens_fw[:num_missing].tolist(), tokens_fw[num_missing:]
                    if not (tokens[i][-1] == eot_token or fillup_tokens[0] == eot_token):
                        fillup_tokens[0] = eot_token
                    buffer.append(tokens[i] + fillup_tokens)
                    num_fw_tokens_train += len(fillup_tokens)
                    num_fm_tokens_train += len(tokens[i])
        while len(buffer) >= B:
            if batch_num % 5 == 0:
                for future in futures:
                    future.result()
                futures = []
            if batch_num < num_fm_val_batches:
                filename = f"fm_toks_val_batch_{batch_num}.bin"
            else:
                filename = f"fm_toks_train_batch_{batch_num - num_fm_val_batches}.bin"
            batch, buffer = torch.tensor(buffer[:B], dtype=torch.int32), buffer[B:]
            save_file(f"data/{filename}", batch)
            batch_num += 1

    num_train_batches = len(list(Path.cwd().glob("data/fm_toks_train_batch*.bin")))
    print(f"{num_train_batches} train batches created")
    print(f"{num_fm_val_batches} val batches created")
    print(f"{num_fw_tokens_train=}\n{num_fm_tokens_train=}\n{num_fm_tokens_val=}")
    print(f"Took {perf_counter() - t0:.2f} seconds")
    return num_train_batches, False


def group_and_upload_tokens(
        num_train_batches: int,
        num_batches_per_group: int = 100,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
):
    hf_token=os.getenv("HF_TOKEN")
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."
    print(f"Uploading {num_train_batches} train batches in groups of {num_batches_per_group} to {repo_id}...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    # Upload train batches in groups of 100 batches
    files = sorted(list(Path.cwd() / "data" / f for f in os.listdir("data") if f.startswith("fm_toks_train")))
    min_groups, remainder = divmod(len(files), num_batches_per_group)
    num_groups = min_groups + 1 if remainder > 0 else min_groups
    print(f"Uploading {num_groups} groups of {num_batches_per_group} train batches...")
    for i in tqdm(range(num_groups), total=num_groups):
        start = i * num_batches_per_group
        end = start + num_batches_per_group
        fslice = files[start:end]
        group = load_file(fslice.pop(0))
        for file in fslice:
            group = torch.cat([group, load_file(file)])
        
        if future is not None:
            future.result()
        future = executor.submit(
            upload_with_backoff, api, group, f"fm_toks_train_batches_{start}-{end}.bin", repo_id, "tokens/train"
        )
    
    if future is not None:
        future.result()
    executor.shutdown()
    
    # Upload val batches (it's only going to be one, just upload that)
    batch = load_file("data/fm_toks_val_batch_0.bin")
    upload_with_backoff(api, batch, "fm_toks_val_batch_0.bin", repo_id, "tokens/val")


def create_and_upload_data(
        from_batch: int = 0,
        to_batch: int = -1,
        skip_fm_val_batches: bool = False,
        skip_fw_val_batches: bool = False,
        B: int = 1024,
        T: int = 1024,
        bytes_per_token: int = 16,
        pad_byte: int = 456,
        eot_byte: int = 457,
        vocab_size: int = 50257,
        num_fm_val_batches: int = 1,
        repo_id: str = "snimu/finemath-fineweb-100B-data-for-MoT",
):
    hf_token=os.getenv("HF_TOKEN")
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    print(f"\n{B=} {T=} {bytes_per_token=} {pad_byte=} {eot_byte=} {vocab_size=} {num_fm_val_batches=}\n")
    print("Creating tokens-to-bytes-embeddings...")
    tokens_to_bytes_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    tokens_to_bytes_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)

    print("Setting up HF API...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, token=hf_token, repo_type="dataset", exist_ok=True)

    print("Finding finemath data files...")
    os.makedirs("data", exist_ok=True)
    fm_files_train = sorted(Path.cwd().glob("data/fm_toks_train_batch*.bin"))
    fm_files_val = sorted(Path.cwd().glob("data/fm_toks_val_batch*.bin"))
    print(f"Found {len(fm_files_train)} finemath train batches and {len(fm_files_val)} finemath val batches")

    def create_and_upload_batch(
            futures: list[concurrent.futures.Future],
            batch_num: int,
            tokens: torch.Tensor,
            filename: str,
            t_start: float,
            t_global_start: float,
            path_in_repo: str,
    ):
        if len(futures) == 5:
            for future in futures:
                future.result()
            futures = []
        if os.path.exists(f"data/{filename}"):
            print(f"Skipping {filename} because it already exists...")
            return
        batch = create_batch(
            tokens=tokens,
            bytes_per_token=bytes_per_token,
            pad_byte=pad_byte,
            eot_byte=eot_byte,
            tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
            tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
        )
        if batch_num % 100 == 0:
            verify_data(f"data/{filename}", batch, B, T, bytes_per_token)
        futures.append(executor.submit(upload_with_backoff, api, batch, filename, repo_id, path_in_repo))
        time_taken_step = perf_counter() - t_start
        time_taken_global = perf_counter() - t_global_start
        print(f"{(batch_num+1)*B*T:_} tokens done in {round(time_taken_step):_}s ({round(time_taken_global):_}s total)")

    idx = 0
    executor = ThreadPoolExecutor(max_workers=5)
    futures = []
    t0 = perf_counter()
    t0_global = perf_counter()

    if not skip_fm_val_batches:
        print("Creating finemath val batches...")
        for batch_num_val in range(len(fm_files_val)):
            filename_toks = fm_files_val[batch_num_val]
            batch = load_file(filename_toks).view(B, T)
            filename = f"fm_val_batch_{batch_num_val}.bin"
            create_and_upload_batch(futures, batch_num_val, batch, filename, t0, t0_global, "bytes/val")
            t0 = perf_counter()
    
    if not skip_fw_val_batches:
        dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
        tokens_fw = None
        print("Creating fineweb val batches...")
        dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
        batch_num_val = 0
        for new_tokens in dl:
            tokens_fw = torch.cat([tokens_fw, new_tokens]) if tokens_fw else new_tokens
            if len(tokens_fw) < B*T:
                continue
            num_batches = len(tokens_fw) // (B*T)
            for i in range(0, num_batches * B*T, B*T):
                if len(tokens_fw[i:]) < B*T:
                    break
                batch_num_val += 1
                filename = f"fw_val_batch_{batch_num_val}.bin"
                create_and_upload_batch(
                    futures=futures,
                    batch_num=batch_num_val,
                    tokens=tokens_fw[i:i+B*T].view(B, T).to(torch.int32),
                    filename=filename,
                    t_start=t0,
                    t_global_start=t0_global,
                    path_in_repo="bytes/val",
                )
                t0 = perf_counter()

            num_batches_processed = len(tokens_fw) // (B*T)
            tokens_fw = tokens_fw[num_batches_processed * B*T:]
    
    print("Creating finemath train batches...")
    batch_num_train = 0
    for idx in range(len(fm_files_train)):
        if batch_num_train < from_batch:
            continue
        if to_batch >= 0 and batch_num_train >= to_batch:
            break
        filename_toks = fm_files_train[idx]
        batch = load_file(filename_toks).view(B, T)
        filename = f"fm_toks_train_batch_{batch_num_train}.bin"
        create_and_upload_batch(futures, batch_num_train, batch, filename, t0, t0_global, "bytes/train")
        batch_num_train += 1
        t0 = perf_counter()

    print("Creating fineweb train batches...")
    print("Setting up fineweb dataloader...")
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin")
    tokens_fw = next(dl)
    batch_num_fw = 0  # distinguish between finemath and fineweb batches but count global batch number for parallel workers
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        if len(tokens_fw) < B*T:
            continue
        num_batches = len(tokens_fw) // B*T
        for i in range(0, num_batches * B*T, B*T):
            if len(tokens_fw[i:]) < B*T:
                break
            batch_num_train += 1  # for tracking from_batch and to_batch
            batch_num_fw += 1  # for naming the fineweb batches
            if batch_num_train < from_batch:
                continue
            if to_batch >= 0 and batch_num_train >= to_batch:
                break
            filename = f"fw_train_batch_{batch_num_fw}.bin"
            create_and_upload_batch(
                futures=futures,
                batch_num=batch_num_train,
                tokens=tokens_fw[i*B*T : (i+1)*B*T].view(B, T).to(torch.int32),
                filename=filename,
                t_start=t0,
                t_global_start=t0_global,
                path_in_repo="bytes/train",
            )
            t0 = perf_counter()

        num_batches_processed = len(tokens_fw) // (B*T)
        tokens_fw = tokens_fw[num_batches_processed * B*T:]

    # Wait for all uploads to finish
    for future in futures:
        future.result()
    futures = []
    executor.shutdown()


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
    # Finemath: 6542 batches (at B=1024, T=1024 --> 6,859,784,192 tokens) (I was dumb -> includes val batch -> 6541 train batches)
    # Fineweb: 85067 batches (at B=1024, T=1024 --> 89,199,214,592 tokens) (This only includes train batches)
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-batch", type=int, default=0)
    parser.add_argument("--to-batch", type=int, default=-1)
    parser.add_argument("--skip-fm-val-batches", action="store_true")
    parser.add_argument("--skip-fw-val-batches", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    if args.tokenize:
        num_train_batches, on_hf = tokenize_finemath(B=1024, T=1024, vocab_size=50257, num_fm_val_batches=1, overlap=128)
        if not on_hf:
            group_and_upload_tokens(num_train_batches)
    create_and_upload_data(args.from_batch, args.to_batch, args.skip_fm_val_batches, args.skip_fw_val_batches)


if __name__ == "__main__":
    main()