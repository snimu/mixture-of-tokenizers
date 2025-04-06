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


# Thanks Google Gemini Pro 2.5 for the 150x speedup!
@torch.compile(mode="reduce-overhead")
def pull_from_right(
    byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> torch.Tensor:
    """
    Pulls valid bytes towards the left boundary of each token, considering EOT tokens
    as sequence breaks. Bytes are taken from the current token up to (but not including)
    the next EOT token. EOT tokens retain their original bytes.
    Vectorized implementation.
    """
    B, T = byte_tensor.size()
    if T == 0: # Handle empty input
        return byte_tensor
    T_reduced = T // bytes_per_token
    assert T % bytes_per_token == 0, "T must be divisible by bytes_per_token"
    if T_reduced == 0: # Handle case where T < bytes_per_token
         return torch.full_like(byte_tensor, pad_byte)

    # 1. Preprocessing
    byte_tensor_view = byte_tensor.view(B, T_reduced, bytes_per_token)
    device = byte_tensor.device

    non_pad_mask = byte_tensor_view != pad_byte  # (B, Tr, bpt)
    is_eot_token = torch.all(byte_tensor_view == eot_byte, dim=2) # (B, Tr)

    valid_bytes_per_token = non_pad_mask.sum(dim=2) # (B, Tr)

    # Cumulative count of *valid* bytes up to the start of each token
    cum_valid_bytes = torch.cumsum(
        torch.cat([torch.zeros_like(valid_bytes_per_token[:, :1]), valid_bytes_per_token], dim=1),
        dim=1
    ) # (B, Tr + 1)
    start_valid_byte_idx = cum_valid_bytes[:, :-1] # (B, Tr) - Global index of first valid byte for token t
    end_valid_byte_idx = cum_valid_bytes[:, 1:]   # (B, Tr) - Global index of last valid byte + 1 for token t
    total_valid_bytes_per_batch = cum_valid_bytes[:, -1] # (B,)

    # 2. Find Next EOT Boundary (using searchsorted loop for now)
    next_eot_token_indices = torch.full_like(is_eot_token, T_reduced, dtype=torch.long) # (B, Tr)
    for b in range(B):
        eot_pos = is_eot_token[b].nonzero(as_tuple=True)[0]
        if eot_pos.numel() > 0:
            token_indices = torch.arange(T_reduced, device=device)
            # Find index in eot_pos for the first EOT >= token_indices
            next_eot_rel_idx = torch.searchsorted(eot_pos, token_indices, side='left')
            # Clamp indices to valid range and get actual token positions
            valid_mask = next_eot_rel_idx < eot_pos.numel()
            next_eot_token_indices[b, valid_mask] = eot_pos[next_eot_rel_idx[valid_mask]]
            # Tokens after the last EOT will correctly point to T_reduced (default value)

    # 3. Calculate Byte Ranges to Pull
    # Global index of the first valid byte in the *next* EOT token (or end of sequence)
    # Use gather with the corrected indexing (no extra unsqueeze)
    next_eot_valid_byte_start_idx = cum_valid_bytes.gather(1, next_eot_token_indices) # (B, Tr)

    # Number of valid bytes available from current token up to (not including) next EOT
    available_bytes = next_eot_valid_byte_start_idx - start_valid_byte_idx # (B, Tr)
    bytes_to_pull = torch.minimum(available_bytes, torch.tensor(bytes_per_token, device=device)) # (B, Tr)
    bytes_to_pull = torch.clamp(bytes_to_pull, min=0) # Ensure non-negative

    # 4. Gather Bytes (Using Flattening Approach)
    flat_indices_b, flat_indices_t, flat_indices_k = non_pad_mask.nonzero(as_tuple=True)
    flat_valid_bytes = byte_tensor_view[flat_indices_b, flat_indices_t, flat_indices_k] # (total_valid_bytes,)

    # Mapping from batch item to its range in flat_valid_bytes
    batch_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), total_valid_bytes_per_batch.cumsum(0)[:-1]]) # (B,)

    # Create indices for the output tensor (B, Tr, bpt)
    k_indices = torch.arange(bytes_per_token, device=device).view(1, 1, bytes_per_token).expand(B, T_reduced, -1)

    # Calculate the global valid byte index we want for each output slot (b, t, k)
    # target_global_valid_idx = start_valid_byte_idx[b, t] + k
    target_global_valid_idx = start_valid_byte_idx.unsqueeze(2) + k_indices # (B, Tr, bpt)

    # Create a mask for indices we actually need to gather (k < bytes_to_pull)
    gather_mask = k_indices < bytes_to_pull.unsqueeze(2) # (B, Tr, bpt)

    # Adjust target global indices to be relative to the flattened array
    absolute_gather_idx = target_global_valid_idx + batch_offsets.view(B, 1, 1) # (B, Tr, bpt)

    # Pad flat_valid_bytes to handle potential out-of-bounds gathers safely
    total_flat_size = batch_offsets[-1] + total_valid_bytes_per_batch[-1] if B > 0 else 0
    # Use a large index for invalid gathers and replace later
    safe_indices = torch.where(gather_mask, absolute_gather_idx, total_flat_size) # Use index outside valid range

    # Add a padding value at the end of flat_valid_bytes
    # Ensure dtype matches byte_tensor
    padded_flat_valid_bytes = torch.cat([flat_valid_bytes, torch.tensor([pad_byte], device=device, dtype=byte_tensor.dtype)])

    # Perform the gather
    # Clamp indices just in case, although safe_indices should handle it
    clamped_indices = torch.clamp(safe_indices, max=total_flat_size)
    gathered_bytes_flat = padded_flat_valid_bytes[clamped_indices] # (B, Tr, bpt)

    # Reshape gathered bytes and apply padding where gather_mask was false
    pulled_non_eot = torch.where(gather_mask, gathered_bytes_flat, torch.tensor(pad_byte, device=device, dtype=byte_tensor.dtype)) # (B, Tr, bpt)

    # 5. Handle EOT Tokens
    # EOT tokens keep their original bytes, exactly as they were.
    final_pulled_tensor = torch.where(
        is_eot_token.unsqueeze(-1),
        byte_tensor_view, # Keep original bytes exactly as they were for EOTs
        pulled_non_eot      # Use the pulled bytes for non-EOTs
    )

    # 6. Reshape back
    return final_pulled_tensor.view(B, T)


@torch.compile(mode="reduce-overhead")
def pull_from_left(
    byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> torch.Tensor:
    """
    Pulls valid bytes towards the right boundary of each token, considering EOT tokens
    as sequence breaks. Bytes are taken from the token after the previous EOT up to
    the current token. The rightmost available bytes are kept if capacity is exceeded.
    EOT tokens retain their original bytes. Vectorized implementation.
    """
    B, T = byte_tensor.size()
    if T == 0: return byte_tensor
    T_reduced = T // bytes_per_token
    assert T % bytes_per_token == 0, "T must be divisible by bytes_per_token"
    if T_reduced == 0: return torch.full_like(byte_tensor, pad_byte)

    # 1. Preprocessing (Identical to pull_from_right)
    byte_tensor_view = byte_tensor.view(B, T_reduced, bytes_per_token)
    device = byte_tensor.device

    non_pad_mask = byte_tensor_view != pad_byte
    is_eot_token = torch.all(byte_tensor_view == eot_byte, dim=2)

    valid_bytes_per_token = non_pad_mask.sum(dim=2)

    cum_valid_bytes = torch.cumsum(
        torch.cat([torch.zeros_like(valid_bytes_per_token[:, :1]), valid_bytes_per_token], dim=1),
        dim=1
    )
    # start_valid_byte_idx = cum_valid_bytes[:, :-1] # Not directly needed here
    end_valid_byte_idx = cum_valid_bytes[:, 1:]   # (B, Tr) - Global index of last valid byte + 1 for token t
    total_valid_bytes_per_batch = cum_valid_bytes[:, -1] # (B,)

    # 2. Find Previous EOT Boundary (using searchsorted loop for now)
    prev_eot_token_indices = torch.full_like(is_eot_token, -1, dtype=torch.long) # (B, Tr)
    for b in range(B):
        eot_pos = is_eot_token[b].nonzero(as_tuple=True)[0]
        if eot_pos.numel() > 0:
            token_indices = torch.arange(T_reduced, device=device)
            # Find index in eot_pos for the last EOT <= token_indices
            prev_eot_rel_idx = torch.searchsorted(eot_pos, token_indices, side='right') - 1
            # Get actual token positions for valid indices
            valid_mask = prev_eot_rel_idx >= 0
            prev_eot_token_indices[b, valid_mask] = eot_pos[prev_eot_rel_idx[valid_mask]]
            # Tokens before the first EOT correctly have -1

    # 3. Calculate Byte Ranges to Pull
    # Global index of the first valid byte *after* the previous EOT token
    # Need to handle prev_eot_token_indices == -1
    prev_eot_plus_1 = (prev_eot_token_indices + 1).clamp(min=0) # Clamp to handle -1 -> 0
    # Use gather with the corrected indexing (no extra unsqueeze)
    pull_range_start_idx = cum_valid_bytes.gather(1, prev_eot_plus_1) # (B, Tr)
    # Use mask for tokens before first EOT (where prev_eot == -1) to ensure start is 0
    pull_range_start_idx = torch.where(prev_eot_token_indices == -1, torch.tensor(0, device=device, dtype=torch.long), pull_range_start_idx)

    # Global index of the last valid byte + 1 for the *current* token `t`
    pull_range_end_idx = end_valid_byte_idx # (B, Tr)

    # Number of valid bytes available from (prev EOT + 1) up to current token `t`
    available_bytes = pull_range_end_idx - pull_range_start_idx # (B, Tr)
    available_bytes = torch.clamp(available_bytes, min=0)

    # We need to take the *rightmost* `bytes_per_token` of these available bytes
    bytes_to_use = torch.minimum(available_bytes, torch.tensor(bytes_per_token, device=device)) # (B, Tr)

    # Calculate the starting global index for the bytes we actually want to gather
    gather_start_global_idx = pull_range_end_idx - bytes_to_use # (B, Tr)

    # 4. Gather Bytes (Using Flattening Approach)
    flat_indices_b, flat_indices_t, flat_indices_k = non_pad_mask.nonzero(as_tuple=True)
    flat_valid_bytes = byte_tensor_view[flat_indices_b, flat_indices_t, flat_indices_k]

    batch_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), total_valid_bytes_per_batch.cumsum(0)[:-1]])

    # Indices for the output tensor relative to the gather operation (0 to bytes_to_use - 1)
    k_indices_relative = torch.arange(bytes_per_token, device=device).view(1, 1, -1) # (1, 1, bpt)

    # Calculate the absolute global valid byte index we want for each relative k
    # target_global_valid_idx = gather_start_global_idx[b, t] + k_relative
    target_global_valid_idx = gather_start_global_idx.unsqueeze(2) + k_indices_relative # (B, Tr, bpt)

    # Create mask for indices we need (k_relative < bytes_to_use)
    gather_mask = k_indices_relative < bytes_to_use.unsqueeze(2) # (B, Tr, bpt)

    # Adjust target global indices for the flattened array
    absolute_gather_idx = target_global_valid_idx + batch_offsets.view(B, 1, 1) # (B, Tr, bpt)

    # Pad flat_valid_bytes and gather safely
    total_flat_size = batch_offsets[-1] + total_valid_bytes_per_batch[-1] if B > 0 else 0
    safe_indices = torch.where(gather_mask, absolute_gather_idx, total_flat_size)
    # Ensure dtype matches byte_tensor
    padded_flat_valid_bytes = torch.cat([flat_valid_bytes, torch.tensor([pad_byte], device=device, dtype=byte_tensor.dtype)])
    clamped_indices = torch.clamp(safe_indices, max=total_flat_size)
    gathered_bytes_flat = padded_flat_valid_bytes[clamped_indices] # (B, Tr, bpt) - These are the desired bytes, but left-aligned in this tensor

    # 5. Place Bytes Right-Aligned and Handle EOTs
    # Create the output tensor, initially padded
    pulled_non_eot = torch.full_like(byte_tensor_view, pad_byte)

    # Calculate the destination k index for placing the gathered bytes
    # The k_th gathered byte (0 <= k < bytes_to_use) should go to slot (bpt - bytes_to_use + k)
    dest_k_indices = bytes_per_token - bytes_to_use.unsqueeze(2) + k_indices_relative # (B, Tr, bpt)

    # We only place where gather_mask is true
    # Create full B, Tr indices for scatter/advanced indexing
    b_indices = torch.arange(B, device=device).view(B, 1, 1).expand_as(dest_k_indices)
    t_indices = torch.arange(T_reduced, device=device).view(1, T_reduced, 1).expand_as(dest_k_indices)

    # Use advanced indexing to place the bytes
    # Only update positions where gather_mask is true
    valid_dest_k = dest_k_indices[gather_mask]
    valid_b = b_indices[gather_mask]
    valid_t = t_indices[gather_mask]
    valid_gathered_bytes = gathered_bytes_flat[gather_mask]

    if valid_b.numel() > 0: # Check if there's anything to place
        pulled_non_eot[valid_b, valid_t, valid_dest_k] = valid_gathered_bytes

    # Handle EOT Tokens: Keep original bytes, exactly as they were.
    final_pulled_tensor = torch.where(
        is_eot_token.unsqueeze(-1),
        byte_tensor_view, # Keep original bytes exactly as they were for EOTs
        pulled_non_eot      # Use the pulled bytes for non-EOTs
    )

    # 6. Reshape back
    return final_pulled_tensor.view(B, T)


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


def upload_with_backoff(
        api: HfApi,
        batch: torch.Tensor,
        filename: str,
        repo_id: str,
        path_in_repo: str = "bytes",
        save_first=True,
        delete_after=True,
):
    if save_first:
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
    if delete_after:
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
        num_batches_per_group: int = 5,
        max_workers_upload: int = 5,
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
    repofiles = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    repofiles = sorted([f.split("/")[-1] for f in repofiles if f.startswith("bytes/")])

    print("Finding finemath data files...")
    os.makedirs("data", exist_ok=True)
    fm_files_train = sorted(Path.cwd().glob("data/fm_toks_train_batch*.bin"))
    fm_files_val = sorted(Path.cwd().glob("data/fm_toks_val_batch*.bin"))
    print(f"Found {len(fm_files_train)} finemath train batches and {len(fm_files_val)} finemath val batches")

    def group_and_save_batches(files: list[str], filename: str):
        files = [f"data/{f}" if not f.startswith("data/") else f for f in files]
        files = sorted(files)
        file0 = files.pop(0)
        group = load_file(file0)
        os.remove(file0)
        for file in files:
            group = torch.cat([group, load_file(file)])
            os.remove(file)
        save_file(f"data/{filename}", group)

    def minmax_filename(filenames: list[str]) -> tuple[int, int]:
        filenums = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
        return min(filenums), max(filenums)

    def create_and_save_batch(
            batch_num: int,
            tokens: torch.Tensor,
            filename: str,
            t_start: float,
            t_global_start: float,
    ):
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
        save_file(f"data/{filename}", batch)
        time_taken_step = perf_counter() - t_start
        time_taken_global = perf_counter() - t_global_start
        print(f"{(batch_num+1-from_batch)*B*T:_} tokens done in {round(time_taken_step):_}s ({round(time_taken_global):_}s total)")

    idx = 0
    t0 = perf_counter()
    t0_global = perf_counter()

    executor = ThreadPoolExecutor(max_workers=max_workers_upload)
    futures = []

    if not skip_fm_val_batches:
        filenames = []
        print("Creating finemath val batches...")
        for batch_num_val in range(len(fm_files_val)):
            # Uploading
            if len(futures) >= max_workers_upload:
                for future in futures:
                    future.result()
                futures = []
            if len(filenames) >= num_batches_per_group:
                min_, max_ = minmax_filename(filenames)
                filename = f"fm_val_batches_{min_}-{max_}.bin"
                group_and_save_batches(filenames, filename)
                filenames = []
                futures.append(executor.submit(upload_with_backoff, api, load_file(f"data/{filename}"), filename, repo_id, "bytes/val", save_first=False))
            
            # Creating
            filename_toks = fm_files_val[batch_num_val]
            batch = load_file(filename_toks).view(B, T)
            filename = f"fm_val_batch_{batch_num_val}.bin"
            filenames.append(filename)
            create_and_save_batch(batch_num_val, batch, filename, t0, t0_global)
            t0 = perf_counter()

        # Uploading remainder
        if len(filenames) > 0:
            for future in futures:
                future.result()
            futures = []
            min_, max_ = minmax_filename(filenames)
            filename = f"fm_val_batches_{min_}-{max_}.bin"
            group_and_save_batches(filenames, filename)
            upload_with_backoff(api, load_file(f"data/{filename}"), filename, repo_id, "bytes/val", save_first=False)
            filenames = []
    
    if not skip_fw_val_batches:
        dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
        tokens_fw = None
        print("Creating fineweb val batches...")
        dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
        batch_num_val = 0
        filenames = []
        for new_tokens in dl:
            tokens_fw = torch.cat([tokens_fw, new_tokens]) if tokens_fw else new_tokens
            if len(tokens_fw) < B*T:
                continue
            num_batches = len(tokens_fw) // (B*T)
            for i in range(0, num_batches * B*T, B*T):
                # Uploading
                if len(futures) >= max_workers_upload:
                    for future in futures:
                        future.result()
                    futures = []
                if len(filenames) >= num_batches_per_group:
                    min_, max_ = minmax_filename(filenames)
                    filename = f"fw_val_batches_{min_}-{max_}.bin"
                    group_and_save_batches(filenames, filename)
                    filenames = []
                    futures.append(executor.submit(upload_with_backoff, api, load_file(f"data/{filename}"), filename, repo_id, "bytes/val", save_first=False))

                # Creating
                if len(tokens_fw[i:]) < B*T:
                    break
                batch_num_val += 1
                filename = f"fw_val_batch_{batch_num_val}.bin"
                create_and_save_batch(
                    batch_num=batch_num_val,
                    tokens=tokens_fw[i:i+B*T].view(B, T).to(torch.int32),
                    filename=filename,
                    t_start=t0,
                    t_global_start=t0_global,
                )
                filenames.append(filename)
                t0 = perf_counter()

            num_batches_processed = len(tokens_fw) // (B*T)
            tokens_fw = tokens_fw[num_batches_processed * B*T:]
        
        # Uploading remainder
        if len(filenames) > 0:
            for future in futures:
                future.result()
            futures = []
            min_, max_ = minmax_filename(filenames)
            filename = f"fw_val_batches_{min_}-{max_}.bin"
            group_and_save_batches(filenames, filename)
            upload_with_backoff(api, load_file(f"data/{filename}"), filename, repo_id, "bytes/val", save_first=False)
            filenames = []
    
    print("Creating finemath train batches...")
    batch_num_train = 0
    filenames = []
    for idx in range(len(fm_files_train)):
        # Uploading
        if len(futures) >= max_workers_upload:
            for future in futures:
                future.result()
            futures = []
        if len(filenames) >= num_batches_per_group:
            min_, max_ = minmax_filename(filenames)
            filename = f"fm_train_batches_{min_}-{max_}.bin"
            group_and_save_batches(filenames, filename)
            filenames = []
            futures.append(executor.submit(upload_with_backoff, api, load_file(f"data/{filename}"), filename, repo_id, "bytes/train", save_first=False))
        
        # Creating
        if batch_num_train < from_batch:
            continue
        if to_batch >= 0 and batch_num_train >= to_batch:
            break
        filename_toks = fm_files_train[idx]
        batch = load_file(filename_toks).view(B, T)
        filename = f"fm_train_batch_{batch_num_train}.bin"
        create_and_save_batch(batch_num_train, batch, filename, t0, t0_global)
        filenames.append(filename)
        batch_num_train += 1
        t0 = perf_counter()

    if len(filenames) > 0:
        for future in futures:
            future.result()
        futures = []
        min_, max_ = minmax_filename(filenames)
        filename = f"fm_train_batches_{min_}-{max_}.bin"
        group_and_save_batches(filenames, filename)
        upload_with_backoff(api, load_file(f"data/{filename}"), filename, repo_id, "bytes/train", save_first=False)

    print("Creating fineweb train batches...")
    print("Setting up fineweb dataloader...")
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin")
    tokens_fw = next(dl)
    batch_num_fw = 0  # distinguish between finemath and fineweb batches but count global batch number for parallel workers
    filenames = []
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        if len(tokens_fw) < B*T:
            continue
        num_batches = len(tokens_fw) // B*T
        for i in range(num_batches):
            # Uploading
            if len(futures) >= max_workers_upload:
                for future in futures:
                    future.result()
                futures = []
            if len(filenames) >= num_batches_per_group:
                min_, max_ = minmax_filename(filenames)
                filename = f"fw_train_batches_{min_}-{max_}.bin"
                group_and_save_batches(filenames, filename)
                filenames = []
                futures.append(executor.submit(upload_with_backoff, api, load_file(f"data/{filename}"), filename, repo_id, "bytes/train", save_first=False))

            # Creating
            if len(tokens_fw[i*B*T:]) < B*T:
                break
            batch_num_train += 1  # for tracking from_batch and to_batch
            batch_num_fw += 1  # for naming the fineweb batches
            if batch_num_train < from_batch:
                continue
            if to_batch >= 0 and batch_num_train >= to_batch:
                break
            filename = f"fw_train_batch_{batch_num_fw}.bin"
            create_and_save_batch(
                batch_num=batch_num_train,
                tokens=tokens_fw[i*B*T : (i+1)*B*T].view(B, T).to(torch.int32),
                filename=filename,
                t_start=t0,
                t_global_start=t0_global,
            )
            filenames.append(filename)
            t0 = perf_counter()

        num_batches_processed = len(tokens_fw) // (B*T)
        tokens_fw = tokens_fw[num_batches_processed * B*T:]

    # Uploading remainder
    if len(filenames) > 0:
        for future in futures:
            future.result()
        futures = []
        min_, max_ = minmax_filename(filenames)
        filename = f"fw_train_batches_{min_}-{max_}.bin"
        group_and_save_batches(filenames, filename)
        upload_with_backoff(api, load_file(f"data/{filename}"), filename, repo_id, "bytes/train", save_first=False)
        filenames = []

    executor.shutdown()


#####################
###### TESTING ######
#####################


def pfr(byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int):  # for trying out more efficient verstions of pull_from_right
    pass


def _time_pfr():
    num_iter_orig = 5
    num_iter_new = 100
    bytes_per_token = 16
    pad_byte = 456
    eot_byte = 457
    vocab_size = 50257

    pfr_orig_times = []
    pfr_times = []

    def make_example_tensor(B: int, T: int):
        tokens = torch.randint(0, 50256, (B, T), dtype=torch.int32)
        eot_positions = torch.rand(B, T)
        tokens = torch.where(eot_positions > 0.8, 50256, tokens)
        bytes_to_tokens_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
        byte_tensor_right_pad = tokens_to_bytes(tokens, bytes_to_tokens_right_pad)
        return byte_tensor_right_pad
    
    print("Original...")
    B, T = 1024, 1024
    for i in tqdm(range(num_iter_orig)):
        byte_tensor_right_pad = make_example_tensor(B, T)
        t0 = perf_counter()
        _ = pull_from_right(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
        pfr_orig_times.append(perf_counter() - t0)
    
    print("Modified...")
    B, T = 1024, 1024
    for i in tqdm(range(num_iter_new)):
        byte_tensor_right_pad = make_example_tensor(B, T)
        t0 = perf_counter()
        _ = pfr(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
        pfr_times.append(perf_counter() - t0)
    
    t_orig = np.mean(pfr_orig_times)
    t = np.mean(pfr_times)
    print(f"pfr_orig: {t_orig:.2f}s")
    print(f"pfr: {t:.2f}s")


def _print_pfr():
    bytes_per_token = 16
    pad_byte = 456
    eot_byte = 457
    vocab_size = 50257

    def make_example_tensor(B, T):
        tokens = torch.randint(0, 50256, (B, T), dtype=torch.int32)
        eot_positions = torch.rand(B, T)
        tokens = torch.where(eot_positions > 0.8, 50256, tokens)
        bytes_to_tokens_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
        byte_tensor_right_pad = tokens_to_bytes(tokens, bytes_to_tokens_right_pad)
        return tokens, byte_tensor_right_pad

    B, T = 1024, 1024
    tokens, byte_tensor_right_pad = make_example_tensor(B, T)
    byte_tensor_pull_from_right = pull_from_right(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
    byte_tensor_pfr = pfr(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
    print(f"{torch.allclose(byte_tensor_pull_from_right, byte_tensor_pfr)=}")

    B, T = 1, 16
    tokens, byte_tensor_right_pad = make_example_tensor(B, T)
    byte_tensor_pull_from_right = pull_from_right(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
    byte_tensor_pfr = pfr(byte_tensor_right_pad, bytes_per_token, pad_byte, eot_byte)
    print(f"{tokens.shape=}\n{byte_tensor_right_pad.shape=}\n\nTOKENS")
    print(tokens)
    print("\n\nBYTES RIGHT PAD")
    print(byte_tensor_right_pad.view(B, T, bytes_per_token))
    print("\n\nBYTES RIGHT PULLED")
    print(byte_tensor_pull_from_right)
    print("\n\nBYTES RIGHT PULLED (pfr)")
    print(byte_tensor_pfr)
    print("\n\n")


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
    parser.add_argument("--start-upload-loop", action="store_true")
    args = parser.parse_args()
    if args.tokenize:
        num_train_batches, on_hf = tokenize_finemath(B=1024, T=1024, vocab_size=50257, num_fm_val_batches=1, overlap=128)
        if not on_hf:
            group_and_upload_tokens(num_train_batches)
    create_and_upload_data(
        args.from_batch, args.to_batch, args.skip_fm_val_batches, args.skip_fw_val_batches,
    )


if __name__ == "__main__":
    main()
