
import json
import os
from time import perf_counter
from pathlib import Path

import torch
from torch import nn
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi


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
    
    # Count valid (non-padding) bytes per token
    valid_bytes_count = non_pad_mask.sum(dim=2)  # Shape: (B, T_reduced)
    
    # Initialize output tensor
    pulled_tensor = torch.empty_like(byte_tensor)
    pulled_tensor.fill_(pad_byte)
    
    # Process each batch
    for batch_idx in range(B):
        # Get data for this batch
        batch_tensor = byte_tensor[batch_idx]  # (T_reduced, bytes_per_token)
        batch_non_pad = non_pad_mask[batch_idx]  # (T_reduced, bytes_per_token)
        batch_is_eot = is_eot_token[batch_idx]   # (T_reduced)
        batch_valid_count = valid_bytes_count[batch_idx]  # (T_reduced)
        
        # Calculate cumulative positions of valid bytes using tensor operations
        token_positions = torch.cat([
            torch.zeros(1, dtype=torch.long, device=byte_tensor.device),
            torch.cumsum(batch_valid_count, dim=0)
        ])  # (T_reduced + 1)
        
        # Extract all valid (non-padded) bytes at once using boolean indexing
        all_valid_bytes = batch_tensor[batch_non_pad]  # Flattened tensor of all valid bytes
        
        # Find positions of EOT tokens
        eot_positions = torch.where(batch_is_eot)[0]
        
        # Vectorized computation of next EOT token for each position
        if len(eot_positions) > 0:
            # For each token position, find the next EOT token efficiently
            token_indices = torch.arange(T_reduced, device=byte_tensor.device)
            next_eot_indices_raw = torch.searchsorted(eot_positions, token_indices)
            
            # Convert to actual EOT positions
            next_eot_indices = torch.full((T_reduced,), T_reduced, device=byte_tensor.device)
            valid_mask = next_eot_indices_raw < len(eot_positions)
            next_eot_indices[valid_mask] = eot_positions[next_eot_indices_raw[valid_mask]]
        else:
            # No EOT tokens in this batch
            next_eot_indices = torch.full((T_reduced,), T_reduced, device=byte_tensor.device)
        
        # We still need to process each token sequentially
        for token_idx in range(T_reduced):
            # For EOT tokens, keep original bytes
            if batch_is_eot[token_idx]:
                token_valid_mask = batch_non_pad[token_idx]
                token_valid_bytes = batch_tensor[token_idx, token_valid_mask]
                if len(token_valid_bytes) > 0:
                    pulled_tensor[batch_idx, token_idx, :len(token_valid_bytes)] = token_valid_bytes
                continue
            
            # For other tokens, pull bytes up to the next EOT token
            start_idx = token_positions[token_idx].item()
            next_eot_idx = next_eot_indices[token_idx].item()
            end_idx = token_positions[next_eot_idx].item()
            
            # Calculate bytes to pull (limited to bytes_per_token)
            bytes_to_pull = min(bytes_per_token, end_idx - start_idx)
            
            if bytes_to_pull > 0 and start_idx < len(all_valid_bytes):
                pulled_bytes = all_valid_bytes[start_idx:start_idx + bytes_to_pull]
                pulled_tensor[batch_idx, token_idx, :len(pulled_bytes)] = pulled_bytes
    
    # Reshape back to original dimensions
    return pulled_tensor.view(B, T)


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
    pulled_tensor = torch.empty_like(byte_tensor)
    pulled_tensor.fill_(pad_byte)
    
    # Process each batch
    for batch_idx in range(B):
        # Get data for this batch
        batch_tensor = byte_tensor[batch_idx]  # (T_reduced, bytes_per_token)
        batch_non_pad = non_pad_mask[batch_idx]  # (T_reduced, bytes_per_token)
        batch_is_eot = is_eot_token[batch_idx]   # (T_reduced)
        
        # Extract all valid bytes and their positions
        valid_bytes_list = []
        token_start_positions = [0]  # Starting position for each token's bytes
        
        for token_idx in range(T_reduced):
            token_valid_mask = batch_non_pad[token_idx]
            token_valid_bytes = batch_tensor[token_idx, token_valid_mask].tolist()
            valid_bytes_list.extend(token_valid_bytes)
            token_start_positions.append(token_start_positions[-1] + len(token_valid_bytes))
        
        # Convert to tensor
        if valid_bytes_list:
            all_valid_bytes = torch.tensor(valid_bytes_list, device=byte_tensor.device)
        else:
            all_valid_bytes = torch.tensor([], device=byte_tensor.device, dtype=torch.int32)
        
        # Find EOT token positions
        eot_positions = torch.where(batch_is_eot)[0].tolist()
        
        # Process each token
        for token_idx in range(T_reduced):
            current_start = token_start_positions[token_idx]
            current_end = token_start_positions[token_idx + 1]
            
            # If this is an EOT token, only use its own bytes
            if batch_is_eot[token_idx]:
                bytes_to_use = all_valid_bytes[current_start:current_end]
                if len(bytes_to_use) > 0:
                    pulled_tensor[batch_idx, token_idx, -len(bytes_to_use):] = bytes_to_use
                continue
            
            # Find the previous EOT token (or -1 if none)
            prev_eot_idx = -1
            for idx in eot_positions:
                if idx < token_idx:
                    prev_eot_idx = max(prev_eot_idx, idx)
            
            # Calculate the starting position for byte pulling
            # If there's a previous EOT, start from the token after it
            if prev_eot_idx >= 0:
                pull_start = token_start_positions[prev_eot_idx + 1]
            else:
                # Otherwise, start from the beginning
                pull_start = 0
            
            # Calculate bytes to pull
            total_bytes = current_end - pull_start
            
            if total_bytes <= bytes_per_token:
                # If we have fewer bytes than needed, use all of them
                bytes_to_use = all_valid_bytes[pull_start:current_end]
            else:
                # If we have more bytes than needed, take the rightmost bytes_per_token bytes
                bytes_to_use = all_valid_bytes[current_end - bytes_per_token:current_end]
            
            # Place bytes in output tensor
            if len(bytes_to_use) > 0:
                pulled_tensor[batch_idx, token_idx, -len(bytes_to_use):] = bytes_to_use
    
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


def distributed_data_generator(filename_pattern: str):
    files = sorted(Path.cwd().glob(filename_pattern))
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    while True:
        yield _load_data_shard(next(file_iter))


def create_and_upload_data(
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

    print(f"\n{B=} {T=} {bytes_per_token=} {pad_byte=} {eot_byte=} {vocab_size=} {num_fm_val_batches=}\n")

    eot_token = vocab_size - 1
    print("Creating tokens-to-bytes-embeddings...")
    tokens_to_bytes_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    tokens_to_bytes_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)
    print("Setting up tiktoken encoding...")
    encoding = tiktoken.encoding_for_model("gpt-2")
    print("Setting up fineweb dataloader...")
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin")
    tokens_fw = next(dl)

    # Download, tokenize, and save the finemath data, and fill it up to T with random fineweb samples
    print("Setting up HF API...")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    batch = []
    idx = 0
    num_fm_tokens_train = 0
    num_fw_tokens_train = 0
    num_fm_tokens_val = 0
    num_fw_tokens_val = 0
    print("Starting data creation...")
    t0 = perf_counter()
    for row in load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True):
        batch_num = idx // B
        is_val_batch = batch_num < num_fm_val_batches
        is_batch_start = idx % B == 0
        is_batch_end = idx % B == B - 1
        if is_batch_start and is_val_batch:
            print(f"finemath val batch {batch_num}...", end="", flush=True)
        elif is_batch_start:
            print(f"finemath train batch {batch_num - num_fm_val_batches}...", end="", flush=True)

        text = row["text"]
        tokens_fm = torch.tensor(encoding.encode(text), dtype=torch.int32)

        # Don't use incomplete finemath samples
        if len(tokens_fm) > T:
            continue

        # The sample will be filled to T with a random fineweb slice;
        # There has to be an EOT token between them.
        # Exception: first N batches, which will be the finemath-validation set
        if is_val_batch:
            tokens = torch.empty((T,), dtype=torch.int32).fill_(eot_token)
            tokens[:len(tokens_fm)] = tokens_fm
            batch.append(tokens.tolist())
            num_fm_tokens_val += len(torch.where(tokens != eot_token))
        else:
            if len(tokens_fm) < T and tokens_fm[-1] != eot_token:
                tokens_fm = torch.cat([tokens_fm, torch.empty((1,), dtype=torch.int32).fill_(eot_token)])

            if len(tokens_fm) == T:
                batch.append(tokens_fm.tolist())
                num_fm_tokens_train += len(tokens_fm)
            else:
                num_tokens_missing = T - len(tokens_fm)  # 0 <= num_tokens_missing <= T, see condition above
                while len(tokens_fw) < num_tokens_missing:
                    tokens_fw = torch.cat([tokens_fw, next(dl)])

                fillup_tokens, tokens_fw = tokens_fw[:num_tokens_missing], tokens_fw[num_tokens_missing:]
                batch.append(torch.cat([tokens_fm, fillup_tokens.to(tokens_fm.dtype)]).tolist())
                num_fm_tokens_train += len(tokens_fm)
                num_fw_tokens_train += len(fillup_tokens)
        
        # Save every B samples; a.k.a. every batch
        if is_batch_end:
            assert len(batch) == B, f"{len(batch)=} != {B=}"
            batch = create_batch(
                tokens=torch.tensor(batch, dtype=torch.int32),
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            if is_val_batch:
                filename = f"val_batch_finemath_{batch_num}.bin"
            else:
                filename = f"train_batch_{batch_num - num_fm_val_batches}.bin"
            torch.save(batch, f"data/{filename}")
            api.upload_file(path_or_fileobj=f"data/{filename}", path_in_repo=filename, repo_id=repo_id, repo_type="dataset")
            time_taken = perf_counter() - t0
            print(f"{(batch_num+1)*B*T:_} tokens done in {round(time_taken*1000):_}ms ({round(time_taken):_}s)")
            batch = []
        idx += 1
    
    # Now, turn the rest of the fineweb-edu-100BT tokens into their own batches with create_batch
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        if len(tokens_fw) < B*T:
            continue
        for i in range(0, len(tokens_fw), B*T):
            batch_num += 1
            batch = tokens_fw[i:i+B*T].view(B, T).to(torch.int32)
            batch = create_batch(
                tokens=batch,
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            filename = f"train_batch_{batch_num - num_fm_val_batches}.bin"
            torch.save(batch, f"data/{filename}")
            api.upload_file(path_or_fileobj=f"data/{filename}", path_in_repo=filename, repo_id=repo_id, repo_type="dataset")
            time_taken = perf_counter() - t0
            print(f"{(batch_num+1)*B*T:_} tokens done in {round(time_taken*1000):_}ms ({round(time_taken):_}s)")
            num_fw_tokens_train += B*T

    # For finemath, the validation data is created above
    # For fineweb, just use the validation set by karpathy
    dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
    tokens_fw = None
    batch_num = 0
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens]) if tokens_fw else new_tokens
        if len(tokens_fw) < B*T:
            continue
        for i in range(0, len(tokens_fw), B*T):
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
            torch.save(batch, f"data/{filename}")
            api.upload_file(path_or_fileobj=f"data/{filename}", path_in_repo=filename, repo_id=repo_id, repo_type="dataset")
            num_fw_tokens_val += B*T
            batch_num += 1

    # Print stats
    print(f"finemath: {num_fm_tokens_train=}")
    print(f"finemath: {num_fm_tokens_val=}")
    print(f"fineweb: {num_fw_tokens_train=}")
    print(f"fineweb: {num_fw_tokens_val=}")
        


#####################
###### TESTING ######
#####################


def _print_batch():
    B, T = 2, 4
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


if __name__ == "__main__":
    create_and_upload_data()
