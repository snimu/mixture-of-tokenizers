
import json
import glob
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
    def strip_padding(byte_tensor: torch.Tensor) -> torch.Tensor:
        return byte_tensor[torch.where(byte_tensor != pad_byte)]
    B, T = byte_tensor.size()
    T_reduced = T // bytes_per_token 
    byte_tensor = byte_tensor.view(B, T_reduced, bytes_per_token)
    pulled_tensor = torch.empty_like(byte_tensor, dtype=torch.int32).fill_(pad_byte)
    for batch_idx in range(B):
        for seq_idx in range(T_reduced):
            stripped_bytes = strip_padding(byte_tensor[batch_idx, seq_idx])
            for next_byte_idx in range(1, T_reduced):
                # Stop if bytes were fully pulled
                if len(stripped_bytes) >= bytes_per_token:
                    pulled_tensor[batch_idx, seq_idx] = stripped_bytes[:bytes_per_token]
                    break
                # Stop if there are no more bytes to pull
                if seq_idx + next_byte_idx >= T_reduced:
                    pulled_tensor[batch_idx, seq_idx, : len(stripped_bytes)] = stripped_bytes
                    break
                # Gather bytes from next token
                stripped_bytes_next = strip_padding(byte_tensor[batch_idx, seq_idx + next_byte_idx])
                # Don't pull eot bytes
                if all([b == eot_byte for b in stripped_bytes_next]):
                    stripped_bytes = stripped_bytes[:bytes_per_token] if len(stripped_bytes) > bytes_per_token else stripped_bytes
                    pulled_tensor[batch_idx, seq_idx, : len(stripped_bytes)] = stripped_bytes
                    break
                # Pull bytes from next token
                stripped_bytes = torch.cat([stripped_bytes, stripped_bytes_next]).squeeze()
    return pulled_tensor.view(B, T)


def pull_from_left(
        byte_tensor: torch.Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> torch.Tensor:
    def strip_padding(byte_tensor: torch.Tensor) -> torch.Tensor:
        return byte_tensor[torch.where(byte_tensor != pad_byte)]
    B, T = byte_tensor.size()
    T_reduced = T // bytes_per_token 
    byte_tensor = byte_tensor.view(B, T_reduced, bytes_per_token)
    pulled_tensor = torch.empty_like(byte_tensor, dtype=torch.int32).fill_(pad_byte)
    for batch_idx in range(B):
        for seq_idx in range(T_reduced):
            stripped_bytes = strip_padding(byte_tensor[batch_idx, seq_idx])
            for next_byte_idx in range(1, T_reduced):
                # Stop if bytes were fully pulled
                if len(stripped_bytes) >= bytes_per_token:
                    pulled_tensor[batch_idx, seq_idx] = stripped_bytes[-bytes_per_token:]
                    break
                # Stop if there are no more bytes to pull
                if seq_idx - next_byte_idx < 0:
                    pulled_tensor[batch_idx, seq_idx, -len(stripped_bytes):] = stripped_bytes
                    break
                # Gather bytes from next token
                stripped_bytes_pulled = strip_padding(byte_tensor[batch_idx, seq_idx - next_byte_idx])
                # Don't pull eot bytes
                if all([b == eot_byte for b in stripped_bytes_pulled]):
                    stripped_bytes = stripped_bytes[-bytes_per_token:] if len(stripped_bytes) > bytes_per_token else stripped_bytes
                    pulled_tensor[batch_idx, seq_idx, -len(stripped_bytes):] = stripped_bytes
                    break
                # Pull bytes from next token
                stripped_bytes = torch.cat([stripped_bytes_pulled, stripped_bytes]).squeeze()
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
        yield from _load_data_shard(next(file_iter))


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
    eot_token = vocab_size - 1
    tokens_to_bytes_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size)
    tokens_to_bytes_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size)
    embedding = tiktoken.encoding_for_model("gpt-2")
    dl = distributed_data_generator("fineweb100B/fineweb_train_*.bin")
    tokens_fw = next(dl)

    # Download, tokenize, and save the finemath data, and fill it up to T with random fineweb samples
    api = HfApi()
    batch = []
    idx = 0
    num_fm_tokens_train = 0
    num_fw_tokens_train = 0
    num_fm_tokens_val = 0
    num_fw_tokens_val = 0
    for row in load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True):
        is_val_batch = idx < num_fm_val_batches
        text = row["text"]
        tokens_fm = embedding.encode(text)

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
                tokens_fm = torch.cat([tokens_fm, torch.tensor([eot_token]).type_as(tokens_fm).squeeze()])

            if len(tokens_fm) == T:
                batch.append(tokens_fm.tolist())
                num_fm_tokens_train += len(tokens_fm)
            else:
                num_tokens_missing = T - len(tokens_fm)  # 0 <= num_tokens_missing <= T, see condition above
                while len(tokens_fw) < num_tokens_missing:
                    tokens_fw = torch.cat([tokens_fw, next(dl)])

                fillup_tokens, tokens_fw = tokens_fw[:num_tokens_missing], tokens_fw[num_tokens_missing:]
                batch.append(torch.cat([tokens_fm, fillup_tokens]).tolist())
                num_fm_tokens_train += len(tokens_fm)
                num_fw_tokens_train += len(fillup_tokens)
        
        # Save every B samples; a.k.a. every batch
        if idx > 0 and idx % B == 0:
            batch = create_batch(
                tokens=torch.tensor(batch, dtype=torch.int32),
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            if is_val_batch:
                filename = f"val_batch_{idx}.bin"
            else:
                filename = f"train_batch_{idx - num_fm_val_batches}.bin"
            torch.save(batch, f"data/{filename}")
            api.upload_file(f"data/{filename}", filename, repo_id=repo_id)
        idx += 1
    
    # Now, turn the rest of the fineweb-edu-100BT tokens into their own batches with create_batch
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        if len(tokens_fw) < B*T:
            continue
        for i in range(0, len(tokens_fw), B*T):
            batch = tokens_fw[i:i+B*T].view(B, T)
            batch = create_batch(
                tokens=batch,
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            filename = f"train_batch_{idx}.bin"
            torch.save(batch, f"data/{filename}")
            api.upload_file(f"data/{filename}", filename, repo_id=repo_id)
            num_fw_tokens_train += B*T
            idx += 1

    # For finemath, the validation data is created above
    # For fineweb, just use the validation set by karpathy
    dl = distributed_data_generator("fineweb100B/fineweb_val_*.bin")
    tokens_fw = None
    idx = 0
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens]) if tokens_fw else new_tokens
        if len(tokens_fw) < B*T:
            continue
        for i in range(0, len(tokens_fw), B*T):
            batch = tokens_fw[i:i+B*T].view(B, T)
            batch = create_batch(
                tokens=batch,
                bytes_per_token=bytes_per_token,
                pad_byte=pad_byte,
                eot_byte=eot_byte,
                tokens_to_bytes_right_pad=tokens_to_bytes_right_pad,
                tokens_to_bytes_left_pad=tokens_to_bytes_left_pad,
            )
            filename = f"val_batch_{idx}.bin"
            torch.save(batch, f"data/{filename}")
            api.upload_file(f"data/{filename}", filename, repo_id=repo_id)
            num_fw_tokens_val += B*T
            idx += 1

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
