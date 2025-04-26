"""Just for testing the data loader, the dataloader will later be in train_gpt.py"""

import os
import argparse
import json
import random
import subprocess as sp
from pathlib import Path
from time import perf_counter

import einops
import torch
import tiktoken
from tqdm import tqdm

from data_download import download
from data_creation import make_embedding, tokens_to_bytes, pull_from_left, pull_from_right


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


def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = (batch_size * seq_len) // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + (batch_size * seq_len) + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size].view(-1, seq_len)
        inputs = buf[:, :-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[:, 1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size * seq_len
        yield inputs, targets


def _load_data_shard_bytes(file_iter):
    while True:
        try:
            file = next(file_iter)
            dtype = torch.int32 if "bytes/" in file.name else torch.uint16
            return _load_data_shard(file, dtype=dtype).to(torch.int32)
        except AssertionError:
            pass


def distributed_data_generator_bytes(
        filename_patterns: str | list[str],
        seq_len: int,
        batch_size: int,
        bytes_per_token: int,
        rank : int,
        world_size : int,
        vocab_size: int = 50257,
        return_bytes_left_padded: bool = True,
        return_bytes_left_pulled: bool = True,
        return_bytes_right_padded: bool = True,
        return_bytes_right_pulled: bool = True,
        device: torch.device = "cpu",
        seed: int = 12345,
):
    assert not (return_bytes_left_pulled and not return_bytes_left_padded)
    assert not (return_bytes_right_pulled and not return_bytes_right_padded)
    ttb_left_pad = make_embedding(f"ttb_{bytes_per_token}_left_pad.json", vocab_size).to(device)
    ttb_right_pad = make_embedding(f"ttb_{bytes_per_token}_right_pad.json", vocab_size).to(device)

    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    files = sorted(Path.cwd().glob(filename_patterns[0]))
    for filename_pattern in filename_patterns[1:]:
        files.extend(sorted(Path.cwd().glob(filename_pattern)))
    random.seed(seed)  # ensure that all shards are shuffled the same way
    random.shuffle(files)
    assert batch_size % world_size == 0
    local_batch_size = (batch_size * seq_len) // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    data, pos = _load_data_shard_bytes(file_iter), 0
    while True:
        if pos + batch_size * seq_len + 1 >= len(data):
            data, pos = _load_data_shard_bytes(file_iter), 0
        tokens = data[pos + rank * local_batch_size:][:local_batch_size].view(-1, seq_len).to(device)
        # bytes_left_padded = einops.rearrange(ttb_left_pad(tokens), "B T bpt -> B (T bpt)") if return_bytes_left_padded else None
        bytes_left_padded = tokens_to_bytes(tokens, ttb_left_pad)
        bytes_left_pulled = pull_from_left(bytes_left_padded, bytes_per_token, 456, 457) if return_bytes_left_pulled else None
        # bytes_right_padded = einops.rearrange(ttb_right_pad(tokens), "B T bpt -> B (T bpt)") if return_bytes_right_padded else None
        bytes_right_padded = tokens_to_bytes(tokens, ttb_right_pad)
        bytes_right_pulled = pull_from_right(bytes_right_padded, bytes_per_token, 456, 457) if return_bytes_right_pulled else None
        pos += batch_size * seq_len
        yield (
            tokens,
            einops.rearrange(bytes_left_padded, "B (T bpt) -> B T bpt", bpt=bytes_per_token) if bytes_left_padded is not None else None,
            einops.rearrange(bytes_left_pulled, "B (T bpt) -> B T bpt", bpt=bytes_per_token) if bytes_left_pulled is not None else None,
            einops.rearrange(bytes_right_padded, "B (T bpt) -> B T bpt", bpt=bytes_per_token) if bytes_right_padded is not None else None,
            einops.rearrange(bytes_right_pulled, "B (T bpt) -> B T bpt", bpt=bytes_per_token) if bytes_right_pulled is not None else None,
        )


def load_byte_decoder() -> dict[int, str]:
    with open("embeddings/int_to_byte.json", "r") as f:
        text = f.read()
    btt = {int(k): v for k, v in json.loads(text).items()}
    return btt


def decode_bytes(byte_tensor: torch.Tensor, byte_decoder: dict[int, str], bytes_per_token: int = 16) -> str:
    text = ""
    for bts in byte_tensor.squeeze().tolist():
        text += "("
        for b in bts:
            char = byte_decoder[int(b)]
            text += char if char != "pad" else ":"
        text += ") "
    return text


def download_test_data():
    if not os.path.exists("fineweb100B") or len(os.listdir("fineweb100B")) < 1029:  # 1028 train, 1 val
        sp.run(["bash", "fineweb100B.sh"])
    download(tokens_or_bytes="tokens")


def time_bytes(
        n_batches: int,
        return_bytes_left_padded: bool = True,
        return_bytes_left_pulled: bool = True,
        return_bytes_right_padded: bool = True,
        return_bytes_right_pulled: bool = True,
        device: torch.device = "cpu",
):
    print(f"\n{n_batches=}")
    print(f"{return_bytes_left_padded=}, {return_bytes_left_pulled=}, {return_bytes_right_padded=}, {return_bytes_right_pulled=}")
    dg = distributed_data_generator_bytes(
        filename_patterns=["data/tokens/train/*.bin", "fineweb100B/fineweb_train_*.bin"],
        seq_len=1024,
        batch_size=1024,
        bytes_per_token=16,
        rank=0,
        world_size=1,
        return_bytes_left_padded=return_bytes_left_padded,
        return_bytes_left_pulled=return_bytes_left_pulled,
        return_bytes_right_padded=return_bytes_right_padded,
        return_bytes_right_pulled=return_bytes_right_pulled,
        device=device,
    )
    if n_batches < 0:
        n_batches = 50271
    t0 = perf_counter()
    for _ in tqdm(range(n_batches)):
        try:
            _, _, _, _, _ = next(dg)
        except (StopIteration, RuntimeError):
            break
    print(f"Time bytes:{perf_counter() - t0:.2f}s\n")


def test_timing(device: torch.device = "cpu"):
    dg = distributed_data_generator("fineweb100B/fineweb_train_*.bin", 1024, 1024, 0, 1)
    t0 = perf_counter()
    n_toks = 0
    for _ in range(16):
        x, _ = next(dg)
        n_toks += len(x) + 1  # +1 because x cuts off one of the tokens, but the byte loader doesn't
    print("Time tokens: ", perf_counter() - t0)

    n_batches = n_toks // 1024
    time_bytes(n_batches, device=device)
    time_bytes(n_batches, return_bytes_left_pulled=False, device=device)
    time_bytes(n_batches, return_bytes_left_padded=False, return_bytes_left_pulled=False, device=device)
    time_bytes(n_batches, return_bytes_left_padded=False, return_bytes_left_pulled=False, return_bytes_right_pulled=False, device=device)
    time_bytes(n_batches, return_bytes_left_padded=False, return_bytes_left_pulled=False, return_bytes_right_padded=False, return_bytes_right_pulled=False, device=device)


def time_full_dataset(device: torch.device = "cpu"):
    time_bytes(n_batches=-1, device=device)
    time_bytes(n_batches=-1, return_bytes_left_pulled=False, device=device)
    time_bytes(n_batches=-1, return_bytes_left_padded=False, return_bytes_left_pulled=False, device=device)
    time_bytes(n_batches=-1, return_bytes_left_padded=False, return_bytes_left_pulled=False, return_bytes_right_pulled=False, device=device)
    time_bytes(n_batches=-1, return_bytes_left_padded=False, return_bytes_left_pulled=False, return_bytes_right_padded=False, return_bytes_right_pulled=False, device=device)


def count_batches(device: torch.device = "cpu"):
    dg = distributed_data_generator_bytes(
        filename_patterns=["data/tokens/train/*.bin", "fineweb100B/fineweb_train_*.bin"],
        seq_len=1024,
        batch_size=1024,
        bytes_per_token=16,
        rank=0,
        world_size=1,
        return_bytes_left_padded=False,
        return_bytes_left_pulled=False,
        return_bytes_right_padded=False,
        return_bytes_right_pulled=False,
        device=device,
    )
    n_batches = 0
    try:
        for _ in dg:
            n_batches += 1
    except (StopIteration, RuntimeError):
        pass
    print(f"\n\n{n_batches=}\n\n")

def check_plausibility(device: torch.device = "cpu"):
    dg = distributed_data_generator_bytes("data/tokens/train/*.bin", 1024, 1024, 16, 0, 1, device=device)
    entry = random.randint(0, 1023)
    tokens, bytes_left_padded, bytes_left_pulled, bytes_right_padded, bytes_right_pulled = next(dg)
    encoding = tiktoken.encoding_for_model("gpt-2")
    print("\n\nTOKENS DECODED:\n\n", encoding.decode(tokens[entry].tolist()))

    byte_decoder = load_byte_decoder()
    print("\n\nBYTES LEFT DECODED:\n\n", decode_bytes(bytes_left_padded[entry], byte_decoder))
    print("\n\nBYTES PULLED LEFT DECODED:\n\n", decode_bytes(bytes_left_pulled[entry], byte_decoder))

    print("\n\nBYTES RIGHT DECODED:\n\n", decode_bytes(bytes_right_padded[entry], byte_decoder))
    print("\n\nBYTES PULLED RIGHT DECODED:\n\n", decode_bytes(bytes_right_pulled[entry], byte_decoder))
 
    assert tuple(tokens.shape) == (1024, 1024), f"{tokens.shape=}"
    assert tuple(bytes_left_padded.shape) == (1024, 1024, 16), f"{bytes_left_padded.shape=}"
    assert tuple(bytes_left_pulled.shape) == (1024, 1024, 16), f"{bytes_left_pulled.shape=}"
    assert tuple(bytes_right_padded.shape) == (1024, 1024, 16), f"{bytes_right_padded.shape=}"
    assert tuple(bytes_right_pulled.shape) == (1024, 1024, 16), f"{bytes_right_pulled.shape=}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-timing", action="store_true")
    parser.add_argument("--check-plausibility", action="store_true")
    parser.add_argument("--time-full-dataset", action="store_true")
    parser.add_argument("--count-batches", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    download_test_data()
    if args.test_timing:
        test_timing(device=args.device)
    if args.check_plausibility:
        check_plausibility(device=args.device)
    if args.time_full_dataset:
        time_full_dataset(device=args.device)
    if args.count_batches:
        count_batches(device=args.device)
