"""Just for testing the data loader, the dataloader will later be in train_gpt.py"""

import json
import random
import subprocess as sp
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
import tiktoken

from data_download import download


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

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets


def _load_data_shard_bytes(file: Path, seq_len: int, batch_size: int, bytes_per_token: int):
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty((batch_size, seq_len, 1 + bytes_per_token * 4), dtype=torch.int32, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
    return tokens

def distributed_data_generator_bytes(
        filename_pattern: str,
        seq_len: int,
        batch_size: int,
        bytes_per_token: int,
        rank : int,
        world_size : int,
):
    files = sorted(Path.cwd().glob(filename_pattern))
    random.shuffle(files)
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    data, pos = _load_data_shard_bytes(next(file_iter), seq_len, batch_size, bytes_per_token), 0
    while True:
        if pos + batch_size + 1 >= len(data):
            data, pos = _load_data_shard_bytes(next(file_iter), seq_len, batch_size, bytes_per_token), 0
        buf = data[pos + rank * local_batch_size:][:local_batch_size + 1]
        tokens = buf[:, :-1, 0].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_left_padded = buf[:, :-1, 1:17].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_pulled_left = buf[:, :-1, 17:33].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_right_padded = buf[:, 1:, 33:49].to(device="cuda", dtype=torch.int64, non_blocking=True)
        bytes_pulled_right = buf[:, 1:, 49:65].to(device="cuda", dtype=torch.int64, non_blocking=True)
        yield tokens, bytes_left_padded, bytes_pulled_left, bytes_right_padded, bytes_pulled_right


# TODO: one dataloader for each combination of padding & pulling
# TODO: test that the tokens are reasonable (decode them) & that the bytes are reasonable (decode them, too)


def load_byte_decoder(alignment: Literal["left", "right"], bytes_per_token: int = 16):
    with open(f"embeddings/ttb_{bytes_per_token}_{alignment}.json", "r") as f:
        text = f.read()
    ttb = json.loads(text)
    btt = {int(v): k for k, v in ttb.items()}
    return btt


def decode_bytes(byte_tensor: torch.Tensor, byte_decoder: dict[int, str], bytes_per_token: int = 16) -> str:
    bts = byte_tensor.tolist()
    text = ""

    for i in range(0, len(bts), bytes_per_token):
        text += byte_decoder[bts[i:i+bytes_per_token]]
    return text


def download_test_data():
    sp.run(["bash", "fineweb100B.sh", "64"])
    download(num_train_files=64, num_fm_val_files=0, num_fw_val_files=0)


def test_timing():
    dg = distributed_data_generator("fineweb100B/fineweb_train_*.bin", 1024, 0, 1)
    t0 = perf_counter()
    n_toks = 0
    for _ in range(16):
        x, _ = next(dg)
        n_toks += len(x) + 1  # +1 because x cuts off one of the tokens, but the byte loader doesn't
    print("Time tokens: ", perf_counter() - t0)

    dg = distributed_data_generator_bytes("data/train_batch_*.bin", 1024, 1024, 16, 0, 1)
    t0 = perf_counter()
    n_toks_b = 0
    while n_toks_b < n_toks:
        tokens, _, _, _, _ = next(dg)
        n_toks_b += len(tokens)
    print("Time bytes: ", perf_counter() - t0)


def check_plausibility():
    dg = distributed_data_generator_bytes("data/train_batch_*.bin", 1024, 1024, 16, 0, 1)
    tokens, bytes_left_padded, bytes_pulled_left, bytes_right_padded, bytes_pulled_right = next(dg)
    encoding = tiktoken.encoding_for_model("gpt-2")
    print("\n\nTOKENS DECODED:\n\n", encoding.decode(tokens.tolist()))

    byte_decoder_left = load_byte_decoder("left")
    print("\n\nBYTES LEFT DECODED:\n\n", decode_bytes(bytes_left_padded, byte_decoder_left))
    print("\n\nBYTES PULLED LEFT DECODED:\n\n", decode_bytes(bytes_pulled_left, byte_decoder_left))

    byte_decoder_right = load_byte_decoder("right")
    print("\n\nBYTES RIGHT DECODED:\n\n", decode_bytes(bytes_right_padded, byte_decoder_right))
    print("\n\nBYTES PULLED RIGHT DECODED:\n\n", decode_bytes(bytes_pulled_right, byte_decoder_right))
    assert len(tokens) == 1024


if __name__ == "__main__":
    download_test_data()
    test_timing()
    check_plausibility()
