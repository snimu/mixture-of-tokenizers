"""Just for testing the data loader, the dataloader will later be in train_gpt.py"""

import random
import subprocess as sp
from pathlib import Path
from time import perf_counter

import torch

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
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * batch_size * seq_len * (1 + bytes_per_token * 4), "number of tokens read does not match provided parameters"
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
    tokens, pos = _load_data_shard_bytes(next(file_iter), seq_len, batch_size, bytes_per_token), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard_bytes(next(file_iter), seq_len, batch_size, bytes_per_token), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        tokens = buf[:, :-1, 0].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_left_padded = buf[:, :-1, 1:17].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_pulled_left = buf[:, :-1, 17:33].to(device="cuda", dtype=torch.int32, non_blocking=True)
        bytes_right_padded = buf[:, 1:, 33:49].to(device="cuda", dtype=torch.int64, non_blocking=True)
        bytes_pulled_right = buf[:, 1:, 49:65].to(device="cuda", dtype=torch.int64, non_blocking=True)
        yield tokens, bytes_left_padded, bytes_pulled_left, bytes_right_padded, bytes_pulled_right


def download_test_data():
    sp.run(["bash", "fineweb100B.sh", "16"])
    download(num_train_files=16, num_fm_val_files=0, num_fw_val_files=0)


def test_timing():
    dg = distributed_data_generator("fineweb100B/fineweb_train_*.bin", 1024, 0, 1)
    t0 = perf_counter()
    for _ in range(10):
        next(dg)
    print("Time tokens: ", perf_counter() - t0)

    dg = distributed_data_generator_bytes("data/train_batch_*.bin", 1024, 1024, 16, 0, 1)
    t0 = perf_counter()
    for _ in range(10):
        next(dg)
    print("Time bytes: ", perf_counter() - t0)


if __name__ == "__main__":
    download_test_data()
    test_timing()
