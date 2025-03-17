"""Just for testing the data loader, the dataloader will later be in train_gpt.py"""

import json
import random
import subprocess as sp
from pathlib import Path
from time import perf_counter

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
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
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
        return_tokens: bool = True,
        return_bytes_left_padded: bool = True,
        return_bytes_pulled_left: bool = True,
        return_bytes_right_padded: bool = True,
        return_bytes_pulled_right: bool = True,
):
    def slice_tensor(tensor: torch.Tensor, dtype: torch.dtype, doit: bool = True) -> torch.Tensor:
        if doit:
            return tensor.to(device="cuda", dtype=dtype, non_blocking=True)
        else:
            return None
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
        tokens = slice_tensor(buf[:, :-1, 0], torch.int32, return_tokens)
        bytes_left_padded = slice_tensor(buf[:, :-1, 1:17],torch.int32, return_bytes_left_padded)
        bytes_pulled_left = slice_tensor(buf[:, :-1, 17:33], torch.int32, return_bytes_pulled_left)
        bytes_right_padded = slice_tensor(buf[:, 1:, 33:49], torch.int64, return_bytes_right_padded)
        bytes_pulled_right = slice_tensor(buf[:, 1:, 49:65], torch.int64, return_bytes_pulled_right)
        pos += batch_size
        yield tokens, bytes_left_padded, bytes_pulled_left, bytes_right_padded, bytes_pulled_right


def load_byte_decoder() -> dict[int, str]:
    with open("embeddings/int_to_byte.json", "r") as f:
        text = f.read()
    btt = {int(k): v for k, v in json.loads(text).items()}
    return btt


def decode_bytes(byte_tensor: torch.Tensor, byte_decoder: dict[int, str], bytes_per_token: int = 16) -> str:
    bts = byte_tensor.squeeze().tolist()
    bts = [char for chars in bts for char in chars]
    text = ""

    for char in bts:
        text += byte_decoder[int(char)]
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
    entry = random.randint(0, 1023)
    tokens, bytes_left_padded, bytes_pulled_left, bytes_right_padded, bytes_pulled_right = next(dg)
    encoding = tiktoken.encoding_for_model("gpt-2")
    print("\n\nTOKENS DECODED:\n\n", encoding.decode(tokens[entry].tolist()))

    byte_decoder = load_byte_decoder()
    print("\n\nBYTES LEFT DECODED:\n\n", decode_bytes(bytes_left_padded[entry], byte_decoder))
    print("\n\nBYTES PULLED LEFT DECODED:\n\n", decode_bytes(bytes_pulled_left[entry], byte_decoder))

    print("\n\nBYTES RIGHT DECODED:\n\n", decode_bytes(bytes_right_padded[entry], byte_decoder))
    print("\n\nBYTES PULLED RIGHT DECODED:\n\n", decode_bytes(bytes_pulled_right[entry], byte_decoder))
    assert len(tokens) == 1023  # tokens etc are all cut off by one


if __name__ == "__main__":
    download_test_data()
    test_timing()
    check_plausibility()
