
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


def create_batch(
        tokens: torch.Tensor,
        bytes_per_token: int,
        pad_byte: int,
        eot_byte: int,
        bytes_to_tokens: nn.Embedding,
) -> torch.Tensor:
    B, T = tokens.size()
    byte_tensor = tokens_to_bytes(tokens, bytes_to_tokens)
    byte_tensor_pulled = pull_from_right(byte_tensor, bytes_per_token, pad_byte, eot_byte)
    full_tensor = torch.cat([
            tokens.unsqueeze(-1),
            byte_tensor.view(B, T, bytes_per_token),
            byte_tensor_pulled.view(B, T, bytes_per_token),
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


def create_data(
        B: int = 1024,
        T: int = 1024,
        bytes_per_token: int = 16,
        pad_byte: int = 456,
        eot_byte: int = 457,
        vocab_size: int = 50257,
):
    eot_token = vocab_size - 1
    bytes_to_tokens = make_embedding(f"ttb_{bytes_per_token}_right.json", vocab_size)
    embedding = tiktoken.encoding_for_model("gpt-2")
    dl = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin")
    tokens_fw = next(dl)

    # Download, tokenize, and save the finemath data, and fill it up to T with random fineweb samples
    batch = []
    idx = 0
    for row in load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True):
        text = row["text"]
        tokens_fm = embedding.encode(text)

        # Don't use incomplete finemath samples
        if len(tokens_fm) > T:
            continue

        # The sample will be filled to T with a random fineweb slice;
        # There has to be an EOT token between them.
        if len(tokens_fm) < T and tokens_fm[-1] != eot_token:
            tokens_fm = torch.cat([tokens_fm, torch.tensor([eot_token]).type_as(tokens_fm).squeeze()])

        if len(tokens_fm) == T:
            batch.append(tokens_fm.tolist())
        else:
            num_tokens_missing = T - len(tokens_fm)  # 0 <= num_tokens_missing <= T, see condition above
            while len(tokens_fw) < num_tokens_missing:
                tokens_fw = torch.cat([tokens_fw, next(dl)])

            fillup_tokens, tokens_fw = tokens_fw[:num_tokens_missing], tokens_fw[num_tokens_missing:]
            batch.append(torch.cat([tokens_fm, fillup_tokens]).tolist())
        
        batch = create_batch(torch.tensor(batch).astype(torch.int32), bytes_per_token, pad_byte, eot_byte, bytes_to_tokens)
        torch.save(batch, f"data/train_batch_{idx}.bin")
        idx += 1
    
    # Now, turn the rest of the fineweb-edu-100BT tokens into their own batches with create_batch
    for new_tokens in dl:
        tokens_fw = torch.cat([tokens_fw, new_tokens])
        for i in range(0, len(tokens_fw), B*T):
            batch = tokens_fw[i:i+B*T].view(B, T)
            batch = create_batch(batch, bytes_per_token, pad_byte, eot_byte, bytes_to_tokens)
            torch.save(batch, f"data/train_batch_{idx}.bin")
            idx += 1


########################################
###### UPLOAD DATA TO HUGGINGFACE ######
########################################


def upload_data():
    api = HfApi()
    for filename in glob.glob("data/train_batch_*.bin"):
        api.upload_file(filename, filename, repo_id="snimu/finemath-fineweb-100B-data-for-MoT")
        


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
    bytes_to_tokens = make_embedding(f"ttb_{bytes_per_token}_left.json", vocab_size)
    byte_tensor = tokens_to_bytes(tokens, bytes_to_tokens)
    batch = create_batch(tokens, bytes_per_token, pad_byte, eot_byte, bytes_to_tokens)
    print(f"{tokens.shape=}\n{batch.shape=}\n\nTOKENS")
    print(tokens)
    print("\n\nBYTES")
    print(byte_tensor.view(B, T, bytes_per_token))
    print("\n\nBATCH")
    print(batch)
    print("\n\n")


if __name__ == "__main__":
    create_data()
    upload_data()
