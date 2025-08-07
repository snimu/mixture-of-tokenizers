"""Create the token-to-byte (or character) mappings."""

import argparse
import json
from typing import Literal

import tiktoken


def create_ttb(bpt: int = 16, pad_position: Literal["left", "right"] = "left"):
    """Create the token-to-byte (or character) mappings."""
    with open("embeddings/byte_to_int.json", "r") as f:
        byte_to_int = json.loads(f.read())
    byte_to_int = {b: int(i) for b, i in byte_to_int.items()}

    encoding = tiktoken.encoding_for_model("gpt-2")
    ttb = {}
    for index in range(encoding.max_token_value):
        token = encoding.decode([index])
        if token == "<|endoftext|>":
            ttb[index] = [byte_to_int["endoftext"]] * bpt
            continue
        b_seq = [byte_to_int[b] for b in token]
        b_seq = b_seq[:bpt]  # Cut the sequence to bpt by dropping the last bytes (chars)
        if pad_position == "left":
            b_seq = [byte_to_int["pad"]] * (bpt - len(b_seq)) + b_seq
        elif pad_position == "right":
            b_seq = b_seq + [byte_to_int["pad"]] * (bpt - len(b_seq))
        else:
            raise ValueError(f"Invalid pad_position: {pad_position}")
        ttb[index] = b_seq
    with open(f"embeddings/ttb_{bpt}_{pad_position}.json", "w") as f:
        f.write(json.dumps(ttb))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpt", type=int, default=16)
    parser.add_argument("--pad_position", choices=["left", "right"], default="left")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_ttb(args.bpt, args.pad_position)
