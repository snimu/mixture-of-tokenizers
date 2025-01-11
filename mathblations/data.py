# /// script
# requires-python = "==3.12"
# dependencies = [
#   "numpy",
#   "torch",
# ]
# ///

"""Generating & loading the dataset."""

import argparse
import random
import math
from typing import Literal, Generator

import torch
from tqdm import tqdm

import model


class GenerateEquations:
    """
    Generate equations.

    Things to vary:

    - The number of characters per token
    - The number of tokens per equation
    - The mathematical operator
    - Whether to do it with modulo
    """
    def __init__(
            self,
            max_digits_per_token: int = 3,
            max_tokens_per_num: int = 10,
            op: Literal["+", "-", "*", "/"] = "+",
            mod: int | None = None,
    ) -> None:
        assert max_digits_per_token > 0, f"max_digits_per_token must be > 0 (got {max_digits_per_token})"
        assert max_tokens_per_num > 0, f"max_tokens_per_num must be > 0 (got {max_tokens_per_num})"
        assert op in ("+", "-", "*", "/"), f"op must be one of '+', '-', '*', '/' (got {op})"
        if mod is not None:
            assert mod > 0, f"mod must be > 0 or None (got {mod})"
        self.max_digits_per_token = max_digits_per_token
        self.max_tokens_per_num = max_tokens_per_num
        self.op = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x // y,
        }[op]
        self.op_name = op
        self.mod = mod

        self.num_numeric_tokens = 10**self.max_digits_per_token
        self.max_single_token_number = int("9"*self.max_digits_per_token)
        self.max_number = int("9"*self.max_digits_per_token*self.max_tokens_per_num)
        
        self.op_token = self.max_single_token_number + 1
        self.eq_token = self.max_single_token_number + 2
        self.pad_token = self.max_single_token_number + 3
        if op == "+":
            max_y = self.max_number * 2
        elif op in ("-", "/"):
            max_y = self.max_number
        else:
            max_y = self.max_number ** 2
        max_digits_in_y = len(str(max_y))
        max_y_tokens = math.ceil(max_digits_in_y / self.max_digits_per_token)
        # nums + result + op & eq sign
        self.max_possible_num_tokens = 2 * self.max_tokens_per_num + max_y_tokens + 2

        self.vocab_size = self.max_single_token_number + 4  # nums + 0 & op & eq sign & pad token
        
    def num_to_tokens(
            self, 
            num: int,
            max_digits_per_token: int,
            max_number: int,
    ) -> list[float]:
        if num <= max_number:
            return [num]
        
        tokens = []
        num_str = str(num)
        for i in range(0, len(num_str), max_digits_per_token):
            part = num_str[i:i+max_digits_per_token]
            tokens.append(int("".join(part)))
        return tokens
    
    def tokens_to_digits(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.tolist()
        digits = []
        for token in tokens:
            new_toks = [self.pad_token] * self.max_digits_per_token
            if token == self.op_token:
                new_toks[-1] = 10
            elif token == self.eq_token:
                new_toks[-1] = 11
            elif token == self.pad_token:
                new_toks[-1] = 12
            else:
                token_str = str(token)
                for i, char in enumerate(reversed(token_str)):
                    new_toks[-i-1] = int(char)

            digits.extend(new_toks)
        return torch.tensor(digits)

    def generate_equation(self) -> torch.Tensor:
        """
        Returns:
        - equation tokens: torch.Tensor
        - output_indices: torch.Tensor (equivalent to tuple (start, end))
        """
        n1, n2 = random.randint(0, self.max_number), random.randint(0, self.max_number)
        y = self.op(n1, n2)
        if self.mod is not None:
            y %= self.mod
        
        n1 = self.num_to_tokens(n1, self.max_digits_per_token, self.max_single_token_number)
        n2 = self.num_to_tokens(n2, self.max_digits_per_token, self.max_single_token_number)
        y = self.num_to_tokens(y, self.max_digits_per_token, self.max_single_token_number)

        start = len(n1) + len(n2) + 2  # nums + op & eq sign
        end = start + len(y)
        return torch.tensor(n1 + [self.op_token] + n2 + [self.eq_token] + y), (start, end)

    def eq_to_str(self, equation: torch.Tensor) -> str:
        equation = equation.squeeze().tolist()
        equation = [token for token in equation if token != self.pad_token]
        equation = "".join(str(x) for x in equation)
        equation = equation.replace(str(self.op_token), self.op_name)
        equation = equation.replace(str(self.eq_token), "=")
        # Nicer formatting for large numbers
        try:
            n1, rest = equation.split(str(self.op_name))
            n2, y = rest.split("=")
            n1, n2, y = int(n1), int(n2), int(y)
            return f"{n1:,} {self.op_name} {n2:,} = {y:,}"
        except ValueError:  # sometimes, n2 is just ''; always n2...
            return f"{equation}"
    
    def __call__(self, *args, **kwds):
        """Returns: (x_tokens, x_digit_tokens, y_tokens, y_indices)"""
        eq_tokens, indices = self.generate_equation()

        # Pad to max length
        all_tokens = torch.full((self.max_possible_num_tokens,), self.pad_token)
        all_tokens[:len(eq_tokens)] = eq_tokens

        # Split into x and y
        x_tokens = all_tokens[:-1]
        y_tokens = all_tokens[1:]

        # Convert to digits
        x_digit_tokens = self.tokens_to_digits(x_tokens)

        # Shift indices to fit y
        y_indices = torch.tensor([indices[0] - 1, indices[1] - 1])
        return x_tokens, x_digit_tokens, y_tokens, y_indices


def make_dataset(
        gen: GenerateEquations, args: argparse.Namespace, loop: tqdm = None,
) -> tuple[dict[Literal["x_tokens", "x_digit_tokens", "y_tokens", "y_indices"], list], ...]:
    # TODO: continually save dataset to json files of batchsize, load them async during training
    loop.write(
        f"\n\nCREATING DATASET: max_digits_per_token={gen.max_digits_per_token}, "
        f"max_tokens_per_num={gen.max_tokens_per_num}, op={gen.op_name}, mod={gen.mod}\n\n"
    )
    trainset = dict(x_tokens=[], x_digit_tokens=[], y_tokens=[], y_indices=[])
    for i in range(args.num_steps * args.batchsize):
        x_tokens, x_digit_tokens, y_tokens, y_indices = gen()
        trainset["x_tokens"].append(x_tokens)
        trainset["x_digit_tokens"].append(x_digit_tokens)
        trainset["y_tokens"].append(y_tokens)
        trainset["y_indices"].append(y_indices)
        if loop and i % 100 == 0:
            loop.set_description(f"Trainset: {((i+1)/(args.num_steps*args.batchsize))*100:.2f}%")

    valset = dict(x_tokens=[], x_digit_tokens=[], y_tokens=[], y_indices=[])
    for i in range(args.num_steps_val * args.batchsize):
        x_tokens, x_digit_tokens, y_tokens, y_indices = gen()
        valset["x_tokens"].append(x_tokens)
        valset["x_digit_tokens"].append(x_digit_tokens)
        valset["y_tokens"].append(y_tokens)
        valset["y_indices"].append(y_indices)
        if loop and i % 100 == 0:
            loop.set_description(f"Valset: {((i+1)/(args.num_steps_val*args.batchsize))*100:.2f}%")

    return trainset, valset


def iterate_dataset(
        dataset: dict[Literal["x_tokens", "x_digit_tokens", "y_tokens", "y_indices"], list],
        args: argparse.Namespace,
        config: model.GPTConfig,
) -> Generator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor], None, None]:
    num_samples = len(dataset["x_tokens"])
    for i in range(0, num_samples, args.batchsize):
        batch_slice = slice(i, i + args.batchsize)
        yield (
            torch.stack(dataset["x_tokens"][batch_slice]).to(args.device),
            torch.stack(dataset["x_digit_tokens"][batch_slice]).to(args.device) if config.use_digits else None,
            torch.stack(dataset["y_tokens"][batch_slice]).to(args.device),
            torch.stack(dataset["y_indices"][batch_slice]).to(args.device)
        )


def slice_logits_and_targets(
        logits: torch.Tensor, y_indices: torch.Tensor, y_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Handle each batch element separately
    batch_logits = [logits[i, start:end] for i, (start, end) in enumerate(y_indices)]
    batch_targets = [y_tokens[i, start:end] for i, (start, end) in enumerate(y_indices)]
    
    # Stack them all together
    return torch.cat(batch_logits), torch.cat(batch_targets)


if __name__ == "__main__":
    from time import perf_counter
    gen = GenerateEquations(max_digits_per_token=3, max_tokens_per_num=2, op="+", mod=None)
    batch_size = 10000
    # Measure time linear
    t0 = perf_counter()
    for _ in range(batch_size):
        _ = gen()
    print(f"Time for {batch_size} equations: {perf_counter() - t0:.2f} s")

    # Create a few equations
    for _ in range(3):
        x_tokens, x_digit_tokens, y_tokens, y_indices = gen()
        x_tokens = x_tokens.tolist()
        y_tokens = y_tokens.tolist()
        x_digit_tokens = x_digit_tokens.tolist()
        all_tokens = torch.tensor(x_tokens + [y_tokens[-1]])
        print()
        print(f"{x_tokens=}")
        print(f"{y_tokens=}")
        print(f"{x_digit_tokens=}")
        print(f"equation={gen.eq_to_str(all_tokens)}")
