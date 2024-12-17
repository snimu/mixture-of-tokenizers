# /// script
# requires-python = "==3.12"
# dependencies = [
#   "torch",
#   "wandb",
#   "polars",
#   "tqdm",
#   "wandb",
# ]
# ///

import argparse
import itertools
import random
from typing import Any, Literal, Generator
from dataclasses import dataclass
from pathlib import Path

import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import polars as pl

import data
import model
from muon import Muon


def _to_list(item: Any, dtype: type) -> list:
    if isinstance(item, dtype):
        return [item]
    elif isinstance(item, list):
        return item
    else:
        raise ValueError(f"Expected {dtype} or list of {dtype}, got {item}")


def get_args():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)

    # Data parameters
    parser.add_argument("--max-digits-per-token", type=int, default=3, nargs="+")
    parser.add_argument("--max-tokens-per-num", type=int, default=1, nargs="+")
    parser.add_argument("--op", type=str, choices=("+", "-", "*", "/"), default="+", nargs="+")
    parser.add_argument("--mod", type=int, default=None, nargs="+")

    # Training parameters
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=50_000)
    parser.add_argument("--num-steps-val", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cooldown-steps", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--clip-min", type=int, default=0)
    parser.add_argument("--clip-max", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=100)
    
    # Ablation parameters
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=385)

    # Model parameters
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--sliding-window-size", type=int, default=100)

    
    args = parser.parse_args()

    args.max_digits_per_token = _to_list(args.max_digits_per_token, int)
    args.max_tokens_per_num = _to_list(args.max_tokens_per_num, int)
    args.op = _to_list(args.op, str)
    if args.mod is not None:
        args.mod = _to_list(args.mod, int)

    assert args.print_every % args.eval_every == 0, \
        f"print_every ({args.print_every}) must be a multiple of eval_every ({args.eval_every})"
    
    return args


def make_dataset(
        gen: data.GenerateEquations, args: argparse.Namespace
) -> tuple[dict[Literal["x_tokens", "x_digit_tokens", "y_tokens", "y_indices"], list], ...]:
    trainset = dict(x_tokens=[], x_digit_tokens=[], y_tokens=[], y_indices=[])
    for _ in range(args.num_steps * args.batchsize):
        x_tokens, x_digit_tokens, y_tokens, y_indices = gen()
        trainset["x_tokens"].append(x_tokens)
        trainset["x_digit_tokens"].append(x_digit_tokens)
        trainset["y_tokens"].append(y_tokens)
        trainset["y_indices"].append(y_indices)

    valset = dict(x_tokens=[], x_digit_tokens=[], y_tokens=[], y_indices=[])
    for _ in range(args.num_steps_val * args.batchsize):
        x_tokens, x_digit_tokens, y_tokens, y_indices = gen()
        valset["x_tokens"].append(x_tokens)
        valset["x_digit_tokens"].append(x_digit_tokens)
        valset["y_tokens"].append(y_tokens)
        valset["y_indices"].append(y_indices)

    return trainset, valset


def iterate_dataset(
        dataset: dict[Literal["x_tokens", "x_digit_tokens", "y_tokens", "y_indices"], list],
        args: argparse.Namespace,
) -> Generator[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor], None, None]:
    loop = zip(
        dataset["x_tokens"], dataset["x_digit_tokens"], dataset["y_tokens"], dataset["y_indices"], 
        strict=True,
    )
    loop = itertools.batched(loop, n=args.batchsize)
    for x_tokens, x_digit_tokens, y_tokens, y_indices in loop:
        x_tokens = torch.tensor(x_tokens, dtype=torch.long).to(args.device)
        x_digit_tokens = torch.tensor(x_digit_tokens, dtype=torch.long).to(args.device) if args.use_digits else None
        y_tokens = torch.tensor(y_tokens, dtype=torch.long).to(args.device)
        y_indices = torch.tensor(y_indices, dtype=torch.long).to(args.device)
        yield x_tokens, x_digit_tokens, y_tokens, y_indices


def slice_logits_and_targets(
        logits: torch.Tensor, y_indices: torch.Tensor, y_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Handle each batch element separately
    batch_logits = [logits[i, start:end] for i, (start, end) in enumerate(y_indices)]
    batch_targets = [y_tokens[i, start:end] for i, (start, end) in enumerate(y_indices)]
    
    # Stack them all together
    return torch.cat(batch_logits), torch.cat(batch_targets)


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    full_accuracy: float


@torch.inference_mode()
def evaluate(
        model: model.GPT, 
        valset: dict[Literal["x_tokens", "x_digit_tokens", "y_tokens", "y_indices"], list],
        args: argparse.Namespace,
        gen: data.GenerateEquations | None = None,
        print_sample: bool = False,
) -> EvalResult:
    model.eval()
    loss = 0.0
    accuracy = 0.0
    full_accuracy = 0.0
    for batch_idx, (x_tokens, x_digit_tokens, y_tokens, y_indices) in enumerate(iterate_dataset(valset, args)):
        x_tokens = x_tokens.to(args.device)
        x_digit_tokens = x_digit_tokens.to(args.device) if args.use_digits else None
        y_tokens = y_tokens.to(args.device)
        y_indices = y_indices.to(args.device)
        logits = model(x_tokens, x_digit_tokens)

        token_logits, token_targets = slice_logits_and_targets(logits, y_indices, y_tokens)
        loss += F.cross_entropy(token_logits, token_targets).item()
        accuracy += (token_logits.argmax(dim=-1) == token_targets).float().mean().item()
        del token_logits, token_targets
        # full accuracy: all target tokens are correct
        full_correct = 0
        for i, (start, end) in enumerate(y_indices):
            pred_tokens = logits[i, start:end].argmax(dim=-1)
            target_tokens = y_tokens[i, start:end]
            # Only count as correct if ALL tokens match
            is_correct = 1 if torch.all(pred_tokens == target_tokens) else 0
            full_correct += is_correct
            if print_sample and i == batch_idx == 0:
                # to see if the full-accuracy calc is correct
                print(f"{pred_tokens=} {target_tokens=}, {is_correct=}")

        full_accuracy += full_correct / len(y_indices)

    loss /= args.num_steps_val
    accuracy /= args.num_steps_val
    full_accuracy /= args.num_steps_val
    model.train()
    return EvalResult(loss=loss, accuracy=accuracy, full_accuracy=full_accuracy)


def train(
        net: model.GPT, 
        trainset: pl.DataFrame, 
        valset: pl.DataFrame, 
        args: argparse.Namespace,
        gen: data.GenerateEquations | None = None,
):
    net = torch.compile(net)
    net = net.to(args.device)
    net.train()

    # Optimizer
    adamw_params = net.lm_head.parameters()
    adamw_params.extend(net.transformer.wte.parameters())
    if net.dte is not None:
        adamw_params.extend(net.dte.parameters())
        adamw_params.extend(net.digit_attn.parameters())
        adamw_params.extend(net.cross_attn.parameters())
    muon_params = [p for p in net.parameters() if p not in adamw_params]
    optimizer = Muon(
        muon_params=muon_params,
        lr=args.learning_rate,
        adamw_params=adamw_params,
        adamw_wd=args.weight_decay,
    )

    # Scheduler
    def get_lr(it):
        assert it <= args.num_steps
        # 1) linear warmup for warmup_steps steps
        if it < args.warmup_steps:
            return (it+1) / args.warmup_steps
        # 2) constant lr for a while
        elif it < args.num_iterations - args.cooldow_steps:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.cooldown_steps
            return decay_ratio
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_full_accuracies = []
    for step in range(args.num_steps):
        # Forward pass
        x_tokens, x_digit_tokens, y_tokens, y_indices = next(iterate_dataset(trainset, args))
        x_tokens = x_tokens.to(args.device)
        x_digit_tokens = x_digit_tokens.to(args.device) if args.use_digits else None
        y_tokens = y_tokens.to(args.device)
        y_indices = y_indices.to(args.device)
        logits = net(x_tokens, x_digit_tokens)
        logits, targets = slice_logits_and_targets(logits, y_indices, y_tokens)
        loss = F.cross_entropy(logits, targets)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if args.use_wandb:
            wandb.log({"train/loss": loss.item(), "train/step": step})
        train_losses.append(loss.item())

        if step % args.eval_every == 0:
            val_result = evaluate(net, valset, args, gen=gen, print_sample=step % args.print_every == 0)
            val_losses.append(val_result.loss)
            val_accuracies.append(val_result.accuracy)
            val_full_accuracies.append(val_result.full_accuracy)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_result.loss:.4f} val_accuracy={val_result.accuracy:.4f}")
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_result.loss, 
                    "val/accuracy": val_result.accuracy,
                    "val/full_accuracy": val_result.full_accuracy,
                })

    return train_losses, val_losses, val_accuracies, val_full_accuracies


def make_run_name(
        max_digits_per_token: int,
        max_tokens_per_num: int,
        op: Literal["+", "-", "*", "/"],
        mod: int | None,
        seed: int,
        num_params: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        T: int,
        length_factor: int,
        sliding_window_size: int | None,
        batchsize: int,
        num_steps: int,
) -> str:
    name = f"{num_params=}_{vocab_size=}_{n_layer=}_{n_head=}_{n_embd=}"
    name += f"_{max_digits_per_token=}_{max_tokens_per_num=}_{op=}_{mod=}"
    name += f"_{seed=}_{batchsize=}_{num_steps=}"
    name += f"_{length_factor=}_{sliding_window_size=}"
    name += f"_{T=}"
    return name


def save(results: dict[str, list], run_name: str):
    df = pl.DataFrame(results)
    if not Path(f"results/{run_name}.csv").exists():
        df.write_csv(f"results/{run_name}.csv", mode="a", index=False)
    else:
        with open(f"results/{run_name}.csv", "ab") as f:
            df.write_csv(f, include_header=False)


def train_and_save(
        args: argparse.Namespace,
        config: model.GPTConfig,
        gen: data.GenerateEquations,
        trainset: pl.DataFrame,
        valset: pl.DataFrame,
        max_digits_per_token: int,
        max_tokens_per_num: int,
        op: Literal["+", "-", "*", "/"],
        mod: int | None,
        seed: int,
):
    net = model.GPT(config)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    run_name = make_run_name(
        max_digits_per_token=max_digits_per_token,
        max_tokens_per_num=max_tokens_per_num,
        op=op,
        mod=mod,
        seed=seed,
        num_params=num_params,
        vocab_size=gen.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        T=gen.max_possible_num_tokens,
        length_factor=max_digits_per_token,
        sliding_window_size=args.sliding_window_size,
        batchsize=args.batchsize,
        num_steps=args.num_steps,
    )
    if args.use_wandb:
        wandb.finish()
        wandb.init(name=run_name, project="mathblations", config=vars(args))
    train_losses, val_losses, val_accuracies, val_full_accuracies = train(
        net, trainset, valset, args, gen=gen
    )

    save(
        results=dict(
            max_digits_per_token=[max_digits_per_token],
            max_tokens_per_num=[max_tokens_per_num],
            op=[op],
            mod=[mod],
            final_train_loss=[train_losses[-1]],
            final_val_loss=[val_losses[-1]],
            final_val_accuracy=[val_accuracies[-1]],
            seed=[seed],
            num_params=[num_params],
            num_steps=[args.num_steps],
            batchsize=[args.batchsize],
            num_val_steps=[args.num_steps_val],
            config=[config],
            train_losses=[str(train_losses)],
            val_losses=[str(val_losses)],
            val_accuracies=[str(val_accuracies)],
            val_full_accuracies=[str(val_full_accuracies)],
        ),
        run_name=run_name,
    )


def main():
    args = get_args()

    total = len(args.max_digits_per_token) * len(args.max_tokens_per_num) * len(args.op) * len(args.mod)
    loop = tqdm(
        itertools.product(
            args.max_digits_per_token, args.max_tokens_per_num, args.op, args.mod
        ), 
        total=total,
    )

    for max_digits_per_token, max_tokens_per_num, op, mod in loop:
        loop.set_description(f"{max_digits_per_token=}, {max_tokens_per_num=}, {op=}, {mod=}")
        gen = data.GenerateEquations(
            max_digits_per_token=max_digits_per_token,
            max_tokens_per_num=max_tokens_per_num,
            op=op,
            mod=mod,
        )
        
        common_config = dict(
            vocab_size=gen.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            T=gen.max_possible_num_tokens,
            length_factor=max_digits_per_token,
            sliding_window_size=args.sliding_window_size,
        )
        config_with_digits = model.GPTConfig(use_digits=True, **common_config)
        config_no_digits = model.GPTConfig(use_digits=False, **common_config)

        seed = args.seed
        for _ in range(args.num_runs):
            torch.manual_seed(seed)
            random.seed(seed)
            seed += 1

            trainset, valset = make_dataset(gen, args)
            
            train_and_save(
                args=args,
                config=config_with_digits,
                gen=gen,
                trainset=trainset,
                valset=valset,
                max_digits_per_token=max_digits_per_token,
                max_tokens_per_num=max_tokens_per_num,
                op=op,
                mod=mod,
                seed=seed,
            )
            train_and_save(
                args=args,
                config=config_no_digits,
                gen=gen,
                trainset=trainset,
                valset=valset,
                max_digits_per_token=max_digits_per_token,
                max_tokens_per_num=max_tokens_per_num,
                op=op,
                mod=mod,
                seed=seed,
            )


if __name__ == "__main__":
    main()
