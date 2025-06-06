# /// script
# requires-python = "==3.12"
# dependencies = [
#   "numpy",
#   "torch",
#   "wandb",
#   "polars",
#   "tqdm",
#   "einops",
# ]
# ///

import os
import argparse
import itertools
import random
from typing import Any, Literal
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import polars as pl

from data import GenerateEquations, make_dataset, iterate_dataset, slice_logits_and_targets
import model
from muon import Muon


def _to_list(item: Any, dtype: type | None) -> list:
    if dtype is None and item is None:
        return [None]
    if isinstance(item, dtype):
        return [item]
    if isinstance(item, list):
        return item
    raise ValueError(f"Expected {dtype} or list of {dtype}, got {item}")


def get_args():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--use-wandb", action="store_true", help="flag")
    parser.add_argument("--print-every", type=int, default=100, help="type=int, default=100")
    parser.add_argument("--savefile", type=str, default="results", help="type=str, default=results")

    # Data parameters
    parser.add_argument("--max-digits-per-token", type=int, default=3, nargs="+", help="type=int, default=3, nargs=+")
    parser.add_argument("--max-tokens-per-num", type=int, default=1, nargs="+", help="type=int, default=1, nargs=+")
    parser.add_argument("--op", type=str, choices=("+", "-", "*", "/"), default="+", nargs="+", help="type=str, choices=('+', '-', '*', '/'), default='+', nargs=+")
    parser.add_argument("--mod", type=int, default=None, help="type=int, default=None, nargs=+")

    # Training parameters
    parser.add_argument("--batchsize", type=int, default=1024, help="type=int, default=1024")
    parser.add_argument("--num-steps", type=int, default=50_000, help="type=int, default=50_000")
    parser.add_argument("--num-steps-val", type=int, default=5, help="type=int, default=5")
    parser.add_argument("--num-epochs", type=int, default=1, help="type=int, default=1")
    parser.add_argument("--device", type=str, default="cuda", help="type=str, default='cuda'")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="type=float, default=0.001")
    parser.add_argument("--warmup-steps", type=int, default=0, help="type=int, default=0")
    parser.add_argument("--cooldown-steps", type=int, default=5_000, help="type=int, default=5_000")
    parser.add_argument("--weight-decay", type=float, default=0, help="type=float, default=0")
    parser.add_argument("--clip-min", type=int, default=0, help="type=int, default=0")
    parser.add_argument("--clip-max", type=int, default=15, help="type=int, default=1")
    parser.add_argument("--eval-every", type=int, default=100, help="type=int, default=100")
    
    # Ablation parameters
    parser.add_argument("--num-runs", type=int, default=1, help="type=int, default=1")
    parser.add_argument("--seed", type=int, default=385, help="type=int, default=385")
    parser.add_argument("--regenerate-dataset-every-run", action="store_true", help="type=FLAG")

    # Model parameters
    parser.add_argument("--n-layer", type=int, default=12, help="type=int, default=12")
    parser.add_argument("--n-head", type=int, default=6, help="type=int, default=6")
    parser.add_argument("--n-embd-tok", type=int, default=768, help="type=int, default=768")
    parser.add_argument("--n-embd-digit", type=int, default=768, help="type=int, default=768")
    parser.add_argument("--n-layer-output", type=int, default=1, help="type=int, default=0")
    parser.add_argument("--digit-mixout-method", choices=("self_attn", "cross_attn", "noop"), default="noop", help="default='noop'")
    parser.add_argument("--digit-mixin-method", choices=("cross_attn", "concat", "noop"), default="noop", help="default='noop'")
    parser.add_argument("--use-digit-self-attn", action="store_true", help="type=FLAG")

    
    args = parser.parse_args()

    args.max_digits_per_token = _to_list(args.max_digits_per_token, int)
    args.max_tokens_per_num = _to_list(args.max_tokens_per_num, int)
    args.op = _to_list(args.op, str)
    args.mod = _to_list(args.mod, None if args.mod is None else int)

    assert args.print_every % args.eval_every == 0, \
        f"print_every ({args.print_every}) must be a multiple of eval_every ({args.eval_every})"
    
    return args


def get_l1_grad_norm(net: model.GPT) -> float:
    norm = 0.0
    for p in net.parameters():
        if p.grad is not None:
            norm += p.grad.abs().sum().item()
    return norm / sum(p.numel() for p in net.parameters())


def print_sample(
        x_tokens: torch.Tensor, 
        y_tokens: torch.Tensor,
        x_digit_tokens: torch.Tensor | None,
        y_digit_tokens: torch.Tensor | None,
        generated_tokens: torch.Tensor, 
        gen: GenerateEquations,
        loop: tqdm = None,
) -> None:
    xt = x_digit_tokens if x_digit_tokens is not None else x_tokens
    yt = y_digit_tokens if y_digit_tokens is not None else y_tokens
    rand_idx = random.randint(0, len(xt) - 1)
    x = xt[rand_idx].cpu().squeeze().tolist()
    y = yt[rand_idx].cpu().squeeze().tolist()
    if x_digit_tokens is not None:
        target_equation = gen.eq_digits_to_str(torch.tensor(x + y[-gen.max_digits_per_token:]))
    else:
        target_equation = gen.eq_to_str(torch.tensor(x + [y[-1]]))
    generated_token = generated_tokens[rand_idx].cpu().squeeze().tolist()

    if loop:
        loop.write(f"{target_equation=}\n{generated_token=}\n")
    else:
        print(f"{target_equation=}\n{generated_token=}\n")


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    full_accuracy: float
    l1: float
    l2: float


@torch.inference_mode()
def evaluate(
        model: model.GPT, 
        valset: dict[
            Literal[
                "x_tokens", "x_digit_tokens",
                "y_tokens", "y_digit_tokens",
                "y_indices", "y_digit_indices"
            ],
            list
        ],
        args: argparse.Namespace,
        config: model.GPTConfig,
) -> EvalResult:
    model.eval()
    loss = 0.0
    accuracy = 0.0
    full_accuracy = 0.0
    val_iterator = iterate_dataset(valset, args, config)
    l1, l2 = 0.0, 0.0
    for batch_idx, (x_tokens, x_digit_tokens, y_tokens, y_digit_tokens, y_indices, y_digit_indices) in enumerate(val_iterator):
        logits = model(x_tokens, x_digit_tokens)

        token_logits, token_targets = slice_logits_and_targets(
            logits,
            y_indices, y_tokens,
            y_digit_indices if config.n_layer_output > 0 else None,
            y_digit_tokens if config.n_layer_output > 0 else None,
        )
        loss += F.cross_entropy(token_logits, token_targets).item()
        accuracy += (token_logits.argmax(dim=-1) == token_targets).float().mean().item()
        del token_logits, token_targets
        # full accuracy: all target tokens are correct
        full_correct = 0
        # l1 and l2 distance from target number: just get a target & predicted vector and then compare
        targets = []
        predictions = []
        indices = y_digit_indices if y_digit_indices is not None and config.n_layer_output > 0 else y_indices
        digit_tokens = y_digit_tokens if y_digit_tokens is not None and config.n_layer_output > 0 else y_tokens
        remove_padding = y_digit_tokens is not None and config.n_layer_output > 0
        for i, (start, end) in enumerate(indices):
            pred_tokens = logits[i, start:end].argmax(dim=-1)
            target_tokens = digit_tokens[i, start:end]
            # Only count as correct if ALL tokens match
            full_correct += int(torch.all(pred_tokens == target_tokens))

            try:
                target_num = int("".join([
                    str(t.item())
                    for t in target_tokens
                    if (not remove_padding) or (t.item() < 11)
                ]))
            except ValueError:
                target_num = 0
            try:
                pred_num = int("".join([
                    str(t.item())
                    for t in pred_tokens
                    if (not remove_padding) or (t.item() < 11)
                ]))
            except ValueError:
                pred_num = 0
            targets.append(target_num)
            predictions.append(pred_num)

        full_accuracy += full_correct / len(y_indices)
        try:
            ttargets = torch.tensor(targets, dtype=torch.long).float()
            tpredictions = torch.tensor(predictions, dtype=torch.long).float()
            l1 += F.l1_loss(ttargets, tpredictions).item()
            l2 += F.mse_loss(ttargets, tpredictions).item()
        except RuntimeError:
            print(
                f"\n\n{max(predictions)=:_}\n"
                f"{min(predictions)=:_}\n\n"
                f"{predictions=}\n\n"
            )
            l1 += float(torch.finfo(torch.float32).max)
            l2 += float(torch.finfo(torch.float32).max)
            full_accuracy += 0.0

    loss /= args.num_steps_val
    accuracy /= args.num_steps_val
    full_accuracy /= args.num_steps_val
    l1 /= args.num_steps_val
    l2 /= args.num_steps_val
    model.train()
    return EvalResult(loss=loss, accuracy=accuracy, full_accuracy=full_accuracy, l1=l1, l2=l2)


def train(
        net: model.GPT, 
        trainset: pl.DataFrame, 
        valset: pl.DataFrame,
        args: argparse.Namespace,
        config: model.GPTConfig,
        gen: GenerateEquations | None = None,
        loop: tqdm = None,
):
    print_ = loop.write if loop else print
    net = torch.compile(net)
    net = net.to(args.device)
    net.train()

    # Optimizer
    adamw_params = (
        list(net.wte.parameters())
        + list(net.dte.parameters())
        + list(net.digit_mixin.parameters())
        + list(net.lm_head.parameters())
    )
    muon_params = list(net.h.parameters()) + list(net.digit_mixout.parameters())
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
        elif it < args.num_steps - args.cooldown_steps:
            return 1.0
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_steps - it) / args.cooldown_steps
            return decay_ratio
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    train_losses = []
    train_l1_grad_norms = []
    val_losses = []
    val_accuracies = []
    val_full_accuracies = []
    val_l1s = []
    val_l2s = []
    timings = []
    epoch = 0
    t0 = t0_global = perf_counter()
    for step in range(args.num_steps * args.num_epochs):
        if step % args.num_steps == 0:
            epoch += 1
            train_iterator = iterate_dataset(trainset, args, config)
        # Forward pass
        x_tokens, x_digit_tokens, y_tokens, y_digit_tokens, y_indices, y_digit_indices = next(train_iterator)
        optimizer.zero_grad(set_to_none=True)
        logits = net(x_tokens, x_digit_tokens)
        if config.n_layer_output > 0:
            predictions, targets = slice_logits_and_targets(
                logits, y_indices, y_tokens, y_digit_indices, y_digit_tokens
            )
        else:
            predictions, targets = slice_logits_and_targets(
                logits, y_indices, y_tokens, None, None
            )
        loss = F.cross_entropy(predictions, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            grad_norm = get_l1_grad_norm(net)
            train_l1_grad_norms.append(grad_norm)

        # Logging
        timings.append(perf_counter() - t0)
        if args.use_wandb:
            wandb.log({
                "train/loss": loss.item(), 
                "train/step": step, 
                "train/epoch": epoch, 
                "train/l1_grad_norm": grad_norm,
                "train/timing": timings[-1],
            })
        train_losses.append(loss.item())

        if step % args.eval_every == 0 or step == args.num_steps - 1:
            val_result = evaluate(net, valset, args, config=config)
            val_losses.append(val_result.loss)
            val_accuracies.append(val_result.accuracy)
            val_full_accuracies.append(val_result.full_accuracy)
            val_l1s.append(val_result.l1)
            val_l2s.append(val_result.l2)
            print_(
                f"step={step} train_loss={loss.item():.4f} "
                f"val_l1={val_result.l1:.4f} "
                f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} "
                f"val_full_acc={val_result.full_accuracy:.4f} "
                f"t_step={int(timings[-1])}s t_total={int(perf_counter() - t0_global)}s"
            )
            if step % args.print_every == 0:
                print_sample(
                    x_tokens=x_tokens,
                    y_tokens=y_tokens,
                    x_digit_tokens=x_digit_tokens if config.n_layer_output > 0 else None,
                    y_digit_tokens=y_digit_tokens if config.n_layer_output > 0 else None,
                    generated_tokens=logits.argmax(-1),
                    gen=gen,
                    loop=loop,
                )
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_result.loss, 
                    "val/accuracy": val_result.accuracy,
                    "val/full_accuracy": val_result.full_accuracy,
                    "val/l1": val_result.l1,
                    "val/l2": val_result.l2,
                    "val/epoch": epoch,
                    "val/step": step,
                })
        t0 = perf_counter()

    return train_losses, val_losses, val_accuracies, val_full_accuracies, val_l1s, val_l2s, timings, perf_counter() - t0_global


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


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
        n_embd_tok: int,
        n_embd_digit: int,
        T: int,
        length_factor: int,
        batchsize: int,
        num_steps: int,
        num_epochs: int,
        n_layer_output: int,
        digit_mixout_method: Literal["sequential", "cross_attention"],
        digit_mixin_method: Literal["cross_attn", "concat"],
        use_digit_self_attn: bool,
        
) -> str:
    op_to_word = {"+": "addition", "-": "substraction", "*": "multiplication", "/": "division"}
    name = f"{format_num_params(num_params, 0)}"
    name += f"_dmi-{digit_mixin_method}"
    name += f"_dmo-{digit_mixout_method}"
    name += f"_nlo{n_layer_output}" if digit_mixout_method != "noop" else ""
    name += "_dsa" if use_digit_self_attn else ""
    name += f"_{max_digits_per_token}dpt_{max_tokens_per_num}tpn_{op_to_word[op]}_mod{mod}"
    name += f"_{vocab_size}vocab_{n_layer}layers_{n_head}heads_{n_embd_tok}Dtok_{n_embd_digit}Ddig"
    name += f"_{seed}seed_{batchsize}bs_{num_steps}steps_{num_epochs}epochs"
    name += f"_{length_factor}lf_{T}T"
    return name


def save(results: dict[str, list], savefile: str):
    df = pl.DataFrame(results)
    if not Path(f"results/{savefile}.csv").exists():
        os.makedirs("results", exist_ok=True)
        df.write_csv(f"results/{savefile}.csv")
    else:
        with open(f"results/{savefile}.csv", "ab") as f:
            df.write_csv(f, include_header=False)


def train_and_save(
        args: argparse.Namespace,
        config: model.GPTConfig,
        gen: GenerateEquations,
        trainset: pl.DataFrame,
        valset: pl.DataFrame,
        max_digits_per_token: int,
        max_tokens_per_num: int,
        op: Literal["+", "-", "*", "/"],
        mod: int | None,
        seed: int,
        loop: tqdm = None,
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
        n_embd_tok=args.n_embd_tok,
        n_embd_digit=args.n_embd_digit,
        T=gen.max_possible_num_tokens,
        length_factor=max_digits_per_token,
        batchsize=args.batchsize,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        n_layer_output=config.n_layer_output,
        digit_mixout_method=config.digit_mixout_method,
        digit_mixin_method=config.digit_mixin_method,
        use_digit_self_attn=config.use_digit_self_attn,
    )
    if args.use_wandb:
        wandb.finish(quiet=True)
        wandb.init(name=run_name, project="mathblations.new", config=vars(args))
    train_losses, val_losses, val_accuracies, val_full_accuracies, val_l1s, val_l2s, timings, total_time = train(
        net, trainset, valset, args, gen=gen, config=config, loop=loop,
    )

    save(
        results=dict(
            max_digits_per_token=[max_digits_per_token],
            max_tokens_per_num=[max_tokens_per_num],
            n_layer_output=[config.n_layer_output],
            digit_mixin_method=[config.digit_mixin_method],
            digit_mixout_method=[config.digit_mixout_method],
            use_digit_self_attn=[config.use_digit_self_attn],
            op=[op],
            mod=[mod],
            final_train_loss=[train_losses[-1]],
            final_val_loss=[val_losses[-1]],
            final_val_accuracy=[val_accuracies[-1]],
            seed=[seed],
            num_params=[num_params],
            depth=[config.n_layer],
            width=[config.n_embd_tok],
            width_digit=[config.n_embd_digit],
            heads=[config.n_head],
            vocab_size=[config.vocab_size],
            num_steps=[args.num_steps],
            batchsize=[args.batchsize],
            num_val_steps=[args.num_steps_val],
            train_losses=[str(train_losses)],
            val_losses=[str(val_losses)],
            val_accuracies=[str(val_accuracies)],
            val_full_accuracies=[str(val_full_accuracies)],
            val_l1s=[str(val_l1s)],
            val_l2s=[str(val_l2s)],
            timings=[str(timings)],
            total_time=[total_time],
        ),
        savefile=args.savefile,
    )


def main():
    args = get_args()
    os.environ["WANDB_SILENT"] = "true"
    total = len(args.max_digits_per_token) * len(args.max_tokens_per_num) * len(args.op) * len(args.mod)
    loop = tqdm(
        itertools.product(
            args.max_digits_per_token, args.max_tokens_per_num, args.op, args.mod
        ), 
        total=total,
    )
    torch.set_float32_matmul_precision('high')

    for max_digits_per_token, max_tokens_per_num, op, mod in loop:
        loop.set_description(f"{max_digits_per_token=}, {max_tokens_per_num=}, {op=}, {mod=}")
        gen = GenerateEquations(
            max_digits_per_token=max_digits_per_token,
            max_tokens_per_num=max_tokens_per_num,
            op=op,
            mod=mod,
        )

        config = model.GPTConfig(
            vocab_size=gen.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd_tok=args.n_embd_tok,
            n_embd_digit=args.n_embd_digit,
            T=gen.max_possible_num_tokens,
            length_factor=max_digits_per_token,
            n_layer_output=args.n_layer_output,
            digit_mixin_method=args.digit_mixin_method,
            digit_mixout_method=args.digit_mixout_method,
            use_digit_self_attn=args.use_digit_self_attn,
        )

        if not args.regenerate_dataset_every_run:
            trainset, valset = make_dataset(gen, args, loop=loop)

        seed = args.seed
        for _ in range(args.num_runs):
            torch.manual_seed(seed)
            random.seed(seed)
            seed += 1

            if args.regenerate_dataset_every_run:
                trainset, valset = make_dataset(gen, args, loop=loop)
            else:
                # Shuffle the trainset
                shuffle_indices = torch.randperm(len(trainset["x_tokens"]))
                trainset["x_tokens"] = [trainset["x_tokens"][i] for i in shuffle_indices]
                trainset["x_digit_tokens"] = [trainset["x_digit_tokens"][i] for i in shuffle_indices]
                trainset["y_tokens"] = [trainset["y_tokens"][i] for i in shuffle_indices]
                trainset["y_digit_tokens"] = [trainset["y_digit_tokens"][i] for i in shuffle_indices]
                trainset["y_indices"] = [trainset["y_indices"][i] for i in shuffle_indices]
                trainset["y_digit_indices"] = [trainset["y_digit_indices"][i] for i in shuffle_indices]

            loop.set_description(f"{max_digits_per_token=}, {max_tokens_per_num=}, {op=}, {mod=}, {seed=}")
            train_and_save(
                args=args,
                config=config,
                gen=gen,
                trainset=trainset,
                valset=valset,
                max_digits_per_token=max_digits_per_token,
                max_tokens_per_num=max_tokens_per_num,
                op=op,
                mod=mod,
                seed=seed,
                loop=loop,
            )


if __name__ == "__main__":
    main()
