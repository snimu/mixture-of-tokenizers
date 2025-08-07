
import ast
from typing import Literal

import matplotlib.pyplot as plt
import math
import numpy as np


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


def get_window_size_blocks(
        step: int, *,
        max_window_size: int = 3456,
        step_size: int = 128,
        schedule: Literal["cubic", "sqrt"] = "cubic",
):
    x = step / 5960 # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    if schedule == "cubic":
        factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x # cubic schedule by @jadenj3o
    elif schedule == "sqrt":
        factor = math.sqrt(x * (2 - x))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return next_multiple_of_n(max_window_size * factor, n=step_size)


def get_lr(step: int, *, num_iterations: int = 5960, cooldown_frac: float = 0.7):
    x = step / num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac

def plot_hparams(
        schedule: Literal["cubic", "sqrt"] | list[Literal["cubic", "sqrt"]] = "cubic",
        plot_lr: bool = True
):
    x = list(range(5960))
    max_window_size: int = 3456
    schedule = [schedule] if isinstance(schedule, str) else schedule
    for s in schedule:
        assert s in ["cubic", "sqrt"]
        ws = [get_window_size_blocks(i, schedule=s) / max_window_size for i in x]
        plt.plot(x, ws, label=f"train seq len with {s} schedule")
    if plot_lr:
        lr = [get_lr(i) for i in x]
        plt.plot(x, lr, label="learning rate")
    plt.xlabel("step")
    plt.ylabel(r"% of maximum")
    plt.legend()
    plt.grid()
    plt.show()


def plot_results(
        header_numbers: list[int | str] | dict[int | str, str],
        filename: str,
        x_axis: str = "step",
):
    with open(filename, "r") as f:
        lines = f.readlines()

    if isinstance(header_numbers, dict):
        descriptions = list(header_numbers.values())
        header_numbers = list(header_numbers.keys())
    else:
        descriptions = ["" for _ in header_numbers]

    parsed = {hnum: {"step": [], "time": [], "loss": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:") and "val_loss" in line:
                parsed[hnum]["loss"].append(float(line.split()[1].split("val_loss:")[-1]))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
                parsed[hnum]["time"].append(float(line.split("train_time:")[1].split("ms")[0]) / 1000)
    
    for i, hnum in enumerate(header_numbers):
        description = f": {descriptions[i]}" if descriptions[i] else ""
        plt.plot(parsed[hnum][x_axis], parsed[hnum]["loss"], label=f"{hnum}{description}")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.ylabel("val_loss")
    plt.legend()
    plt.grid()
    plt.show()


def plot_byte_stats(header_numbers: list[int | str], filename: str, x_axis: str = "step"):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    parsed = {hnum: {"step": [], "bytes_total": [], "bytes_pulled": [], "bytes_blocked": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                parsed[hnum]["bytes_total"].append(int(line.split("total_bytes:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["bytes_pulled"].append(int(line.split("total_pulled:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["bytes_blocked"].append(int(line.split("total_blocked:")[1].split(" ")[0].replace("_", "")))
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))

    for hnum in header_numbers:
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_total"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: total")
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_pulled"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: pulled")
        plt.plot(parsed[hnum][x_axis], np.array(parsed[hnum]["bytes_blocked"]) / max(parsed[hnum]["bytes_total"]), label=f"{hnum}: blocked")
    plt.xlabel("step" if x_axis == "step" else "time (s)")
    plt.legend()
    plt.title("Byte stats: total, pulled, blocked")
    plt.grid()
    plt.show()


def plot_norm_lambdas(header_numbers: list[int | str], filename: str, norm: bool = False):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    parsed = {hnum: {"step": [], "token_lambda": [], "byte_lambda": []} for hnum in header_numbers}
    for hnum in header_numbers:
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                extract = True
                continue
            if extract and line.startswith("##"):
                break
            if extract and line.startswith("step:"):
                parsed[hnum]["step"].append(int(line.split("step:")[1].split("/")[0]))
            if extract and line.startswith("token_lambda"):
                parsed[hnum]["token_lambda"].append(float(line.split("token_lambda=")[1].split(",")[0].strip()))
                parsed[hnum]["byte_lambda"].append(float(line.split("byte_lambda=")[1].strip()))

    for hnum in header_numbers:
        if norm:
            norm_val = np.array(parsed[hnum]["token_lambda"]) + np.array(parsed[hnum]["byte_lambda"])
        else:
            norm_val = np.ones_like(np.array(parsed[hnum]["token_lambda"]))
        plt.plot(parsed[hnum]["step"], np.array(parsed[hnum]["token_lambda"]) / norm_val, label=f"{hnum}: token-lambda")
        plt.plot(parsed[hnum]["step"], np.array(parsed[hnum]["byte_lambda"]) / norm_val, label=f"{hnum}: byte-lambda")
    plt.xlabel("step")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()


def plot_skip_weights(header_numbers: list[int | str], filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()

    parsed = {hnum: {"skip_weights": ""} for hnum in header_numbers}
    for hnum in header_numbers:
        correct_hnum = False
        extract= False
        for line in lines:
            if line.strip() == f"## {hnum}":
                correct_hnum = True
                continue
            if correct_hnum and line.startswith("skip_weights="):
                extract = True
            if correct_hnum and extract and line.startswith("lambdas="):
                extract=False
                correct_hnum = False
                break
            if extract:
                parsed[hnum]["skip_weights"] += line.split("skip_weights=")[-1].strip()

    for hnum in header_numbers:
        parsed[hnum]["skip_weights"] = np.array(ast.literal_eval(parsed[hnum]["skip_weights"]))
        plt.plot(parsed[hnum]["skip_weights"], label=f"{hnum}: skip-weights")
    plt.xlabel("layer")
    plt.ylabel("skip-weight")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_hparams(schedule=["cubic", "sqrt"], plot_lr=False)
    plot_results(
        {
            # 0: "Baseline",
            1: "Baseline",
            # 7: "MoT",
            # 71: "MoT-sum, lr_tok=0.3, lr_byte=0.1",
            # 72: "MoT, hparams",
            # 73: "MoT-sum, norm-then-sum",
            # 74: "MoT-sum, norm-then-sum with lambdas",
            # 75: "MoT, hparams, token_dim=896",
            # "01": "Baseline, lr-schedule",
            # 76: "MoT, hparams, token_dim=896, lr-schedule",
            # "02": "Baseline, shuffled data",
            # 77: "MoT, hparams, token_dim=896, shuffled data",
            # "03": "Baseline, seq-len schedule",
            # 78: "MoT, hparams, token_dim=896, seq-len schedule",
            # 711: "MoT-concat",
            # 712: "MoT-concat, Add 758 to MLP hidden dim",
            # 713: "MoT-concat, switched Attention & MLP",
            71011: "MoT-sum, lr_tok=0.3, lr_byte=0.3",
            # 71012: "MoT-sum, lr_tok=0.3, lr_byte=0.4",
            # 71013: "MoT-sum, lr_tok=0.3, lr_byte=0.2",
            # 71021: "MoT-sum, lr_tok=0.4, lr_byte=0.3",
            # 71022: "MoT-sum, lr_tok=0.2, lr_byte=0.3",
            # 71031: "MoT-sum, lr_tok=0.4, lr_byte=0.4",
            # 71032: "MoT-sum, lr_tok=0.35, lr_byte=0.4",
            # 71033: "MoT-sum, lr_tok=0.4, lr_byte=0.45",
            # 71034: "MoT-sum, lr_tok=0.35, lr_byte=0.45",
            # 71035: "MoT-sum, lr_tok=0.4, lr_byte=0.5",
            # 71036: "MoT-sum, lr_tok=0.35, lr_byte=0.5",
            71041: "MoT-sum, lr_tok=0.3, lr_byte=0.3, lambdas",
            71042: "MoT-sum, lr_tok=0.3, lr_byte=0.3, normed lambdas",
            71043: "MoT-sum, lr_tok=0.3, lr_byte=0.3, normed lambdas, (0.99, 0.01)",
            71044: "MoT-sum, lr_tok=0.3, lr_byte=0.3, normed lambdas, (0.6, 0.4)",
        },
        filename="results.md",
        x_axis="step",
    )
    # plot_norm_lambdas([71041, 71042, 71043, 71044], "results.md", norm=False)
    # plot_byte_stats([79], "results.md", x_axis="step")
    # plot_skip_weights([71041], "results.md")