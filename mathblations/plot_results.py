
import ast
import itertools
from typing import Literal

import polars as pl
import numpy as np
from rich import print
import matplotlib.pyplot as plt

def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series | str) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        try:
            return np.array(ast.literal_eval(series))
        except ValueError:
            series = series.replace("nan", "0")
            series = series.replace("inf", str(float(np.finfo(np.float16).max)))
            return np.array(ast.literal_eval(series))


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


def running_average(tensor: np.ndarray, window_size: int) -> np.ndarray:
    kernel = np.ones(window_size) / window_size
    output = np.convolve(tensor, kernel, mode='same')
    return output


def load_xs_ys_avg_y(
        file: str,
        use_digits: bool | None = None,
        max_digits_per_token: int | None = None,
        max_tokens_per_num: int | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        num_heads: int | None = None,
        seed: int | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses"] = "val_accuracies",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("final_val_loss").ge(0))  # initial condition -> always true

    if use_digits is not None:
        filters &= (pl.col("use_digits") == use_digits)
    if max_digits_per_token is not None:
        filters &= (pl.col("max_digits_per_token") == max_digits_per_token)
    if max_tokens_per_num is not None:
        filters &= (pl.col("max_tokens_per_num") == max_tokens_per_num)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if seed is not None:
        filters &= (pl.col("seed") == seed)

    df = pl.scan_csv(file).filter(filters).collect()
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]
    
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    avg_ys = np.mean(ys, axis=0)
    xs = np.arange(len(ys[0]))
    if "val" in to_plot:
        xs = xs * 100  # TODO: save the actual steps in future runs and use those
    return xs, ys, avg_ys


def plot_digits_vs_tokens(
        file: str,
        max_digits_per_token: int | list[int] | None = None,
        max_tokens_per_num: int | list[int] | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses"] = "val_accuracies",
        show: bool = True,
):
    assert (max_digits_per_token is None and max_tokens_per_num is None) or (max_digits_per_token is not None and max_tokens_per_num is not None)
    if max_digits_per_token is None and max_tokens_per_num is None:
        settings = pl.scan_csv(file).select(
            "max_digits_per_token", "max_tokens_per_num"
        ).collect().unique()
        settings = [(dpt, tpn) for dpt, tpn in zip(settings["max_digits_per_token"], settings["max_tokens_per_num"])]
    else:
        max_digits_per_token = [max_digits_per_token] if isinstance(max_digits_per_token, int) else max_digits_per_token
        max_tokens_per_num = [max_tokens_per_num] if isinstance(max_tokens_per_num, int) else max_tokens_per_num
        settings = list(itertools.product(max_digits_per_token, max_tokens_per_num))

    for dpt, tpn in settings:
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file=file,
            use_digits=True,
            max_digits_per_token=dpt,
            max_tokens_per_num=tpn,
        )
        plt.plot(xs, avg_ys, label=f"dpt={dpt}, tpn={tpn}; digits")
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file=file,
            use_digits=False,
            max_digits_per_token=dpt,
            max_tokens_per_num=tpn,
        )
        plt.plot(xs, avg_ys, label=f"dpt={dpt}, tpn={tpn}; tokens")

    to_plot_to_label = {
        "val_losses": "validation loss",
        "val_accuracies": "validation accuracy",
        "val_full_accuracies": "validation full accuracy",
        "train_losses": "training loss",
    }
    plt.legend()
    plt.xlabel("step")
    plt.ylabel(to_plot_to_label[to_plot])
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f"{to_plot}_vs_max_digits_per_token_max_tokens_per_num.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


if __name__ == "__main__":
    file = "results_deep.csv"
    plot_digits_vs_tokens(
        file=file,
        max_digits_per_token=2,
        max_tokens_per_num=3,
        to_plot="val_full_accuracies",
    )
