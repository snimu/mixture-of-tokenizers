
import ast
from typing import Literal

import seaborn as sns
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from tabulate import tabulate


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
        mod: int | None = None,
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
    filters &= (pl.col("mod") == mod) if mod is not None else (pl.col("mod").is_null())

    df = pl.scan_csv(file).filter(filters).collect()
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]
    
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    avg_ys = np.mean(ys, axis=0)
    xs = np.arange(len(ys[0]))
    if "val" in to_plot:
        xs = xs * 100  # TODO: save the actual steps in future runs and use those
    return xs, ys, avg_ys


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def plot_digits_vs_tokens(
        file: str,
        max_digits_per_token: int | list[int] | None = None,
        max_tokens_per_num: int | list[int] | None = None,
        mod: int | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses"] = "val_accuracies",
        show: bool = True,
        plot_all: bool = False,
):
    if mod is None:
        filter_ = pl.col("mod").is_null()
    else:
        filter_ = pl.col("mod") == mod
    settings =(
        pl.scan_csv(file)
        .filter(filter_)
        .select("max_digits_per_token", "max_tokens_per_num")
        .collect()
        .unique()
    )
    settings = [
        (dpt, tpn)
        for dpt, tpn in zip(
            settings["max_digits_per_token"],
            settings["max_tokens_per_num"],
        )
    ]
    if max_digits_per_token is not None:
        max_digits_per_token = [max_digits_per_token] if isinstance(
            max_digits_per_token, int
        ) else max_digits_per_token
        settings = [
            (dpt, tpn)
            for dpt, tpn in settings
            if dpt in max_digits_per_token
        ]
    if max_tokens_per_num is not None:
        max_tokens_per_num = [max_tokens_per_num] if isinstance(
            max_tokens_per_num, int
        ) else max_tokens_per_num
        settings = [
            (dpt, tpn)
            for dpt, tpn in settings
            if tpn in max_tokens_per_num
        ]
    settings = list(set(settings))
    settings = sorted(settings, key=lambda x: x[0])
    settings = sorted(settings, key=lambda x: x[1])

    colors = generate_distinct_colors(len(settings) * 2)
    for dpt, tpn in settings:
        color_digits = colors.pop(0)
        xs_d, ys_d, avg_ys_d = load_xs_ys_avg_y(
            file=file,
            use_digits=True,
            max_digits_per_token=dpt,
            max_tokens_per_num=tpn,
            mod=mod,
        )

        color_tokens = colors.pop(0)
        xs_t, ys_t, avg_ys_t = load_xs_ys_avg_y(
            file=file,
            use_digits=False,
            max_digits_per_token=dpt,
            max_tokens_per_num=tpn,
            mod=mod,
        )
        if plot_all:
            for y in ys_d:
                plt.plot(xs_d, y, color=color_digits, alpha=0.2)
            for y in ys_t:
                plt.plot(xs_t, y, color=color_tokens, alpha=0.2)
        plt.plot(xs_d, avg_ys_d, label=f"dpt={dpt}, tpn={tpn}; MoT ({len(ys_d)} samples)", color=color_digits, linestyle="--")
        plt.plot(xs_t, avg_ys_t, label=f"dpt={dpt}, tpn={tpn}; Baseline ({len(ys_t)} samples)", color=color_tokens)

    to_plot_to_label = {
        "val_losses": "loss (validation)",
        "val_accuracies": "token accuracy (validation)",
        "val_full_accuracies": "full-number accuracy (validation)",
        "train_losses": "loss (training)",
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


def heatmap_final_measure(
      file: str,
      avg_last_n: int = 5,
      to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses"] = "val_accuracies",
      show: bool = True,
):
    settings = pl.scan_csv(file).select(
        "max_digits_per_token", "max_tokens_per_num"
    ).collect().unique()
    settings = [(dpt, tpn) for dpt, tpn in zip(settings["max_digits_per_token"], settings["max_tokens_per_num"])]

    dpts = sorted(set(dpt for dpt, _ in settings))
    tpns = sorted(set(tpn for _, tpn in settings))
    heatmap = np.zeros((len(dpts), len(tpns)))

    for dpt, tpn in settings:
        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot=to_plot, use_digits=True,
        )
        y_digits = np.mean(avg_ys[-avg_last_n:])

        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot=to_plot, use_digits=False,
        )
        y_tokens = np.mean(avg_ys[-avg_last_n:])

        i = dpts.index(dpt)
        j = tpns.index(tpn)
        ratio = y_digits / (y_tokens + 1e-6)
        heatmap[i, j] = ratio

    if show:
        plt.figure(figsize=(6,5))
        sns.heatmap(
            heatmap, annot=True, fmt='.3f', cmap='viridis', 
            vmin=0.95, vmax=1.01, center=1.0,
            xticklabels=tpns, yticklabels=dpts,
            annot_kws={'size': 8}, cbar_kws={'label': to_plot},
        )
        plt.xlabel('Tokens per number')
        plt.ylabel('Digits per token')
        plt.tight_layout()
        plt.show()
    close_plt()

    return heatmap, dpts, tpns


def get_other_metrics(file: str, mod: int | None = None):
    if mod is None:
        filter_ = pl.col("mod").is_null()
    else:
        filter_ = pl.col("mod") == mod
    df =(
        pl.scan_csv(file)
        .filter(filter_)
        .sort(pl.col("max_digits_per_token"), pl.col("max_tokens_per_num"))
        .collect()
    )
    settings = df.select("max_digits_per_token", "max_tokens_per_num").unique()
    settings = [
        (dpt, tpn)
        for dpt, tpn in zip(
            settings["max_digits_per_token"],
            settings["max_tokens_per_num"],
        )
    ]

    results = {
        "dpt": [],
        "tpn": [],
        "num_equations_seen": [],
        "num_tokens_seen": [],
        "num_unique_tokens": [],
        "times_token_is_seen": [],
        "num_possible_equations": [],
        "times_eq_seen_in_training": [],
        "final_val_accuracy_digits": [],
        "final_val_accuracy_tokens": [],
        "final_val_accuracy_full_digits": [],
        "final_val_accuracy_full_tokens": [],
    }
    for dpt, tpn in settings:
        df_loc = (
            df
            .filter(pl.col("max_digits_per_token") == dpt)
            .filter(pl.col("max_tokens_per_num") == tpn)
        )
        num_equations_seen = df_loc["batchsize"].unique()[0] * df_loc["num_steps"].unique()[0]
        num_unique_tokens = int("9" * dpt) + 1
        num_tokens_seen = num_equations_seen * tpn * 2  # underestimate because the model also sees some result tokens
        times_token_is_seen = num_tokens_seen / num_unique_tokens
        num_possible_equations = (num_unique_tokens * tpn) ** 2
        times_eq_seen_in_training = num_equations_seen / num_possible_equations
        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_accuracies", use_digits=True,
        )
        final_val_accuracy_digits = avg_ys[-1]
        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_accuracies", use_digits=False,
        )
        final_val_accuracy_tokens = avg_ys[-1]
        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_full_accuracies", use_digits=True,
        )
        final_val_accuracy_full_digits = avg_ys[-1]
        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_full_accuracies", use_digits=False,
        )
        final_val_accuracy_full_tokens = avg_ys[-1]
        results["dpt"].append(dpt)
        results["tpn"].append(tpn)
        results["num_equations_seen"].append(num_equations_seen)
        results["num_tokens_seen"].append(num_tokens_seen)
        results["num_unique_tokens"].append(num_unique_tokens)
        results["times_token_is_seen"].append(round(times_token_is_seen))
        results["num_possible_equations"].append(num_possible_equations)
        results["times_eq_seen_in_training"].append(times_eq_seen_in_training)
        results["final_val_accuracy_digits"].append(final_val_accuracy_digits)
        results["final_val_accuracy_tokens"].append(final_val_accuracy_tokens)
        results["final_val_accuracy_full_digits"].append(final_val_accuracy_full_digits)
        results["final_val_accuracy_full_tokens"].append(final_val_accuracy_full_tokens)
    return results


def print_other_metrics(file: str, mod: int | None = None):
    results = get_other_metrics(file=file, mod=mod)
    name_map = {
        "dpt": "dpt",
        "tpn": "tpn",
        "times_token_is_seen": "times tok seen",
        "times_eq_seen_in_training": "times eq. seen",
        "final_val_accuracy_digits": "val acc (digits)",
        "final_val_accuracy_tokens": "val acc (tokens)",
        "final_val_accuracy_full_digits": "val acc full (digits)",
        "final_val_accuracy_full_tokens": "val acc full (tokens)",
    }
    results = {name_map[k]: v for k, v in results.items() if k in name_map}
    print(tabulate(results, headers="keys", intfmt="_", floatfmt="_.3f"))


def scatter_acc_over_times_tok_seen(file: str, mod: int | None = None):
    results = get_other_metrics(file=file, mod=mod)
    plt.scatter(results["times_token_is_seen"], results["final_val_accuracy_digits"], label="MoT")
    plt.scatter(results["times_token_is_seen"], results["final_val_accuracy_tokens"], label="Baseline")
    plt.xlabel("average time a token is seen")
    plt.ylabel("final val accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    close_plt()


def scatter_acc_over_times_eq_seen(file: str, mod: int | None = None):
    results = get_other_metrics(file=file, mod=mod)
    plt.scatter(results["times_eq_seen_in_training"], results["final_val_accuracy_digits"], label="MoT")
    plt.scatter(results["times_eq_seen_in_training"], results["final_val_accuracy_tokens"], label="Baseline")
    plt.xlabel("average time an equation is seen")
    plt.ylabel("final val accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    close_plt()


def scatter_full_acc_over_times_tok_seen(file: str, mod: int | None = None):
    results = get_other_metrics(file=file, mod=mod)
    plt.scatter(results["times_token_is_seen"], results["final_val_accuracy_full_digits"], label="MoT")
    plt.scatter(results["times_token_is_seen"], results["final_val_accuracy_full_tokens"], label="Baseline")
    plt.xlabel("average time a token is seen")
    plt.ylabel("final val accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    close_plt()


def scatter_full_acc_over_times_eq_seen(file: str, mod: int | None = None):
    results = get_other_metrics(file=file, mod=mod)
    plt.scatter(results["times_eq_seen_in_training"], results["final_val_accuracy_full_digits"], label="MoT")
    plt.scatter(results["times_eq_seen_in_training"], results["final_val_accuracy_full_tokens"], label="Baseline")
    plt.xlabel("average time an equation is seen")
    plt.ylabel("final val accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    close_plt()


if __name__ == "__main__":
    file = "results1.csv"
    # plot_digits_vs_tokens(
    #     file=file,
    #     max_digits_per_token=None,
    #     max_tokens_per_num=None,
    #     to_plot="val_full_accuracies",
    #     plot_all=False,
    #     mod=None,
    # )
    # heatmap_final_measure(
    #     file=file,
    #     to_plot="val_full_accuracies",
    #     avg_last_n=1,
    # )
    # print_other_metrics(file=file, mod=None)
    scatter_acc_over_times_tok_seen(file=file, mod=None)
