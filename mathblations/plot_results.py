
import ast
import os
from typing import Literal

import seaborn as sns
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from tabulate import tabulate
from tqdm import tqdm


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
        digit_mixin_method: Literal["cross_attn", "concat", "noop"] | None = None,
        digit_mixout_method: Literal["self_attn", "cross_attn", "noop"] | None = None,
        use_digit_self_attn: bool | None = None,
        max_digits_per_token: int | None = None,
        max_tokens_per_num: int | None = None,
        depth: int | None = None,
        width: int | None = None,
        width_digit: int | None = None,
        num_params: int | None = None,
        num_heads: int | None = None,
        seed: int | None = None,
        op: Literal["+", "*"] | None = None,
        mod: int | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("final_val_loss").ge(0))  # initial condition -> always true

    if digit_mixin_method is not None:
        filters &= (pl.col("digit_mixin_method") == digit_mixin_method)
    if digit_mixout_method is not None:
        filters &= (pl.col("digit_mixout_method") == digit_mixout_method)
    if use_digit_self_attn is not None:
        filters &= (pl.col("use_digit_self_attn") == use_digit_self_attn)
    if max_digits_per_token is not None:
        filters &= (pl.col("max_digits_per_token") == max_digits_per_token)
    if max_tokens_per_num is not None:
        filters &= (pl.col("max_tokens_per_num") == max_tokens_per_num)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if width_digit is not None:
        filters &= (pl.col("width_digit") == width_digit)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if seed is not None:
        filters &= (pl.col("seed") == seed)
    if op is not None:
        filters &= (pl.col("op") == op)
    filters &= (pl.col("mod") == mod) if mod is not None else (pl.col("mod").is_null())

    df = pl.scan_csv(file).filter(filters).collect()
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]
    
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    aggregate_map = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    avg_ys = aggregate_map[aggregate_method](ys, axis=0)
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


TO_PLOT_TO_LABEL = {
    "val_losses": "loss (val)",
    "val_accuracies": "token accuracy (val)",
    "val_full_accuracies": "full-number accuracy (val)",
    "train_losses": "loss (train)",
    "val_l1s": "L1 error (val)",
    "val_l2s": "L2 error (val)",
}


def plot_digits_vs_tokens(
        file: str,
        max_digits_per_token: int | list[int] | None = None,
        max_tokens_per_num: int | list[int] | None = None,
        op: Literal["+", "*"] = "+",
        mod: int | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        show: bool = True,
        plot_all: bool = False,
):
    if mod is None:
        filter_ = pl.col("mod").is_null()
    else:
        filter_ = pl.col("mod") == mod
    settings = (
        pl.scan_csv(file)
        .filter(filter_ & (pl.col("op") == op))
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
            op=op,
            mod=mod,
            to_plot=to_plot,
            aggregate_method=aggregate_method,
        )

        color_tokens = colors.pop(0)
        xs_t, ys_t, avg_ys_t = load_xs_ys_avg_y(
            file=file,
            use_digits=False,
            max_digits_per_token=dpt,
            max_tokens_per_num=tpn,
            op=op,
            mod=mod,
            to_plot=to_plot,
            aggregate_method=aggregate_method,
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
        "val_l1s": "L1 distance to ground truth (validation)",
        "val_l2s": "L2 distance to ground truth (validation)",
    }
    plt.legend()
    plt.xlabel("step")
    plt.ylabel(to_plot_to_label[to_plot])
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        def name(x):
            if x is None:
                return "all"
            if isinstance(x, list):
                return "-".join(str(y) for y in x)
            return str(x)
        dpts = name(max_digits_per_token)
        tpts = name(max_tokens_per_num)
        plt.savefig(
            f"plot_digits_vs_tokens__{to_plot}__{aggregate_method}__mod-{mod}"
            f"{'__all' if plot_all else ''}__dpt-{dpts}__tpt-{tpts}.png",
            dpi=300,
        )
    close_plt()  # in case you call this function multiple times with different settings


def heatmap_final_measure(
      file: str,
      avg_last_n: int = 5,
      to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
      aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
      show: bool = True,
      op: Literal["+", "*"] = "+",
):
    settings = pl.scan_csv(file).filter(pl.col("op") == op).select(
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
            aggregate_method=aggregate_method,
            op=op,
        )
        y_digits = np.mean(avg_ys[-avg_last_n:])

        _, _, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot=to_plot, use_digits=False,
            aggregate_method=aggregate_method,
            op=op,
        )
        y_tokens = np.mean(avg_ys[-avg_last_n:])

        i = dpts.index(dpt)
        j = tpns.index(tpn)
        ratio = y_digits / (y_tokens + 1e-6)
        heatmap[i, j] = ratio

    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(6,5))
    sns.heatmap(
        heatmap, annot=True, fmt='.3f', cmap='viridis', 
        vmin=0.95 if "accuracy" in to_plot else None, vmax=1.01 if "accuracy" in to_plot else None,
        center=1.0, xticklabels=tpns, yticklabels=dpts,
        annot_kws={'size': 8}, cbar_kws={
            'label': (
                f"{aggregate_method.capitalize()} {TO_PLOT_TO_LABEL[to_plot]}: "
                + r"$\frac{\mathrm{MoT}}{\mathrm{Baseline}}$"
            ),
        },
    )
    plt.xlabel('Tokens per number')
    plt.ylabel('Digits per token')
    plt.title(f"{aggregate_method.capitalize()} {TO_PLOT_TO_LABEL[to_plot]}: MoT / Baseline")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"heatmap__{to_plot}__{aggregate_method}__last_n_samples-{avg_last_n}.png",
            dpi=300,
        )
    close_plt()
    plt.rcParams['text.usetex'] = False

    return heatmap, dpts, tpns


def get_other_metrics(
        file: str,
        mod: int | None = None,
        last_n_samples: int = 1,
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        do_aggregate: bool = True,
):
    if mod is None:
        filter_ = pl.col("mod").is_null()
    else:
        filter_ = pl.col("mod") == mod
    df =(
        pl.scan_csv(file)
        .filter(filter_)
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
    settings = sorted(settings, key=lambda x: x[1])
    settings = sorted(settings, key=lambda x: x[0])

    results = {
        "dpt": [],
        "tpn": [],
        "num_equations_seen": [],
        "num_tokens_seen": [],
        "num_unique_tokens": [],
        "times_token_is_seen": [],
        "num_possible_equations": [],
        "times_eq_seen_in_training": [],
        "final_val_accuracies_digits": [],
        "final_val_accuracies_tokens": [],
        "final_val_full_accuracies_digits": [],
        "final_val_full_accuracies_tokens": [],
    }
    for dpt, tpn in settings:
        _, ys, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_accuracies", use_digits=True, mod=mod,
            aggregate_method=aggregate_method,
        )
        final_val_accuracies_digits = [avg_ys[-last_n_samples:].mean()] if do_aggregate else ys[..., -last_n_samples:].mean(axis=-1).tolist()
        _, ys, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_accuracies", use_digits=False, mod=mod,
            aggregate_method=aggregate_method,
        )
        final_val_accuracies_tokens = [avg_ys[-last_n_samples:].mean()] if do_aggregate else ys[..., -last_n_samples:].mean(axis=-1).tolist()
        _, ys, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_full_accuracies", use_digits=True, mod=mod,
            aggregate_method=aggregate_method,
        )
        final_val_full_accuracies_digits = [avg_ys[-last_n_samples:].mean()] if do_aggregate else ys[..., -last_n_samples:].mean(axis=-1).tolist()
        _, ys, avg_ys = load_xs_ys_avg_y(
            file=file, max_digits_per_token=dpt, max_tokens_per_num=tpn,
            to_plot="val_full_accuracies", use_digits=False, mod=mod,
            aggregate_method=aggregate_method,
        )
        final_val_full_accuracies_tokens = [avg_ys[-last_n_samples:].mean()] if do_aggregate else ys[..., -last_n_samples:].mean(axis=-1).tolist()

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

        factor = 1 if do_aggregate else ys.shape[0]
        results["dpt"].extend([dpt] * factor)
        results["tpn"].extend([tpn] * factor)
        results["num_equations_seen"].extend([num_equations_seen] * factor)
        results["num_tokens_seen"].extend([num_tokens_seen] * factor)
        results["num_unique_tokens"].extend([num_unique_tokens] * factor)
        results["times_token_is_seen"].extend([round(times_token_is_seen)] * factor)
        results["num_possible_equations"].extend([num_possible_equations] * factor)
        results["times_eq_seen_in_training"].extend([times_eq_seen_in_training] * factor)
        results["final_val_accuracies_digits"].extend(final_val_accuracies_digits)
        results["final_val_accuracies_tokens"].extend(final_val_accuracies_tokens)
        results["final_val_full_accuracies_digits"].extend(final_val_full_accuracies_digits)
        results["final_val_full_accuracies_tokens"].extend(final_val_full_accuracies_tokens)
    return results


def print_other_metrics(
        file: str, mod: int | None = None, last_n_samples: int = 1,
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        tablefmt: str = "pipe",
        exclude: list[str] | None = None,
):
    exclude = exclude or []
    results = get_other_metrics(
        file=file, mod=mod, last_n_samples=last_n_samples, aggregate_method=aggregate_method,
    )
    name_map = {
        "dpt": "dpt",
        "tpn": "tpn",
        "times_token_is_seen": "times tok seen",
        "times_eq_seen_in_training": "times eq. seen",
        "final_val_accuracies_digits": "val acc (digits)",
        "final_val_accuracies_tokens": "val acc (tokens)",
        "final_val_full_accuracies_digits": "val acc full (digits)",
        "final_val_full_accuracies_tokens": "val acc full (tokens)",
    }
    results = {name_map[k]: v for k, v in results.items() if k in name_map and k not in exclude}
    print(tabulate(results, headers="keys", intfmt=",", floatfmt=",.3f", tablefmt=tablefmt))


def scatter_metric_over_times_tok_or_eq_seen(
        file: str, mod: int | None = None, last_n_samples: int = 1,
        to_plot: Literal["val_accuracies", "val_full_accuracies"] = "val_accuracies",
        plot_over: Literal["times_token_is_seen", "times_eq_seen_in_training"] = "times_token_is_seen",
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        do_aggregate: bool = True,
        fit_order: int = 1,
        confidence_interval: int | None = 95,
        show: bool = True,
):
    results = get_other_metrics(
        file=file, mod=mod, last_n_samples=last_n_samples,
        aggregate_method=aggregate_method, do_aggregate=do_aggregate,
    )
    
    sns.regplot(
        x=results[plot_over],
        y=results[f"final_{to_plot}_digits"],
        label="MoT",
        scatter_kws={'alpha':0.5},
        order=fit_order,
        ci=confidence_interval,
    )
    sns.regplot(
        x=results[plot_over],
        y=results[f"final_{to_plot}_tokens"],
        label="Baseline",
        scatter_kws={'alpha':0.5},
        order=fit_order,
        ci=confidence_interval,
    )
    

    plt.xlabel(
        "average time a token is seen"
        if plot_over == "times_token_is_seen"
        else "average time an equation is seen"
    )
    plt.ylabel(f"final {TO_PLOT_TO_LABEL[to_plot]}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"scatter__{to_plot}__{plot_over}__{aggregate_method if do_aggregate else 'no-aggregation'}"
            f"__fit-order-{fit_order}__ci-{confidence_interval}.png",
            dpi=300,
        )
    close_plt()


def get_num_params(file: str, filters) -> int:
    return int(pl.scan_csv(file).filter(filters).select("num_params").collect()["num_params"].mean())


def plot_results_new(
        file: str,
        digit_mixin_methods: list[Literal["cross_attn", "concat", "noop"]],
        digit_mixout_methods: list[Literal["self_attn", "cross_attn", "noop"]],
        depth: int = 6,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        show: bool = True,
        plot_all: bool = False,
):
    settings = (
        pl.scan_csv(file)
        .filter(
            (pl.col("depth") == depth)
            & pl.col("digit_mixin_method").is_in(digit_mixin_methods)
            & pl.col("digit_mixout_method").is_in(digit_mixout_methods)
        )
        .select("digit_mixin_method", "digit_mixout_method")
        .collect()
        .unique()
    )
    settings = [
        (dmi, dmo)
        for dmi, dmo in zip(
            settings["digit_mixin_method"],
            settings["digit_mixout_method"],
        )
    ]

    settings = list(set(settings))
    colors = generate_distinct_colors(len(settings))
    for (dmi, dmo) in settings:
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file=file,
            digit_mixin_method=dmi,
            digit_mixout_method=dmo,
            depth=depth,
            to_plot=to_plot,
            aggregate_method=aggregate_method,
        )
        
        nparam = format_num_params(get_num_params(
            file,
            filters=(
                (pl.col("digit_mixin_method") == dmi)
                & (pl.col("digit_mixout_method") == dmo)
                & (pl.col("depth") == depth)
            )
        ))
        label = f"{dmi=}, {dmo=}, {nparam=}"
        color = colors.pop(0)
        if plot_all:
            for y in ys:
                plt.plot(xs, y, color=color, alpha=0.2)
        plt.plot(xs, avg_ys, color=color, label=label)
    
    to_plot_to_label = {
        "val_losses": "loss (validation)",
        "val_accuracies": "token accuracy (validation)",
        "val_full_accuracies": "full-number accuracy (validation)",
        "train_losses": "loss (training)",
        "val_l1s": "L1 distance to ground truth (validation)",
        "val_l2s": "L2 distance to ground truth (validation)",
    }
    plt.legend()
    plt.xlabel("step")
    plt.ylabel(to_plot_to_label[to_plot])
    plt.grid()
    plt.tight_layout()

    plt.show()
    close_plt()


def merge_results():
    df = pl.read_csv("results1.csv")
    for i in tqdm(range(2, 8)):
        df2 = pl.read_csv(f"results{i}.csv")
        df = pl.concat([df, df2], how="vertical_relaxed")
    df.write_csv("results.csv")


def merge_results_new():
    files = [f for f in os.listdir(".") if f != "results-mixin.csv" and "results-mixin" in f]
    df = pl.read_csv(files.pop())
    for file in files:
        df2 = pl.read_csv(file)
        df = pl.concat([df, df2], how="vertical_relaxed")
    df.write_csv("results-mixin.csv")


if __name__ == "__main__":
    # merge_results()
    merge_results_new()
    file = "results-mixin.csv"
    plot_results_new(
        file=file,
        digit_mixin_methods=["cross_attn", "concat", "noop"],
        digit_mixout_methods=["self_attn", "noop"],
        to_plot="val_full_accuracies",
    )
    # file = "results.csv"
    # print_other_metrics(
    #     file=file, mod=None, last_n_samples=1, aggregate_method="median",
    #     exclude=["final_val_accuracies_digits", "final_val_accuracies_tokens"],
    # )
    # plot_digits_vs_tokens(
    #     file=file,
    #     max_digits_per_token=4,
    #     max_tokens_per_num=3,
    #     to_plot="val_full_accuracies",
    #     plot_all=False,
    #     op="+",
    #     mod=None,
    #     aggregate_method="mean",
    #     show=True,
    # )
    # heatmap_final_measure(
    #     file=file,
    #     to_plot="val_l1s",
    #     avg_last_n=1,
    #     aggregate_method="median",
    #     op="+",
    #     show=True,
    # )
    # scatter_metric_over_times_tok_or_eq_seen(
    #     file=file, mod=None,
    #     to_plot="val_full_accuracies",
    #     plot_over="times_token_is_seen",
    #     fit_order=1,
    #     confidence_interval=None,
    #     show=True,
    #     do_aggregate=False,
    # )
    # scatter_metric_over_times_tok_or_eq_seen(
    #     file=file, mod=None,
    #     to_plot="val_full_accuracies",
    #     plot_over="times_eq_seen_in_training",
    #     fit_order=1,
    #     confidence_interval=None,
    #     show=True,
    #     do_aggregate=False,
    # )
