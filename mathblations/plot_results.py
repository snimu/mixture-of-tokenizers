
import ast
import os
from typing import Literal

import seaborn as sns
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines # For adding lines
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


def generate_distinct_colors(n: int, palette: str = "colorblind") -> list:
    return list(iter(sns.color_palette(palette, n_colors=n)))
    

TO_PLOT_TO_LABEL = {
    "val_losses": "Loss",
    "val_accuracies": "Token Accuracy",
    "val_full_accuracies": "Full-Number Accuracy",
    "train_losses": "Loss",
    "val_l1s": "L1 Error",
    "val_l2s": "L2 Error",
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


def seconds_to_hhmmss(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def plot_results_new(
        file: str,
        digit_mixin_methods: list[Literal["cross_attn", "concat", "noop"]],
        digit_mixout_methods: list[Literal["self_attn", "noop"]],
        use_digit_self_attn: bool | None = None,
        depth: int | None = None,
        to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
        aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
        plot_all: bool = False,
        loglog: bool = False,
        steps: tuple[int | None, int | None] | None = None,
        ylim: tuple[float | int, float | int] | None = None,
        show: bool = True,
        show_times: bool = False,
):
    steps = steps or (None, None)
    steps = list(steps)
    steps[0] = steps[0] // 100 if steps[0] else None
    steps[1] = steps[1] // 100 if steps[1] else None
    use_digit_self_attn = [False, True] if use_digit_self_attn is None else [use_digit_self_attn]
    settings = (
        pl.scan_csv(file)
        .filter(
            ((pl.col("depth") == depth) if depth else pl.col("depth").is_not_null())
            & pl.col("digit_mixin_method").is_in(digit_mixin_methods)
            & pl.col("digit_mixout_method").is_in(digit_mixout_methods)
            & pl.col("use_digit_self_attn").is_in(use_digit_self_attn)
        )
        .select("digit_mixin_method", "digit_mixout_method", "use_digit_self_attn", "depth")
        .collect()
        .unique()
    )
    settings = [
        (dmi, dmo, dsa, dep)
        for dmi, dmo, dsa, dep in zip(
            settings["digit_mixin_method"],
            settings["digit_mixout_method"],
            settings["use_digit_self_attn"],
            settings["depth"],
        )
    ]

    dmi_names = {
        "cross_attn": "In: Bytes (cross-attn)",
        "concat": "In: Bytes(concat)",
        "noop": "In: Tokens..........."
    }
    dmo_names = {
        "self_attn": "Out: Bytes..",
        "noop": "Out: Tokens"
    }

    settings = list(set(settings))
    for i in range(4):
        settings = sorted(settings, key=lambda x: x[len(x)-1-i])
    colors = generate_distinct_colors(n=len(settings), palette="tab10")
    for (dmi, dmo, dsa, dep) in settings:
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file=file,
            digit_mixin_method=dmi,
            digit_mixout_method=dmo,
            use_digit_self_attn=dsa,
            depth=dep,
            to_plot=to_plot,
            aggregate_method=aggregate_method,
        )
        xs = xs[steps[0]:steps[1]]
        ys = ys[:, steps[0]:steps[1]]
        avg_ys = avg_ys[steps[0]:steps[1]]
        filters=(
            (pl.col("digit_mixin_method") == dmi)
            & (pl.col("digit_mixout_method") == dmo)
            & (pl.col("use_digit_self_attn") == dsa)
            & (pl.col("depth") == dep)
        )
        nparam = format_num_params(get_num_params(file, filters))
        label = f"{dmi_names[dmi]}... {dmo_names[dmo]}... "
        label += f"{dsa=}... " if use_digit_self_attn == [False, True] else ""
        label += f"Size: {nparam}"
        label += f", {dep=}" if depth is None else ""
        if show_times:
            time_taken = pl.scan_csv(file).filter(filters).select("total_time").collect()["total_time"].mean()
            label += f", Time: {seconds_to_hhmmss(time_taken)}"

        color = colors.pop(0)
        if plot_all:
            for y in ys:
                plt.plot(xs, y, color=color, alpha=0.6, linestyle="dotted")
        plt.plot(xs, avg_ys, color=color, label=label, linewidth=2.5)
    
    to_plot_to_label = {
        "val_losses": "Loss (validation)",
        "val_accuracies": "Token accuracy (validation)",
        "val_full_accuracies": "Full-number accuracy (validation)",
        "train_losses": "Loss (training)",
        "val_l1s": "L1 distance to ground truth (validation)",
        "val_l2s": "L2 distance to ground truth (validation)",
    }

    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if ylim and not loglog:
        plt.ylim(*ylim)
    plt.legend(
        fontsize=10,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        borderaxespad=0.,
    )
    plt.subplots_adjust(
        left=0.1,
        right=0.95,
        bottom=0.25,
        top=0.90,
    )
    plt.xlabel("step")
    plt.ylabel(to_plot_to_label[to_plot])
    plt.grid()
    plt.tight_layout()

    if show:
        plt.show()
        close_plt()
    else:
        plt.savefig(
            f"plot_digits_vs_tokens__{to_plot}__{aggregate_method}"
            + ("__all" if plot_all else ''),
            dpi=300,
        )
    

def adjust_lightness(color, amount=0.5):
    """Adjust the lightness of a color. amount > 1 lightens, < 1 darkens."""
    try:
        import matplotlib.colors as mc
        c = mc.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        new_l = max(0, min(1, amount * l))
        return colorsys.hls_to_rgb(h, new_l, s)
    except ImportError:
        print("Warning: matplotlib not found. Cannot adjust lightness.")
        if amount > 1:
            return tuple(min(1, c + (1-c)*0.4) for c in mc.to_rgb(color)) # Mix with white approx
        else:
             return tuple(max(0, c * 0.6) for c in mc.to_rgb(color)) # Darken approx


def plot_results_grid(
    file: str,
    to_plot: Literal["val_losses", "val_accuracies", "val_full_accuracies", "train_losses", "val_l1s", "val_l2s"] = "val_accuracies",
    plot_all: bool = False,
    aggregate_method: Literal["mean", "median", "max", "min"] = "mean",
    show: bool = True,
    ylim: tuple[float | int, float | int] | None = None,
    loglog: bool = False,
    steps: tuple[int | None, int | None] | None = None,
):
    input_methods = ["noop", "concat", "cross_attn"]
    output_methods = ["noop", "self_attn"]

    # Full names for annotations/titles
    input_names_full = { "noop": "Tokens", "concat": "Bytes (Concat)", "cross_attn": "Bytes (Cross-Attn)" }
    output_names_full = { "noop": "Tokens", "self_attn": "Bytes (Self-Attn)" }

    # Fixed parameters
    depth = 6
    use_digit_self_attn = False

    # Define base colors for rows
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # tab10 blue, orange, green

    # Make figure wider
    fig, axes = plt.subplots(
        nrows=len(input_methods),
        ncols=len(output_methods),
        figsize=(11.0, 7.0), # Keep original width for now, adjust if needed
        sharex=True,
        sharey=True
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
        axes = np.array([[axes]])

    # Define desired x-ticks
    x_ticks = [0, 5000, 10000, 15000, 20000]
    x_tick_labels = ['0', '5k', '10k', '15k', '20k']

    plot_idx = 0
    for r, dmi in enumerate(input_methods):
        base_color = base_colors[r]
        for c, dmo in enumerate(output_methods):
            ax = axes[r, c]

            # Determine color shade
            if c == 0: plot_color = adjust_lightness(base_color, 1.3) # Lighter
            else: plot_color = adjust_lightness(base_color, 0.7) # Darker

            try:
                # Load potentially full data first
                xs_full, ys_full, avg_ys_full = load_xs_ys_avg_y(
                    file=file,
                    digit_mixin_method=dmi,
                    digit_mixout_method=dmo,
                    use_digit_self_attn=use_digit_self_attn,
                    depth=depth,
                    to_plot=to_plot,
                    aggregate_method=aggregate_method,
                )

                # Apply step slicing if specified
                start_idx, end_idx = None, None
                if steps and steps[0] is not None:
                    start_idx = np.searchsorted(xs_full, steps[0], side='left')
                if steps and steps[1] is not None:
                    # Ensure end_idx is within bounds after slicing
                    end_idx_search = np.searchsorted(xs_full, steps[1], side='right')
                    end_idx = min(end_idx_search, len(xs_full))


                xs = xs_full[start_idx:end_idx]
                # Ensure ys and avg_ys slicing matches xs
                if ys_full.ndim == 2 and ys_full.shape[1] > 0:
                    ys = ys_full[:, start_idx:end_idx]
                else:
                    ys = np.array([[]]) # Handle empty ys_full case

                if avg_ys_full.ndim == 1 and avg_ys_full.shape[0] > 0:
                     avg_ys = avg_ys_full[start_idx:end_idx]
                else:
                     avg_ys = np.array([]) # Handle empty avg_ys_full case


                if xs.size == 0 or avg_ys.size == 0:
                    # print(f"Warning: No data found for In:{input_names_short[dmi]} / Out:{output_names_short[dmo]} in step range {steps}. Skipping plot.") # Optional: uncomment for debug
                    # Make empty plots less visually intrusive
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                    ax.spines[:].set_visible(False) # Hide spines for empty plots
                    continue # Skip plotting for this subplot

                if plot_all and ys.shape[0] > 0 and ys.shape[1] > 0: # Check ys has data points
                    for y_run in ys:
                        ax.plot(xs, y_run, color=plot_color, alpha=0.3, linestyle="-", linewidth=0.7)

                if avg_ys.size > 0: # Check avg_ys has data points
                    # Removed label= from here as it's not used
                    ax.plot(xs, avg_ys, color=plot_color, linewidth=1.8)
                else: # Handle case where avg_ys became empty after slicing
                     # print(f"Warning: Average data became empty after slicing for In:{input_names_short[dmi]} / Out:{output_names_short[dmo]}. Skipping average line.") # Optional: uncomment for debug
                     pass # Don't plot if no average data


            except Exception as e:
                print(f"Error loading/plotting data for In:{input_names_full[dmi]} / Out:{output_names_full[dmo]}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                ax.spines[:].set_visible(False) # Hide spines for error plots
                continue

            # --- Subplot Configuration ---
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
            # Removed legend display as individual line labels were removed
            # handles, labels = ax.get_legend_handles_labels()
            # if handles:
            #      ax.legend(loc='best', fontsize=9)


            if loglog:
                ax.set_xscale('log')
                ax.set_yscale('log')
                from matplotlib.ticker import LogFormatterSciNotation, NullFormatter, LogLocator
                # Use LogLocator for better tick placement on log scales
                ax.yaxis.set_major_locator(LogLocator(numticks=6)) # Adjust numticks as needed
                ax.xaxis.set_major_locator(LogLocator(numticks=8))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
                # Set tick label size for log scale as well
                ax.tick_params(axis='both', which='major', labelsize=9) # Apply to major ticks
            elif ylim:
                 ax.set_ylim(*ylim)

            # --- Axis Labels and Specific Row/Column Titles ---
            if c == 0: # Leftmost column
                 # Specific Row Annotation (Input Method) - Black color
                 ax.annotate(input_names_full[dmi], xy=(-0.18, 0.5), xycoords='axes fraction', # Adjusted x slightly for larger font
                             textcoords='offset points', xytext=(0,0),
                             ha='right', va='center', fontsize=12, rotation=90, color="black") # Set to black
                 # Y-axis metric label
                 ax.set_ylabel(TO_PLOT_TO_LABEL[to_plot], fontsize=9)
                 # ***** MODIFICATION: Set Y-tick label size *****
                 ax.tick_params(axis='y', labelsize=9)
            else:
                # Turn off y-tick labels for inner/right columns
                ax.tick_params(axis='y', labelleft=False)

            if r == len(input_methods) - 1: # Bottom row
                ax.set_xlabel("Step", fontsize=9)
                if not loglog: # Set specific ticks only for linear scale
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_tick_labels)
                    # Set x-axis limits to ensure ticks are visible if data doesn't span full range
                    current_xlim = ax.get_xlim()
                    # Ensure xlim accommodates the specified ticks if data is narrower
                    ax.set_xlim(left=min(current_xlim[0], x_ticks[0] - 500),
                                right=max(current_xlim[1], x_ticks[-1] + 500))
                # ***** MODIFICATION: Set X-tick label size *****
                # Apply labelsize=9 regardless of log scale (affects major ticks)
                ax.tick_params(axis='x', labelsize=9)
            else:
                # Turn off x-tick labels for inner/top rows
                ax.tick_params(axis='x', labelbottom=False)

            if r == 0: # Top row - Add Specific Column Titles (Output Method) - Black color
                 ax.set_title(output_names_full[dmo], fontsize=12, pad=10, color='black') # Ensure black color


            plot_idx += 1

    # --- Hierarchical Titles ---
    # Define coordinates based on subplots_adjust values
    left_margin = 0.15 # Keep increased left margin
    right_margin = 0.97
    top_margin = 0.90 # May need slight adjustment if titles overlap axes
    bottom_margin = 0.10
    input_title_x = 0.02 # Position of the "Input" text (kept small and fixed)
    output_title_y = 0.97 # Position of the "Output" text

    # Main "Output" title
    fig.text( (left_margin + right_margin) / 2 , output_title_y, "Output", ha='center', va='center', fontsize=12, fontweight='bold') # Center between new margins

    # Main "Input" title (Position remains fixed relative to figure)
    fig.text(input_title_x, 0.5, "Input", ha='left', va='center', rotation=90, fontsize=12, fontweight='bold')

    # --- Figure Configuration ---
    # Adjust spacing - Use the modified margins
    plt.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin,
                        wspace=0.08, hspace=0.18) # wspace/hspace might need tweaking

    if show:
        plt.show()
        close_plt()
    else:
        plt.savefig("plot_results_grid.png", dpi=300)


def merge_results():
    df = pl.read_csv("results1.csv")
    for i in tqdm(range(2, 8)):
        df2 = pl.read_csv(f"results{i}.csv")
        df = pl.concat([df, df2], how="vertical_relaxed")
    df.write_csv("results.csv")


def merge_results_new():
    files = [f for f in os.listdir(".") if "results-mixin" in f]
    f0 = files.pop()
    df = pl.read_csv(f0)
    os.remove(f0)
    for file in files:
        df2 = pl.read_csv(file)
        df = pl.concat([df, df2], how="vertical_relaxed")
        os.remove(file)
    df.write_csv("results-mixin.csv")


if __name__ == "__main__":
    # merge_results()
    merge_results_new()
    file = "results-mixin.csv"
    # plot_results_grid(
    #     file=file,
    #     to_plot="val_full_accuracies",
    #     plot_all=True, # Show individual runs as faint lines
    #     aggregate_method="mean",
    #     ylim=(-0.05, 1.05), # Example Y limit for accuracy
    #     show=False,
    # )
    plot_results_new(
        file=file,
        digit_mixin_methods=["concat"],
        digit_mixout_methods=["self_attn", "noop"],
        to_plot="val_full_accuracies",
        plot_all=True,
        use_digit_self_attn=False,
        ylim=(0, 1),
        depth=6,
        loglog=False,
        steps=(None, None),
        show_times=True,
        show=False,
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
