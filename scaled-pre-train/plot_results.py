
import json
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import polars as pl


def load_data(
        file: str,
        mixin_mixout: list[tuple[str, str]] | None = None,
        ignore_add: bool = False,
):
    assert file.endswith(".json")
    with open(file, "r") as f:
        data = json.loads(f.read())
    assert isinstance(data, list)
    assert all(isinstance(item, dict) for item in data)

    if mixin_mixout is not None:
        data = [item for item in data if (item["byte_mixin_method"], item["byte_mixout_method"]) in mixin_mixout]
    if ignore_add:
        data = [item for item in data if not item["add_padded_and_pulled"]]
    return data


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


def tabulate_results(
        file: str,
        mixin_mixout: list[tuple[str, str]] | None = None,
        ignore_add: bool = False,
        data_frac: float | None = None,
        info_columns: list[Literal["n_params", "step_time", "min", "mean", "median", "std"]] | None = None,
        tablefmt: str = "simple",
):
    info_columns = info_columns or ["mean", "std"]
    data = load_data(file, mixin_mixout, ignore_add)

    stats = dict(mixin=[], mixout=[], D_model=[], D_tok=[], D_byte=[])
    if includes_mixout := (mixin_mixout is None or any(mixout != "noop" for _, mixout in mixin_mixout)):
        stats["# layers out"] = []
    if not ignore_add:
        stats["add"] = []
    if "n_params" in info_columns:
        stats["# params"] = []
    if "step_time" in info_columns:
        stats["step time [s]"] = []
    if "min" in info_columns:
        stats["min fw"], stats["min fm"] = [], []
    if "mean" in info_columns:
        stats["mean fw"], stats["mean fm"] = [], []
    if "std" in info_columns:
        stats["std fw"], stats["std fm"] = [], []
    if "median" in info_columns:
        stats["median fw"], stats["median fm"] = [], []
    for item in data:
        stats["mixin"].append(item["byte_mixin_method"])
        stats["mixout"].append(item["byte_mixout_method"])
        stats["D_model"].append(item["model_dim"])
        stats["D_tok"].append(item["token_dim"])
        stats["D_byte"].append(item["byte_dim"])

        if includes_mixout:
            stats["# layers out"].append(item["n_layer_out"])
        if not ignore_add:
            stats["add"].append(item["add_padded_and_pulled"])
        if "n_params" in info_columns:
            stats["# params"].append(format_num_params(item["num_params"]))
        if "step_time" in info_columns:
            stats["step time [s]"].append(item["step_avg_train_time"])

        losses_fw = np.array(item["val_losses_fw"])
        losses_fm = np.array(item["val_losses_fm"])
        if data_frac is not None:
            assert data_frac >= 0
            start = int(len(losses_fw) * (1-data_frac))
            losses_fw = losses_fw[start:]
            start = int(len(losses_fm) * (1-data_frac))
            losses_fw = losses_fm[start:]

        if "min" in info_columns:
            stats["min fw"].append(float(losses_fw.min()))
            stats["min fm"].append(float(losses_fm.min()))
        if "mean" in info_columns:
            stats["mean fw"].append(float(losses_fw.mean()))
            stats["mean fm"].append(float(losses_fm.mean()))
        if "std" in info_columns:
            stats["std fw"].append(float(losses_fw.std()))
            stats["std fm"].append(float(losses_fm.std()))
        if "median" in info_columns:
            stats["median fw"].append(float(np.median(losses_fw)))
            stats["median fm"].append(float(np.median(losses_fm)))
    stats = pl.DataFrame(stats).sort(["D_byte", "D_tok", "mixout", "mixin"], descending=True)
    stats = {k: v.to_list() for k, v in stats.to_dict().items()}
    table = tabulate(
        stats, headers="keys", floatfmt=".2f", tablefmt=tablefmt,
    )
    data_frac = data_frac or 1.0
    print("\n*fw*: validation loss on fineweb data;\n*fm*: validation loss on finemath data.")
    print(f"Statistics calculated over last {round(data_frac * 100)}% of loss curve.\n")
    print(table)
    print()


if __name__ == "__main__":
    file = "results100_000steps.json"
    tabulate_results(
        file,
        mixin_mixout=[("concat", "split"), ("concat", "copy")],
        ignore_add=True,
        data_frac=0.1,
        tablefmt="pipe",
        info_columns=["mean", "std", "n_params", "step_time"],
    )

