
import json
import os
import random
import itertools
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

import dspy
import numpy as np
import tiktoken
from tqdm import tqdm
from tabulate import tabulate
import polars as pl


def load_data(
        file: str,
        mixin_mixout: list[tuple[str, str]] | None = None,
        ignore_add: bool = False,
):
    assert file.endswith(".json")
    with open(f"results/runs/{file}", "r") as f:
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


def tabulate_evals(
        files: list[str],
        names: list[str] | None = None,
        extra_forbidden_metrics: list[str] | None = None,
        extra_forbidden_evals: list[str] | None = None,
        tablefmt: str = "pipe",
):
    forbidden_metrics = ["alias", "stderr"] + (extra_forbidden_metrics or [])
    forbidden_evals = ["mmlu_"] + (extra_forbidden_evals or [])
    higher_is_better = ["acc", "mcc", "f1", "rouge", "bleu"]
    results = []
    if names:
        assert len(files) == len(names)
    names = names or [file.replace(".json", "") for file in files]
    for file in files:
        with open(f"results/evals/{file}", "r") as f:
            result = json.loads(f.read())["results"]
        
        metric_names = []
        metric_values = []
        for eval in result:
            if any(term in eval for term in forbidden_evals):
                continue
            for metric in result[eval]:
                if any(term in metric for term in forbidden_metrics):
                    continue
                metric_names.append(eval + " " + metric.replace(",none", ""))
                metric_values.append(result[eval][metric])
        results.append({"metrics": metric_names, "results": metric_values})
    tabledata = {"metric": results[0]["metrics"], **{names[i]: results[i]["results"] for i in range(len(files))}}

    # Add a "best" column
    tabledata["best"] = []
    for metric_idx, metric in enumerate(tabledata["metric"]):
        compare_fct = max if any(term in metric for term in higher_is_better) else min
        values = [float(tabledata[names[name_idx]][metric_idx]) for name_idx in range(len(files))]
        best_value = compare_fct(values)
        best_idx = values.index(best_value)
        tabledata["best"].append(names[best_idx])

    table = tabulate(tabledata, headers="keys", floatfmt=".4f", tablefmt=tablefmt)
    print(f"\n\n{table}\n\n")


def judge_completions(model: str, response1: str, response2: str) -> tuple[int, int, bool]:
    llm = dspy.LM(f"openai/{model}", temperature=0.2)
    dspy.settings.configure(lm=llm)
    switched = bool(random.randint(0, 1))
    if switched:
        response1, response2 = response2, response1
    judge = dspy.ChainOfThought(
        dspy.Signature(
            "query, answer1, answer2 -> better_answer_idx: int",
            "1 or 2. Criteria: grammar, internal consistency, consistency with query.",
        )
    )
    for _ in range(5):  # maximum of 5 tries per judgement
        chosen_idx = judge(query=response1, answer1=response1, answer2=response2).better_answer_idx - 1
        if chosen_idx in (0, 1):
            break
    if chosen_idx not in (0, 1):
        return (-1, -1, False)
    position = chosen_idx
    if switched:
        chosen_idx = 1 - chosen_idx
    return chosen_idx, position, switched


def extract_temperature(file: str) -> float:
    return float(file.split("-")[-1].split(".")[0]) / 100  # float('050') == 50.0 -> divide by 100


def compare_two_generations(
        files: list[str],
        names: list[Literal["MoT", "Baseline"]],
        tokens_out: list[int] | None = None,
        save_to: str | None = None,
        cmp_model: str = "gpt-4o-mini",
        n_samples_per_completion: int = 10,
        queries_file: str = "queries.json",
):
    assert len(files) == len(names) == 2, f"{len(files)=}, {len(names)=}"
    tokens_out = tokens_out or [20, 100, 500]
    generations = []
    for file in files:
        with open(f"results/generation/{file}", "r") as f:
            generations.append(json.loads(f.read()))
    assert len(generations[0]) == len(generations[1]), f"{len(generations[0])=}, {len(generations[1])=}"

    with open(queries_file, "r") as f:
        queries = json.loads(f.read())
    assert len(queries) == len(generations[0]), f"{len(queries)=}, {len(generations[0])=}"
    enc = tiktoken.encoding_for_model("gpt-2")
    temperature0 = extract_temperature(files[0])
    temperature1 = extract_temperature(files[1])

    results = {
        "name": [],
        "tokens_in": [],
        "tokens_out": [],
        "sample_idx": [],
        names[0]: [],
        names[1]: [],
        "chosen_idx": [],
        "order_switched": [],
        "temperature1": [],
        "temperature2": [],
        "cmp_model": [],
    }
    loop = tqdm(range(len(generations[0])))
    for query_idx in loop:
        completion0 = generations[0][query_idx]
        completion1 = None
        for completion in generations[1]:
            if completion["name"] == completion0["name"]:
                completion1 = completion
                break
        assert completion1 is not None
        assert set(completion0["tokens_in"]) == set(completion1["tokens_in"])
        assert completion0["name"] == completion1["name"], f"{completion0['name']=}, {completion1['name']=}"
        query = queries[query_idx]["text"]

        for toks_in_idx, toks_in in enumerate(set(completion0["tokens_in"])):
            query = enc.decode(enc.encode(query)[:toks_in])
            candidates0 = [completion0["responses"][i] for i in range(len(completion0["responses"])) if completion0["tokens_in"][i] == toks_in]
            candidates1 = [completion1["responses"][i] for i in range(len(completion1["responses"])) if completion1["tokens_in"][i] == toks_in]
            random.shuffle(candidates0)
            random.shuffle(candidates1)
            for toks_out_idx, toks_out in enumerate(tokens_out):
                samples0 = itertools.cycle([enc.decode(enc.encode(resp)[:toks_out]) for resp in candidates0])
                samples1 = itertools.cycle([enc.decode(enc.encode(resp)[:toks_out]) for resp in candidates1])
                judgements = []
                with ThreadPoolExecutor(max_workers=n_samples_per_completion) as executor:
                    futures = [executor.submit(judge_completions, cmp_model, next(samples0), next(samples1)) for _ in range(n_samples_per_completion)]
                    for future in futures:
                        judgements.append(future.result())
                for sample_idx, (chosen_idx, position, switched) in enumerate(judgements):
                    if chosen_idx == -1:
                        continue

                    # Record the results
                    results["chosen_idx"].append(position)  # is to track if the first or second response is chosen; for normalizing
                    results["order_switched"].append(int(switched))

                    resp0_is_better = chosen_idx == 0
                    resp1_is_better = chosen_idx == 1

                    results["name"].append(completion0["name"])
                    results["tokens_in"].append(toks_in)
                    results["tokens_out"].append(toks_out)
                    results["sample_idx"].append(sample_idx)
                    results[names[0]].append(resp0_is_better)
                    results[names[1]].append(resp1_is_better)
                    results["temperature1"].append(temperature0)
                    results["temperature2"].append(temperature1)
                    results["cmp_model"].append(cmp_model)

                    # Save the results
                    df = pl.DataFrame(results)
                    df.write_csv(f"results/generation/{save_to}.csv")

                    # Give feedback
                    first, second = df[names[0]].sum(), df[names[1]].sum()
                    description = f"toks_in: {toks_in_idx+1}/{len(set(completion0['tokens_in']))}, "
                    description += f"toks_out: {toks_out_idx+1}/{len(tokens_out)}; "
                    description += f"{names[0]}: {first}, {names[1]}: {second} ({100*first/(first+second):.2f}% {names[0]}); "
                    description += f"mean idx: {df['chosen_idx'].mean():.2f}; "
                    description += f"mean shuffle: {df['order_switched'].mean():.2f}"
                    loop.set_description(description)


def compare_generations(
        file_pairs: list[tuple[str, str]],
        name_pairs: list[tuple[Literal["MoT", "Baseline"], Literal["MoT", "Baseline"]]],
        cmp_models: list[str],
        save_to: str,
        tokens_out: list[int] | None = None,
        n_samples_per_completion: int = 10,
        queries_file: str = "queries.json",
):
    assert len(file_pairs) == len(name_pairs), f"{len(file_pairs)=}, {len(name_pairs)=}"
    df = None
    for (file1, file2), (name1, name2) in zip(file_pairs, name_pairs):
        assert file1.endswith(".json") and file2.endswith(".json")
        for cmp_model in cmp_models:
            compare_two_generations(
                files=[file1, file2],
                names=[name1, name2],
                tokens_out=tokens_out,
                save_to="_tmp.csv",
                cmp_model=cmp_model,
                n_samples_per_completion=n_samples_per_completion,
                queries_file=queries_file,
            )
            df_local = pl.read_csv("_tmp.csv")
            if df is None:
                df = df_local
            else:
                df = pl.cat([df, df_local])
            df.write_csv(f"results/generation/{save_to}.csv")


def tabulate_comparisons(temperature: float):
    if temperature == 0:
        temp = "T000"
    elif temperature < 1:
        temp = f"T0{int(temperature*100)}"
    else:
        temp = f"T{int(temperature*100)}"
    file_titles = [
        file
        for file in os.listdir("results/generation")
        if file.endswith("-summary.csv") and temp in file
        
    ]
    dfs = []
    for file in file_titles:
        dfs.append(pl.read_csv(f"results/generation/{file}"))
    df: pl.DataFrame = pl.concat(dfs)
    results = {col: [] for col in df.columns}
    for domain in dfs[0]["domain"]:
        df_domain = df.filter(pl.col("domain") == domain)
        for col in df.columns:
            if col == "domain":
                continue
            if col in ("% MoT", "mean chosen_idx"):
                results[col].append(df_domain[col].mean())
            else:
                results[col].append(df_domain[col].sum())
    results["domain"] = dfs[0]["domain"]
    table = tabulate(results, headers="keys", floatfmt=".4f", tablefmt="pipe")
    print(table)


if __name__ == "__main__":
    file = "results100_000steps.json"
    # tabulate_results(
    #     file,
    #     mixin_mixout=[("concat", "split"), ("concat", "copy")],
    #     ignore_add=True,
    #     data_frac=0.1,
    #     tablefmt="pipe",
    #     info_columns=["mean", "std", "n_params", "step_time"],
    # )

    # tabulate_evals(
    #     files=[
    #         "noop-noop-1024-1024-1024-greedy.json",
    #         "concat-noop-48-256-1024-greedy.json",
    #     ],
    #     names=["Baseline", "MoT"],
    #     extra_forbidden_evals=["truthfulqa"],
    #     tablefmt="pipe",
    # )

    file_pairs=[
        ("bytes-tokens-000.json", "tokens-tokens-000.json"),
        ("bytes-tokens-010.json", "tokens-tokens-010.json"),
        ("bytes-tokens-020.json", "tokens-tokens-020.json"),
        ("bytes-tokens-030.json", "tokens-tokens-030.json"),
        ("bytes-tokens-040.json", "tokens-tokens-040.json"),
        ("bytes-tokens-050.json", "tokens-tokens-050.json"),
        ("bytes-tokens-060.json", "tokens-tokens-060.json"),
        ("bytes-tokens-070.json", "tokens-tokens-070.json"),
        ("bytes-tokens-080.json", "tokens-tokens-080.json"),
        ("bytes-tokens-090.json", "tokens-tokens-090.json"),
        ("bytes-tokens-100.json", "tokens-tokens-100.json"),
    ]
    name_pairs=[("MoT", "Baseline")] * len(file_pairs)
    cmp_models=["gpt-4.1"]
    compare_generations(
        file_pairs=file_pairs,
        name_pairs=name_pairs,
        cmp_models=cmp_models,
        save_to="comparison-between-models.csv",
        tokens_out=[20, 100, 500],
    )

    file_pairs=[]
    for i in range(10):
        T1 = f"0{i}" + ("0" if i < 10 else "")
        for j in range(i+1, 11):
            T2 = f"0{j}" + ("0" if j < 10 else "")
            file_pairs.append((f"bytes-tokens-{T1}.json", f"bytes-tokens-{T2}.json"))
    name_pairs=[("MoT", "MoT")] * len(file_pairs)
    cmp_models=["gpt-4.1"]
    compare_generations(
        file_pairs=file_pairs,
        name_pairs=name_pairs,
        cmp_models=cmp_models,
        save_to="comparison-temperatures_MoT.csv",
        tokens_out=[20, 100, 500],
        n_samples_per_completion=5,
    )

    file_pairs=[]
    for i in range(10):
        T1 = f"0{i}" + ("0" if i < 10 else "")
        for j in range(i+1, 11):
            T2 = f"0{j}" + ("0" if j < 10 else "")
            file_pairs.append((f"tokens-tokens-{T1}.json", f"tokens-tokens-{T2}.json"))
    name_pairs=[("Baseline", "Baseline")] * len(file_pairs)
    cmp_models=["gpt-4.1"]
    compare_generations(
        file_pairs=file_pairs,
        name_pairs=name_pairs,
        cmp_models=cmp_models,
        save_to="comparison-temperatures_Baseline.csv",
        tokens_out=[20, 100, 500],
        n_samples_per_completion=5,
    )
