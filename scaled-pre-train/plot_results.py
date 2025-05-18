
import json
import random
from collections import Counter
from typing import Literal

import dspy
import numpy as np
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


def compare_generations(
        files: list[str],
        names: list[str] | None = None,
        save_to: str | None = None,
        cmp_model: str | None = None,
        n_samples: int = 10,
):
    names = names or [file.replace(".json", "") for file in files]
    assert len(files) == len(names)
    generations = []
    for file in files:
        with open(f"results/generation/{file}", "r") as f:
            generations.append(json.loads(f.read()))
    
    queries = [[g["query"] for g in gen] for gen in generations]
    assert set(queries[0]) == set(sum(queries, []))  # all generations have the same queries
    
    # Set up LLM
    if cmp_model:
        llm = dspy.LM(f"openai/{cmp_model}", temperature=0.2)
        dspy.settings.configure(lm=llm)
        judge = dspy.ChainOfThought(
            dspy.Signature(
                "query, answer1, answer2 -> better_answer_idx: int",
                "1 or 2. Criteria: grammar, internal consistency, consistency with query.",
            )
        )
    results = dict(summary=Counter())
    loop = tqdm(range(n_samples), disable=not cmp_model)
    for _ in loop:
        query = random.choice(queries[0])
        if query not in results:
            results[query] = Counter()
        answers = []

        # Extract answers from generations
        for gen in generations:
            for g in gen:
                if g["query"] == query:
                    answers.append(random.choice(g["responses"]))
                    break
        
        # Give choices
        order = random.sample(range(len(answers)), len(answers))
        if cmp_model:
            resp_idx = -1
            while resp_idx-1 not in order:
                resp_idx = judge(query=query, answer1=answers[order[0]], answer2=answers[order[1]]).better_answer_idx
        else:
            print(f"\n\nQUERY:\n{query}")
            answers_repr = ""
            for idx, answer_idx in enumerate(order):
                answer = answers[answer_idx]
                answers_repr += f"\n\n{idx+1}. {answer}"
            print(answers_repr)
            resp_idx = int(input("\nChoose answer: "))
        answer_idx = order.index(resp_idx-1)
        results[query][names[answer_idx]] += 1
        results["summary"][names[answer_idx]] += 1
        if not cmp_model:
            print("\n\n")
            print(results["summary"])
        else:
            loop.set_description(f"{results['summary']}")
        if save_to:
            with open(save_to, "w") as f:
                f.write(json.dumps(results, indent=2))


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
    #         "noop-noop-1024-1024-1024-temp-050.json",
    #         "noop-noop-1024-1024-1024-temp-100.json",
    #         "concat-noop-48-256-1024-greedy.json",
    #         "concat-noop-48-256-1024-temp-050.json",
    #         "concat-noop-48-256-1024-temp-100.json",
    #     ],
    #     names=["T (0.0)", "T (0.5)", "T (1.0)", "B (0.0)", "B (0.5)", "B (1.0)"],
    #     extra_forbidden_evals=["arithmetic", "cola", "lambada_openai", "mrpc", "rte", "wnli"],
    #     tablefmt="pipe",
    # )
    for model in ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]:
        compare_generations(
            ["bytes-tokens.json", "tokens-tokens.json"],
            save_to=f"results/generation/comparison_{model.replace('-', '_')}.json",
            cmp_model=model,
            n_samples=500,
        )
