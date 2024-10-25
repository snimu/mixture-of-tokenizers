

"""
INSTRUCTIONS:

- you need to run `huggingface-cli login` once and add your token
- you need access to the meta-llama weights

ON LAMBDA LABS:

- you need to run `pip install -U torch torchvision` before running this script
"""
import argparse
from typing import Literal

import torch
from torch import nn
from transformers import AutoModelForCausalLM

def load_model(which: Literal["base", "instruct"] = "base") -> nn.Module:
    if which == "base":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    else:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    return model


def measure_weight_norm(model: nn.Module, measure: Literal["L1", "L2"] = "L2") -> float:
    numels = []
    norms = []

    for param in model.parameters():
        numels.append(param.numel())
        norms.append(torch.abs(param).sum().item() if measure == "L1" else param.pow(2).sum().item())

    return sum(norms) / sum(numels)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="L2", choices=["L1", "L2"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    base_model = load_model("base")
    instruct_model = load_model("instruct")

    base_norm = measure_weight_norm(base_model, measure=args.measure)
    instruct_norm = measure_weight_norm(instruct_model, measure=args.measure)

    for param_base, param_instruct in zip(base_model.parameters(), instruct_model.parameters()):
        param_instruct.data = param_instruct.data - param_base.data

    diff_norm = measure_weight_norm(instruct_model, measure=args.measure)

    print(f"MEASURE: {args.measure}")
    print(f"Base model weight norm: {base_norm:.4f}")
    print(f"Instruct model weight norm: {instruct_norm:.4f}")
    print(f"Diff model weight norm: {diff_norm:.4f}")
    print(f"diff_norm / base_norm: {diff_norm / base_norm:.4f}")

    """
    RESULTS:

    Base model weight norm: 30886.8619
    Instruct model weight norm: 27765.2804
    Diff model weight norm: 1127.5273
    diff_norm / base_norm: 0.0365
    """
