
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

    results = 0.0
    for i in range(len(numels)):
        results += norms[i] * numels[i]

    return results / sum(numels)


if __name__ == "__main__":
    base_model = load_model("base")
    instruct_model = load_model("instruct")

    base_norm = measure_weight_norm(base_model)
    instruct_norm = measure_weight_norm(instruct_model)

    diff_model = instruct_model.copy()
    for param_base, param_instruct in zip(base_model.parameters(), instruct_model.parameters()):
        param_instruct.data = param_instruct.data - param_base.data

    diff_norm = measure_weight_norm(diff_model)

    print(f"Base model weight norm: {base_norm:.4f}")
    print(f"Instruct model weight norm: {instruct_norm:.4f}")
    print(f"Instruct model weight norm: {diff_norm:.4f}")
    print(f"diff_norm / base_norm: {diff_norm / base_norm:.4f}")
