# Inference

Set up:

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt


Script contains basic run_inference(model_version, input_text) function.

Valid models:
- "one_residual"
- "two_residual"
- "no_residual"

Loads models from https://huggingface.co/nickcdryan
