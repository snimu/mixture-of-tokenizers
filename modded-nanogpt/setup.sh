git clone https://github.com/snimu/modded-nanogpt.git
cd modded-nanogpt
git switch mot
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py

torchrun --standalone --nproc_per_node=8 0-4_train_gpt_medium.py
torchrun --standalone --nproc_per_node=8 71061_mot-in_toks-valemb.py
torchrun --standalone --nproc_per_node=8 71062_mot-in_toks-valemb.py
torchrun --standalone --nproc_per_node=8 71063_mot-in_toks-valemb.py
torchrun --standalone --nproc_per_node=8 71064_mot-in_toks-valemb.py
torchrun --standalone --nproc_per_node=8 71065_mot-in_toks-valemb.py
torchrun --standalone --nproc_per_node=8 71066_mot-in_toks-valemb.py
