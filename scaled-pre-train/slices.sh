# A util to copy&paste the slice creation into different terminals
# First, you need to add "export HF_TOKEN=<your-token>" to your .bashrc

# There are 91,608 total training batches -> go in steps of 1000

# Machine 1
uv run data_creation.py --from-batch 0 --to-batch 200 --skip-fw-val-batches --tokenize
uv run data_creation.py --from-batch 200 --to-batch 400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 400 --to-batch 600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 600 --to-batch 800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 800 --to-batch 1000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 1000 --to-batch 1200 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 1200 --to-batch 1400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 1400 --to-batch 1600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 1600 --to-batch 1800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 1800 --to-batch 2000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 2
uv run data_creation.py --from-batch 2000 --to-batch 2200 --skip-fm-val-batches --skip-fw-val-batches --tokenize
uv run data_creation.py --from-batch 2200 --to-batch 2400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 2400 --to-batch 2600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 2600 --to-batch 2800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 2800 --to-batch 3000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 3000 --to-batch 3200 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 3200 --to-batch 3400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 3400 --to-batch 3600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 3600 --to-batch 3800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 3800 --to-batch 4000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 3
uv run data_creation.py --from-batch 4000 --to-batch 4200 --skip-fm-val-batches --skip-fw-val-batches --tokenize
uv run data_creation.py --from-batch 4200 --to-batch 4400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 4400 --to-batch 4600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 4600 --to-batch 4800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 4800 --to-batch 5000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 5000 --to-batch 5200 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 5200 --to-batch 5400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 5400 --to-batch 5600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 5600 --to-batch 5800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 5800 --to-batch 6000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 4
uv run data_creation.py --from-batch 6000 --to-batch 6200 --skip-fm-val-batches --skip-fw-val-batches --tokenize
uv run data_creation.py --from-batch 6200 --to-batch 6400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 6400 --to-batch 6600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 6600 --to-batch 6800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 6800 --to-batch 7000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 7000 --to-batch 7200 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 7200 --to-batch 7400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 7400 --to-batch 7600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 7600 --to-batch 7800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 7800 --to-batch 8000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 5
uv run data_creation.py --from-batch 8000 --to-batch 8200 --skip-fm-val-batches --skip-fw-val-batches --tokenize
uv run data_creation.py --from-batch 8200 --to-batch 8400 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 8400 --to-batch 8600 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 8600 --to-batch 8800 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 8800 --to-batch 9000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 6
uv run data_creation.py --from-batch 9000 --skip-fm-val-batches
