# A util to copy&paste the slice creation into different terminals
# First, you need to add "export HF_TOKEN=<your-token>" to your .bashrc

# There are 91,608 total training batches -> go in steps of 1000

# Machine 1
uv run data_creation.py --from-batch 0 --to-batch 2000 --skip-fw-val-batches --tokenize --start-upload-loop
uv run data_creation.py --from-batch 2000 --to-batch 4000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 4000 --to-batch 6000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 6000 --to-batch 8000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 8000 --to-batch 10000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 10000 --to-batch 12000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 12000 --to-batch 14000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 14000 --to-batch 16000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 16000 --to-batch 18000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 18000 --to-batch 20000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 2
uv run data_creation.py --from-batch 20000 --to-batch 22000 --skip-fm-val-batches --skip-fw-val-batches --tokenize --start-upload-loop
uv run data_creation.py --from-batch 22000 --to-batch 24000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 24000 --to-batch 26000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 26000 --to-batch 28000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 28000 --to-batch 30000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 30000 --to-batch 32000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 32000 --to-batch 34000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 34000 --to-batch 36000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 36000 --to-batch 38000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 38000 --to-batch 40000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 3
uv run data_creation.py --from-batch 40000 --to-batch 42000 --skip-fm-val-batches --skip-fw-val-batches --tokenize --start-upload-loop
uv run data_creation.py --from-batch 42000 --to-batch 44000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 44000 --to-batch 46000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 46000 --to-batch 48000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 48000 --to-batch 50000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 50000 --to-batch 52000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 52000 --to-batch 54000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 54000 --to-batch 56000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 56000 --to-batch 58000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 58000 --to-batch 60000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 4
uv run data_creation.py --from-batch 60000 --to-batch 62000 --skip-fm-val-batches --skip-fw-val-batches --tokenize --start-upload-loop
uv run data_creation.py --from-batch 62000 --to-batch 64000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 64000 --to-batch 66000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 66000 --to-batch 68000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 68000 --to-batch 70000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 70000 --to-batch 72000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 72000 --to-batch 74000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 74000 --to-batch 76000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 76000 --to-batch 78000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 78000 --to-batch 80000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 5
uv run data_creation.py --from-batch 80000 --to-batch 82000 --skip-fm-val-batches --skip-fw-val-batches --tokenize --start-upload-loop
uv run data_creation.py --from-batch 82000 --to-batch 84000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 84000 --to-batch 86000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 86000 --to-batch 88000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 88000 --to-batch 90000 --skip-fm-val-batches --skip-fw-val-batches

# Machine 6
uv run data_creation.py --from-batch 90000 --skip-fm-val-batches --tokenize --start-upload-loop
