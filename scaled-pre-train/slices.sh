# A util to copy&paste the slice creation into different terminals
# First, you need to add "export HF_TOKEN=<your-token>" to your .bashrc

# There are 91,608 total training batches -> go in steps of 1000

uv run data_creation.py --from-batch 0 --to-batch 1000 --skip-fw-val-batches
uv run data_creation.py --from-batch 1000 --to-batch 2000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 2000 --to-batch 3000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 3000 --to-batch 4000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 4000 --to-batch 5000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 5000 --to-batch 6000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 6000 --to-batch 7000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 7000 --to-batch 8000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 8000 --to-batch 9000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 9000 --to-batch 10000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 10000 --to-batch 11000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 11000 --to-batch 12000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 12000 --to-batch 13000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 13000 --to-batch 14000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 14000 --to-batch 15000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 15000 --to-batch 16000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 16000 --to-batch 17000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 17000 --to-batch 18000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 18000 --to-batch 19000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 19000 --to-batch 20000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 20000 --to-batch 21000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 21000 --to-batch 22000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 22000 --to-batch 23000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 23000 --to-batch 24000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 24000 --to-batch 25000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 25000 --to-batch 26000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 26000 --to-batch 27000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 27000 --to-batch 28000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 28000 --to-batch 29000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 29000 --to-batch 30000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 30000 --to-batch 31000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 31000 --to-batch 32000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 32000 --to-batch 33000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 33000 --to-batch 34000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 34000 --to-batch 35000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 35000 --to-batch 36000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 36000 --to-batch 37000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 37000 --to-batch 38000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 38000 --to-batch 39000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 39000 --to-batch 40000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 40000 --to-batch 41000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 41000 --to-batch 42000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 42000 --to-batch 43000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 43000 --to-batch 44000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 44000 --to-batch 45000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 45000 --to-batch 46000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 46000 --to-batch 47000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 47000 --to-batch 48000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 48000 --to-batch 49000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 49000 --to-batch 50000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 50000 --to-batch 51000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 51000 --to-batch 52000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 52000 --to-batch 53000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 53000 --to-batch 54000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 54000 --to-batch 55000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 55000 --to-batch 56000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 56000 --to-batch 57000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 57000 --to-batch 58000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 58000 --to-batch 59000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 59000 --to-batch 60000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 60000 --to-batch 61000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 61000 --to-batch 62000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 62000 --to-batch 63000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 63000 --to-batch 64000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 64000 --to-batch 65000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 65000 --to-batch 66000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 66000 --to-batch 67000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 67000 --to-batch 68000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 68000 --to-batch 69000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 69000 --to-batch 70000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 70000 --to-batch 71000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 71000 --to-batch 72000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 72000 --to-batch 73000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 73000 --to-batch 74000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 74000 --to-batch 75000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 75000 --to-batch 76000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 76000 --to-batch 77000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 77000 --to-batch 78000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 78000 --to-batch 79000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 79000 --to-batch 80000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 80000 --to-batch 81000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 81000 --to-batch 82000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 82000 --to-batch 83000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 83000 --to-batch 84000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 84000 --to-batch 85000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 85000 --to-batch 86000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 86000 --to-batch 87000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 87000 --to-batch 88000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 88000 --to-batch 89000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 89000 --to-batch 90000 --skip-fm-val-batches --skip-fw-val-batches

uv run data_creation.py --from-batch 90000 --to-batch 91000 --skip-fm-val-batches --skip-fw-val-batches
uv run data_creation.py --from-batch 91000 --skip-fm-val-batches
