# ADDITION and MODULAR ADDITION(grouped by num-steps so that per-GPU we have similar numbers)

# GPU 1: addition & modular addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 1 --savefile results --seed 12345 --num-steps 500 --cooldown-steps 50
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 2 --savefile results --seed 23456 --num-steps 1000 --cooldown-steps 100
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 3 --savefile results --seed 23456 --num-steps 1000 --cooldown-steps 100
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 1 --savefile results --seed 34567 --num-steps 2000 --cooldown-steps 200
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 2 --savefile results --seed 45678 --num-steps 5000 --cooldown-steps 500
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 1 --savefile results --seed 12345 --num-steps 500 --cooldown-steps 50 --mod 23
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 2 --savefile results --seed 23456 --num-steps 1000 --cooldown-steps 100 --mod 23
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 2 --max-tokens-per-num 3 --savefile results --seed 23456 --num-steps 1000 --cooldown-steps 100 --mod 23
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 1 --savefile results --seed 34567 --num-steps 2000 --cooldown-steps 200 --mod 23
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 2 --savefile results --seed 45678 --num-steps 5000 --cooldown-steps 500 --mod 23

# GPU 2: addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 3 --savefile results --seed 45678 --num-steps 10000 --cooldown-steps 1000
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 1 --savefile results --seed 56789 --num-steps 10000 --cooldown-steps 1000

# GPU 3: modular addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 3 --max-tokens-per-num 3 --savefile results --seed 45678 --num-steps 10000 --cooldown-steps 1000 --mod 23
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 1 --savefile results --seed 56789 --num-steps 10000 --cooldown-steps 1000 --mod 23

# GPU 4: addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 2 --savefile results --seed 67890 --num-steps 20000 --cooldown-steps 2000

# GPU 5: modular addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 2 --savefile results --seed 67890 --num-steps 20000 --cooldown-steps 2000 --mod 23

# GPU 6: addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 3 --savefile results --seed 67890 --num-steps 40000 --cooldown-steps 4000

# GPU 7: modular addition
uv run main.py --weight-decay 0.001 --num-runs 5 --regenerate-dataset-every-run --n-layer 6 --n-head 2 --n-embd 256 --use-wandb --max-digits-per-token 4 --max-tokens-per-num 3 --savefile results --seed 67890 --num-steps 40000 --cooldown-steps 4000 --mod 23
