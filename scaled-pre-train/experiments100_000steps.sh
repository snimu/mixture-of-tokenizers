# 0. Baseline: Token-Token

torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method noop \
    --byte-mixout-method noop \
    --model-dim 1024 \
    --token-dim 1024 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

#    Output tokens to make them comparable

# 1.1 byte-dim*bpt + token-dim = model-dim
#     But bytes get a larger embedding, so the token dim is smaller
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests


# 1.2 slightly larger byte- and token-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 64 \
    --token-dim 768 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.3 Same, but with larger byte-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 128 \
    --token-dim 768 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.4 First byte-dim but larger token-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 64 \
    --token-dim 896 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.5 Larger byte- and token-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 128 \
    --token-dim 896 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.6 token-dim like in baseline, small byte-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 64 \
    --token-dim 1024 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.7 token-dim like in baseline, large byte-dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 128 \
    --token-dim 1024 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 2. Compare different mixout methods

# 2.1 Split with 0 self-attn layers
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method split \
    --n-layer-out 0 \
    --padding-in left \
    --padding-out right \
    --pull-in \
    --pull-out \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests

# 2.2 copy with 1 self-attn layer
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 1000 \
    --save-checkpoint-every 1000 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method copy \
    --n-layer-out 1 \
    --padding-in left \
    --padding-out right \
    --pull-in \
    --pull-out \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 773322 \
    --wandb-project MoT-scaled-pre-train-tests
