# 0. Baseline: Token-Token

torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method noop \
    --byte-mixout-method noop \
    --model-dim 1024 \
    --token-dim 1024 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

#    Output tokens to make them comparable

# 1.1 byte-dim*bpt + token-dim = model-dim
#     But bytes get a larger embedding, so the token dim is smaller
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.2 fairly large token-dim (but still smaller than model-dim); decently large byte-dim
#     Let's see if this causes an OOM error or not.
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 64 \
    --token-dim 512 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.3 Same as 1.1, but with attn on the bytes at the input
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --use-byte-self-attn \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.4 Same as 1.3, but with bidirectional attn between the bytes of a single token
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --use-byte-self-attn \
    --mix-bytes-within-tok-in \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.3 Cross-attn
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method cross_attn \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 1024 \
    --token-dim 1024 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 2. Pull-in vs. no-pull-in vs. add-padded-and-pulled

# 2.1 pull-in

# 2.2 add-padded-and-pulled
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --add-padded-and-pulled \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 3. Compare different mixout methods

# 3.1 Split with 0 self-attn layers
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
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
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests

# 3.2 copy with 1 self-attn layer
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 10000 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 250 \
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
    --seed 9090333 \
    --save-checkpoint-every 500 \
    --wandb-project MoT-scaled-pre-train-tests
