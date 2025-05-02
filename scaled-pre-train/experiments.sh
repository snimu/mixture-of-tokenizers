# 0. Baseline: Token-Token

torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 10 \
    --save-checkpoint-every 0 \
    --bytes-per-token 16 \
    --byte-mixin-method noop \
    --byte-mixout-method noop \
    --model-dim 1024 \
    --token-dim 1024 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

# Number of parameters: 454_496_326

# 1. Check some variations of byte-mixin-method=concat: different sizes of byte vs token embeddings
#    Output tokens to make them comparable

# 1.1 byte-dim*bpt + token-dim = model-dim
#     Make half the model dim the byte dim, half the token dim
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 10 \
    --save-checkpoint-every 0 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 32 \
    --token-dim 512 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

# Number of parameters: 429_827_974

# 1.2 byte-dim*bpt + token-dim = model-dim
#     But bytes get a larger embedding, so the token dim is smaller
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 10 \
    --save-checkpoint-every 0 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

# Number of parameters: 416_969_510


# 1.3 slightly larger byte embs
torchrun --nproc_per_node=8 train_gpt.py \
    --num-iterations 100 \
    --cooldown-frac 0.6 \
    --seq-len 1024 \
    --batch-size-train 64 \
    --batch-size-val 32 \
    --val-loss-every 10 \
    --save-checkpoint-every 0 \
    --bytes-per-token 16 \
    --byte-mixin-method concat \
    --byte-mixout-method noop \
    --padding-in left \
    --pull-in \
    --model-dim 1024 \
    --byte-dim 56 \
    --token-dim 128 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-test

# Number of parameters: 410_540_278