##########################################################
####################### 2025-05-01 #######################
##########################################################

# Run on commit b860cdc (b860cdc6fa5dfc12bb3c53c6d2747998f6cbc979)

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
    --pull-in True \
    --add-padded-and-pulled False \
    --use-byte-self-attn False \
    --model-dim 1024 \
    --byte-dim 32 \
    --token-dim 512 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

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
    --pull-in True \
    --add-padded-and-pulled False \
    --use-byte-self-attn False \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

# 1.3 byte-dim = token-dim; byted-dim*bpt + token-dim < model-dim
#     Let's see how it goes if this thing is byte-dominant
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
    --pull-in True \
    --add-padded-and-pulled False \
    --use-byte-self-attn False \
    --model-dim 1024 \
    --byte-dim 60 \
    --token-dim 60 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests