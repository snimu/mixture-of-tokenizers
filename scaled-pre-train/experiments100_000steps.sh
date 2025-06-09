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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 51_463_168
# num_params_byte_embs: 0
# num_params_mixin: 0
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 205_852_672
# num_params_total: 454_496_326

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 12_865_792
# num_params_byte_embs: 21_984
# num_params_mixin: 1_048_576
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 167_277_280
# num_params_total: 416_969_510


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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 38_597_376
# num_params_byte_embs: 29_312
# num_params_mixin: 1_835_008
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 193_016_192
# num_params_total: 443_494_854

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 38_597_376
# num_params_byte_embs: 58_624
# num_params_mixin: 2_883_584
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 193_045_504
# num_params_total: 444_572_742

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 45_030_272
# num_params_byte_embs: 29_312
# num_params_mixin: 1_966_080
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 199_449_088
# num_params_total: 450_058_822

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 45_030_272
# num_params_byte_embs: 58_624
# num_params_mixin: 3_014_656
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 199_478_400
# num_params_total: 451_136_710

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 51_463_168
# num_params_byte_embs: 29_312
# num_params_mixin: 2_097_152
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 205_881_984
# num_params_total: 456_622_790

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 51_463_168
# num_params_byte_embs: 58_624
# num_params_mixin: 3_145_728
# num_params_mixout: 0
# num_params_lm_head: 51_511_296
# num_params_embs_total: 205_911_296
# num_params_total: 457_700_678

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 12_865_792
# num_params_byte_embs: 21_984
# num_params_mixin: 1_048_576
# num_params_mixout: 0
# num_params_lm_head: 32_768
# num_params_embs_total: 167_277_280
# num_params_total: 365_490_982

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
# num_params_blocks: 197_132_350
# num_params_ve: 154_389_504
# num_params_token_embs: 12_865_792
# num_params_byte_embs: 21_984
# num_params_mixin: 1_048_576
# num_params_mixout: 0
# num_params_lm_head: 524_288
# num_params_embs_total: 167_277_280
# num_params_total: 365_982_502
