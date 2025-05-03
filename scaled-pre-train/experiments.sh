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

    # "num_params": 454_496_326,
    # "final_val_loss_fw": 4.626010894775391,
    # "min_val_loss_fw": 4.626010894775391,
    # "final_val_loss_fm": 6.2828545570373535,
    # "min_val_loss_fm": 6.079019546508789,
    # "step_avg_train_time": 469.9990721980459

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

    # "num_params": 429_827_974,
    # "final_val_loss_fw": 4.838685035705566,
    # "min_val_loss_fw": 4.838685035705566,
    # "final_val_loss_fm": 5.895299911499023,
    # "min_val_loss_fm": 5.868696212768555,
    # "step_avg_train_time": 489.14060482174233

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

    # "num_params": 416_969_510,
    # "final_val_loss_fw": 4.8055739402771,
    # "min_val_loss_fw": 4.8055739402771,
    # "final_val_loss_fm": 5.56953763961792,
    # "min_val_loss_fm": 5.527792930603027,
    # "step_avg_train_time": 552.0098968119224

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

    # "num_params": 410_540_278,
    # "final_val_loss_fw": 4.843713760375977,
    # "min_val_loss_fw": 4.843713760375977,
    # "final_val_loss_fm": 5.643980026245117,
    # "min_val_loss_fm": 5.643980026245117,
    # "step_avg_train_time": 548.2767846237273

# CONCLUSION 1: --byte-dim 48 --token-dim 256

# 2. Pull-in vs. no-pull-in vs. add-padded-and-pulled

# 2.1 pull-in
# see above

# 2.2 no-pull-in
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
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

    # "num_params": 416_969_510,
    # "final_val_loss_fw": 4.9417877197265625,
    # "min_val_loss_fw": 4.9417877197265625,
    # "final_val_loss_fm": 6.242247104644775,
    # "min_val_loss_fm": 6.239513874053955,
    # "step_avg_train_time": 481.2802737029271

# 2.3 add-padded-and-pulled
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
    --add-padded-and-pulled \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

    # "num_params": 416_969_510,
    # "final_val_loss_fw": 4.878345489501953,
    # "min_val_loss_fw": 4.878345489501953,
    # "final_val_loss_fm": 5.819784641265869,
    # "min_val_loss_fm": 5.819784641265869,
    # "step_avg_train_time": 549.3409705544187

# CONCLUSION 2: Use pull-in but not add-padded-and-pulled

# 3. Compare different mixout methods

# 3.1 Split with 0 self-attn layers
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
    --byte-mixout-method split \
    --n-layer-out 0 \
    --padding-in left \
    --padding-out right \
    --pull-in \
    --pull-out \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

        # "num_params": 365_490_982,
        # "final_val_loss_fw": 3.0287742614746094,
        # "min_val_loss_fw": 3.0287742614746094,
        # "final_val_loss_fm": 4.808889865875244,
        # "min_val_loss_fm": 4.657312393188477,
        # "step_avg_train_time": 670.2071643664054

# 3.2 split with 1 self-attn layer
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
    --byte-mixout-method split \
    --n-layer-out 1 \
    --padding-in left \
    --padding-out right \
    --pull-in \
    --pull-out \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

        # "num_params": 365_490_982,
        # "final_val_loss_fw": 3.0277862548828125,
        # "min_val_loss_fw": 3.0277862548828125,
        # "final_val_loss_fm": 4.789813995361328,
        # "min_val_loss_fm": 4.712076187133789,
        # "step_avg_train_time": 516.550470089167

# 3.3 copy with 1 self-attn layer
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
    --byte-mixout-method copy \
    --n-layer-out 1 \
    --padding-in left \
    --padding-out right \
    --pull-in \
    --pull-out \
    --model-dim 1024 \
    --byte-dim 48 \
    --token-dim 256 \
    --seed 12345 \
    --wandb-project MoT-scaled-pre-train-tests

    # "num_params": 365_982_502,
    # "final_val_loss_fw": 3.1354775428771973,
    # "min_val_loss_fw": 3.1354775428771973,
    # "final_val_loss_fm": 4.467406272888184,
    # "min_val_loss_fm": 4.204345226287842,
    # "step_avg_train_time": 492.57973214851444
