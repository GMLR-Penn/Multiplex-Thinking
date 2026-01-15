#!/bin/bash
set -x

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --enable_soft_think)
            ENABLE_SOFT_THINK="$2"
            shift 2
            ;;
        --enable_mixed_rollout)
            ENABLE_MIXED_ROLLOUT="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_token_len_per_gpu)
            MAX_TOKEN_LEN_PER_GPU="$2"
            shift 2
            ;;
        --multiplex_width)
            MULTIPLEX_WIDTH="$2"
            shift 2
            ;;
        --loss_mode)
            LOSS_MODE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --temp)
            TEMP="$2"
            shift 2
            ;;
        --early_stopping_entropy_threshold)
            EARLY_STOPPING_ENTROPY_THRESHOLD="$2"
            shift 2
            ;;
        --early_stopping_length_threshold)
            EARLY_STOPPING_LENGTH_THRESHOLD="$2"
            shift 2
            ;;
        --enable_entropy_mask)
            ENABLE_ENTROPY_MASK="$2"
            shift 2
            ;;
        --entropy_mask_threshold)
            ENTROPY_MASK_THRESHOLD="$2"
            shift 2
            ;;
        --gpu_mem_util)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --enable_gumbel)
            ENABLE_GUMBEL="$2"
            shift 2
            ;;
        --gumbel_tau)
            GUMBEL_TAU="$2"
            shift 2
            ;;
        --n_gpus_per_node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --after_thinking_temperature)
            AFTER_THINKING_TEMPERATURE="$2"
            shift 2
            ;;
        --after_thinking_top_k)
            AFTER_THINKING_TOP_K="$2"
            shift 2
            ;;
        --after_thinking_min_p)
            AFTER_THINKING_MIN_P="$2"
            shift 2
            ;;
        --enable_replacement)
            ENABLE_REPLACEMENT="$2"
            shift 2
            ;;
        --enable_gumbel_after_thinking)
            ENABLE_GUMBEL_AFTER_THINKING="$2"
            shift 2
            ;;
        --after_thinking_top_p)
            AFTER_THINKING_TOP_P="$2"
            shift 2
            ;;
        --val_before_train)
            VAL_BEFORE_TRAIN="$2"
            shift 2
            ;;
        --total_training_steps)
            TOTAL_TRAINING_STEPS="$2"
            shift 2
            ;;
        --resume_from_path)
            RESUME_FROM_PATH="$2"
            shift 2
            ;;
        --resume_mode)
            RESUME_MODE="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --test_freq)
            TEST_FREQ="$2"
            shift 2
            ;;
        --val_rollout_n)
            VAL_ROLLOUT_N="$2"
            shift 2
            ;;
        --val_dataset)
            VAL_DATASET="$2"
            shift 2
            ;;
        --val_batch_size)
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --enable_unweighting)
            ENABLE_UNWEIGHTING="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default values if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="DeepSeek/DeepSeek-R1-Distill-Qwen-32B"
fi

if [ -z "$EXP_NAME" ]; then
    EXP_NAME="multiplex-thinking"
fi

if [ -z "$ENABLE_SOFT_THINK" ]; then
    ENABLE_SOFT_THINK=True
fi 

if [ -z "$ENABLE_MIXED_ROLLOUT" ]; then
    ENABLE_MIXED_ROLLOUT=False
fi

if [ -z "$TRAIN_BATCH_SIZE" ]; then
    TRAIN_BATCH_SIZE=256
fi

if [ -z "$MAX_TOKEN_LEN_PER_GPU" ]; then
    MAX_TOKEN_LEN_PER_GPU=65536
fi

if [ -z "$MULTIPLEX_WIDTH" ]; then
    MULTIPLEX_WIDTH=3
fi

if [ -z "$LOSS_MODE" ]; then
    LOSS_MODE="vanilla"
fi 

if [ -z "$TOP_P" ]; then
    TOP_P=1.0
fi

if [ -z "$TEMP" ]; then
    TEMP=1.0
fi

if [ -z "$EARLY_STOPPING_ENTROPY_THRESHOLD" ]; then
    EARLY_STOPPING_ENTROPY_THRESHOLD=-1.0
fi

if [ -z "$EARLY_STOPPING_LENGTH_THRESHOLD" ]; then
    EARLY_STOPPING_LENGTH_THRESHOLD=256
fi

if [ -z "$ENABLE_ENTROPY_MASK" ]; then
    ENABLE_ENTROPY_MASK=False
fi

if [ -z "$ENTROPY_MASK_THRESHOLD" ]; then
    ENTROPY_MASK_THRESHOLD=0.0
fi

if [ -z "$GPU_MEMORY_UTILIZATION" ]; then
    GPU_MEMORY_UTILIZATION=0.9
fi



if [ -z "$ENABLE_GUMBEL" ]; then
    ENABLE_GUMBEL=False
fi
if [ -z "$ROLLOUT_ENABLE_GUMBEL" ]; then
    ROLLOUT_ENABLE_GUMBEL=$ENABLE_GUMBEL
fi
if [ -z "$GUMBEL_TAU" ]; then
    GUMBEL_TAU=1.0
fi

if [ -z "$N_GPUS_PER_NODE" ]; then
    N_GPUS_PER_NODE=8
fi

if [ -z "$AFTER_THINKING_TEMPERATURE" ]; then
    AFTER_THINKING_TEMPERATURE=1.0
fi

if [ -z "$ENABLE_REPLACEMENT" ]; then
    ENABLE_REPLACEMENT=True
fi

if [ -z "$ENABLE_GUMBEL_AFTER_THINKING" ]; then
    ENABLE_GUMBEL_AFTER_THINKING=False
fi

if [ -z "$VAL_BEFORE_TRAIN" ]; then
    VAL_BEFORE_TRAIN=True
fi

if [ -z "$TOTAL_TRAINING_STEPS" ]; then
    TOTAL_TRAINING_STEPS=300
fi

if [ -z "$RESUME_FROM_PATH" ]; then
    RESUME_FROM_PATH=null
fi

if [ -z "$RESUME_MODE" ]; then
    RESUME_MODE="auto"
fi

if [ -z "$MAX_RESPONSE_LENGTH" ]; then
    MAX_RESPONSE_LENGTH=8192
fi

if [ -z "$SAVE_FREQ" ]; then
    SAVE_FREQ=25
fi

if [ -z "$TEST_FREQ" ]; then
    TEST_FREQ=25
fi

if [ -z "$VAL_ROLLOUT_N" ]; then
    VAL_ROLLOUT_N=4
fi

if [ -z "$VAL_DATASET" ]; then
    VAL_DATASET=aime
fi

if [ -z "$VAL_BATCH_SIZE" ]; then
    VAL_BATCH_SIZE=512
fi

if [ -z "$WANDB_PROJECT" ]; then
    WANDB_PROJECT=MultiplexThinning
fi

if [ -z "$AFTER_THINKING_TOP_P" ]; then
    AFTER_THINKING_TOP_P=1.0
fi

if [ -z "$AFTER_THINKING_TOP_K" ]; then
    AFTER_THINKING_TOP_K=-1
fi

if [ -z "$AFTER_THINKING_MIN_P" ]; then
    AFTER_THINKING_MIN_P=0.0
fi

if [ -z "$ENABLE_UNWEIGHTING" ]; then
    ENABLE_UNWEIGHTING=False
fi

export TOKENIZERS_PARALLELISM=true
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$WANDB_API_KEY" ]; then
    echo "[WARN] WANDB_API_KEY is empty. Set it via env var if you want wandb logging."
fi
if [ -z "$HF_TOKEN" ]; then
    echo "[WARN] HF_TOKEN is empty. Set it via env var if you need to access gated/private HuggingFace models."
fi
export NCCL_TIMEOUT=36000
export NCCL_SOCKET_IFNAME=ibp24s0
export NCCL_IB_HCA=mlx5_4
############## ray_node_setup.sh ##############
echo ${MASTER_ADDR}
echo $OMPI_COMM_WORLD_RANK


export NCCL_TIMEOUT=72000
# Force GPU cache cleanup (helps reduce OOM flakiness between runs)
export CUDA_LAUNCH_BLOCKING=1
# Avoid multi-process GPU contention
export CUDA_DEVICE_ORDER=PCI_BUS_ID




unset RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES

# Provide default values for local execution if not set by a job scheduler
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-0}

# if OMPI_COMM_WORLD_RANK is 0, then start the ray cluster, else print the value of MASTER_ADDR
if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
    # Start Ray head node
    ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $N_GPUS_PER_NODE
else
    echo ${MASTER_ADDR}
    ray start --address ${MASTER_ADDR}:6379  --num-gpus $N_GPUS_PER_NODE
fi



sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=deepscaler/hdfs_data/train.parquet \
    data.val_files=deepscaler/hdfs_data/$VAL_DATASET.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.return_raw_chat=True \
    data.truncation=right \
    +data.use_online_transform=False \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=$TEMP \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP \
    actor_rollout_ref.rollout.enable_soft_thinking=$ENABLE_SOFT_THINK \
    actor_rollout_ref.rollout.enable_mixed_rollout=$ENABLE_MIXED_ROLLOUT \
    actor_rollout_ref.rollout.after_thinking_top_p=$AFTER_THINKING_TOP_P \
    actor_rollout_ref.rollout.after_thinking_top_k=$AFTER_THINKING_TOP_K \
    actor_rollout_ref.rollout.after_thinking_min_p=$AFTER_THINKING_MIN_P \
    actor_rollout_ref.rollout.max_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.used_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.early_stopping_entropy_threshold=$EARLY_STOPPING_ENTROPY_THRESHOLD \
    actor_rollout_ref.rollout.early_stopping_length_threshold=$EARLY_STOPPING_LENGTH_THRESHOLD \
    actor_rollout_ref.rollout.enable_entropy_mask=$ENABLE_ENTROPY_MASK \
    actor_rollout_ref.rollout.entropy_mask_threshold=$ENTROPY_MASK_THRESHOLD \
    actor_rollout_ref.rollout.enable_gumbel=$ENABLE_GUMBEL \
    actor_rollout_ref.rollout.gumbel_tau=$GUMBEL_TAU \
    actor_rollout_ref.rollout.after_thinking_temperature=$AFTER_THINKING_TEMPERATURE \
    actor_rollout_ref.rollout.enable_replacement=$ENABLE_REPLACEMENT \
    actor_rollout_ref.rollout.enable_gumbel_after_thinking=$ENABLE_GUMBEL_AFTER_THINKING \
    actor_rollout_ref.rollout.enable_unweighting=$ENABLE_UNWEIGHTING \
    actor_rollout_ref.rollout.val_kwargs.enable_replacement=$ENABLE_REPLACEMENT \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel_after_thinking=$ENABLE_GUMBEL_AFTER_THINKING \
    actor_rollout_ref.rollout.val_kwargs.enable_unweighting=$ENABLE_UNWEIGHTING \
    actor_rollout_ref.rollout.val_kwargs.enable_gumbel=$ENABLE_GUMBEL \
    actor_rollout_ref.rollout.val_kwargs.gumbel_tau=$GUMBEL_TAU \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_temperature=$AFTER_THINKING_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_entropy_threshold=$EARLY_STOPPING_ENTROPY_THRESHOLD \
    actor_rollout_ref.rollout.val_kwargs.early_stopping_length_threshold=$EARLY_STOPPING_LENGTH_THRESHOLD \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_p=$AFTER_THINKING_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_top_k=$AFTER_THINKING_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.after_thinking_min_p=$AFTER_THINKING_MIN_P \
    actor_rollout_ref.rollout.val_kwargs.max_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.used_topk=$MULTIPLEX_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path=$RESUME_FROM_PATH \
    reward_model.reward_manager=hf_math_verify \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_sleep_hack=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    reward_model.enable=False \
    trainer.default_local_dir=./${WANDB_PROJECT}/${EXP_NAME}

