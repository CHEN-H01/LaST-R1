export PYTHONPATH=/mnt/cpfs/chenhao/last-r1:/mnt/cpfs/chenhao/last-r1/transformers:$PYTHONPATH
export PATH=/root/miniconda3/envs/qwen-rl-2_5_1/bin:$PATH
source /root/miniconda3/bin/activate qwen-rl-2_5_1
cd /mnt/cpfs/chenhao/last-r1

export NCCL_DEBUG=WARN
export WANDB_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

BASE_DIR="/mnt/cpfs/chenhao/exps"
PROJECT_NAME='qwen3-vla-oft-latent-rl-lapo'

EXPERIMENT_NAME='qwen3-vla-oft-rl-lapo-latent-ar-libero'

EXPERIMENT_DIR="${BASE_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
MODEL_SAVE_DIR="${EXPERIMENT_DIR}/model_save"
OUTPUTS_DIR="${EXPERIMENT_DIR}/outputs"
ROLLOUT_DIR="${EXPERIMENT_DIR}/rollouts"
WANDB_DIR="${EXPERIMENT_DIR}/wandb"

mkdir -p "$MODEL_SAVE_DIR" "$OUTPUTS_DIR" "$ROLLOUT_DIR" "$WANDB_DIR"
export EXPERIMENT_DIR MODEL_SAVE_DIR OUTPUTS_DIR
export WANDB_DIR
export ROLLOUT_DIR


SFT_MODEL_PATH="/path/to/warmup_models"  #  your warmup model path
DATASET_NAME="libero_spatial"  # libero_10, libero_spatial, libero_object, libero_goal
VLA_NAME="qwen-oft"
DATA_STATUS="/path/to/dataset_statistics.json"  # Uses training-set action/state 1st/99th percentiles (q01/q99 in JSON) for normalization; replace with your statistics path

NUM_GPUS=8
NUM_NODES=1
ALIGN_PATH="/path/to/align.json"  #  your align.json path

VAL_BEFORE_TRAIN=True
VAL_ONLY=True

NEED_TO_SUB=3
TEMPERATURE=1.6
LR=3e-5
VALUE_HEAD_LR=3e-4
WEIGHT_DECAY=0
REWARD_COEF=5
GRAD_CLIP=10
center_crop=True
max_prompt_length=95   # 95 for libero_spatial, 95 for libero_object, 95 for libero_goal, 100 for libero_10
image_size=256

value_choice='latent_end'
bootstrap='none'
attn_mode='causal'
use_latent=True
use_latent_loss=True
input_mode='ids'
latent_mode='ar'
latent_length=8
latent_bind=0
latent_end_num=4
latent_loss_weight=0.1
action_loss_weight=1.0
latent_end_loss_weight=1.0
latent_group_mode='first'
end_do_sample=True

libero_spatial_max_steps=240
libero_object_max_steps=320
libero_goal_max_steps=320
libero_10_max_steps=576

val_micro_batch_size=16
val_batch_size=500  # If you use 8 GPUs for training, change it to 496; if you use 4 GPUs for training, change it to 500.
train_batch_size=512
n_samples=1
micro_batch_size=16
traj_mini_batch_size=15   # 15 for libero_spatial, 20 for libero_object, 20 for libero_goal, 18 for libero_10

clip_ratio_high=0.28
clip_ratio_low=0.2
accuracy_lower_bound=0
accuracy_upper_bound=1

SAVE_FREQ=25
TEST_FREQ=5
TOTAL_EPOCHS=500
LR_HALF_STEP=-1

GRAD_OFFLOAD=True
OPTIMIZER_OFFLOAD=True

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    hydra.run.dir=$OUTPUTS_DIR \
    verifier.reward_coef=$REWARD_COEF \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=$n_samples \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=$accuracy_lower_bound \
    data.accuracy_upper_bound=$accuracy_upper_bound \
    data.oversample_factor=1 \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.value_head_lr=$VALUE_HEAD_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((train_batch_size * n_samples / 4)) \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=$GRAD_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=$traj_mini_batch_size \
    actor_rollout_ref.actor.action_token_len=7 \
    actor_rollout_ref.actor.action_chunks_len=8 \
    actor_rollout_ref.actor.value_choice=$value_choice \
    actor_rollout_ref.actor.attn_mode=$attn_mode \
    actor_rollout_ref.actor.use_latent=$use_latent \
    actor_rollout_ref.actor.use_latent_loss=$use_latent_loss \
    actor_rollout_ref.actor.latent_loss_weight=$latent_loss_weight \
    actor_rollout_ref.actor.action_loss_weight=$action_loss_weight \
    actor_rollout_ref.actor.latent_end_loss_weight=$latent_end_loss_weight \
    actor_rollout_ref.actor.latent_length=$latent_length \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.actor.latent_mode=$latent_mode \
    actor_rollout_ref.actor.latent_end_num=$latent_end_num \
    actor_rollout_ref.actor.latent_bind=$latent_bind \
    actor_rollout_ref.actor.input_mode=$input_mode \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=$val_micro_batch_size \
    actor_rollout_ref.rollout.image_size=$image_size \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=$micro_batch_size \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.libero_spatial_max_steps=$libero_spatial_max_steps \
    actor_rollout_ref.rollout.libero_object_max_steps=$libero_object_max_steps \
    actor_rollout_ref.rollout.libero_goal_max_steps=$libero_goal_max_steps \
    actor_rollout_ref.rollout.libero_10_max_steps=$libero_10_max_steps \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=$center_crop \
    actor_rollout_ref.rollout.action_token_len=7 \
    actor_rollout_ref.rollout.action_chunks_len=8 \
    actor_rollout_ref.rollout.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.data_status=$DATA_STATUS \
    actor_rollout_ref.rollout.need_to_sub=$NEED_TO_SUB \
    actor_rollout_ref.rollout.value_choice=$value_choice \
    actor_rollout_ref.rollout.bootstrap=$bootstrap \
    actor_rollout_ref.rollout.attn_mode=$attn_mode \
    actor_rollout_ref.rollout.use_latent=$use_latent \
    actor_rollout_ref.rollout.latent_length=$latent_length \
    actor_rollout_ref.rollout.latent_end_num=$latent_end_num \
    actor_rollout_ref.rollout.latent_bind=$latent_bind \
    actor_rollout_ref.rollout.input_mode=$input_mode \
    actor_rollout_ref.rollout.latent_mode=$latent_mode \
    actor_rollout_ref.rollout.latent_group_mode=$latent_group_mode \
    actor_rollout_ref.rollout.end_do_sample=$end_do_sample \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$MODEL_SAVE_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_only=$VAL_ONLY \
    algorithm.adv_estimator=gae \
    algorithm.lam=0.95 \
    algorithm.gamma=0.99 \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.lr_half_step=$LR_HALF_STEP \

