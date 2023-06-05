

export CODE='/home/s20235025/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py'
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_ID="/home/s20235025/tobigs-pokemon/dataset"
export BUILD_MODEL_DIR="/home/s20235025/tobigs-pokemon/build"
# export CHECKPOINT="/home/s20235025/final_output_interpolation_1step/checkpoint-26000"
export BATCH_SIZE=16
export TRAIN_EPOCH=100

accelerate launch --config_file="/home/s20235025/.cache/huggingface/accelerate/uncondi.yaml" \
  $CODE \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_ID \
 --use_ema \
 --train_batch_size=$BATCH_SIZE --gradient_accumulation_steps=4 --gradient_checkpointing \
 --num_train_epochs=$TRAIN_EPOCH \
 --checkpointing_steps=1000 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 \
 --output_dir=$BUILD_MODEL_DIR \

## build
accelerate launch --config_file="/home/s20235025/.cache/huggingface/accelerate/uncondi.yaml" \
   $CODE \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_ID \
 --use_ema \
 --train_batch_size=$BATCH_SIZE --gradient_accumulation_steps=4 --gradient_checkpointing \
 --num_train_epochs=$TRAIN_EPOCH \
 --checkpointing_steps=5000 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 \
 --output_dir=$BUILD_MODEL_DIR \
 --resume_from_checkpoint=$CHECKPOINT
