#!/bin/bash
# Training script for LSUI dataset with distributed training

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际可用GPU数量调整

# 分布式训练参数
NUM_GPUS=4  # GPU数量
MASTER_PORT=29500  # 主节点端口

# 训练参数
DATASET_ROOT="/home/admin1/Data/water_enhance/LSUI"
MODEL_NAME="DMC_slf_yuv420_lsui"
QUALITY_LEVEL=3
BATCH_SIZE=1
FRAME_COUNT=3  # Each image is replicated 3 times to form a GOP
TEST_GOP=3     # Test with 3-frame GOPs
PATCH_SIZE="256 256"
LEARNING_RATE=1e-4
EPOCHS=180

# 模型路径
I_FRAME_MODEL="checkpoints/cvpr2025_image.pth.tar"
P_FRAME_MODEL="pretrained/DMC_slf_yuv420/1/checkpoint_vd.pth.tar"

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    train_vd_phase_3.py \
    --model ${MODEL_NAME} \
    --dataset_root ${DATASET_ROOT} \
    --quality-level ${QUALITY_LEVEL} \
    --batch-size ${BATCH_SIZE} \
    --frame-count ${FRAME_COUNT} \
    --test-gop ${TEST_GOP} \
    --patch-size ${PATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --epochs ${EPOCHS} \
    --model_path_i ${I_FRAME_MODEL} \
    --checkpoint ${P_FRAME_MODEL} \
    --save \
    --num-workers 4 \
    --clip_max_norm 1.0

echo "Training completed!"

