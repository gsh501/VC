#!/bin/bash
# Script to start TensorBoard for monitoring training

# 参数
MODEL_NAME="DMC_slf_yuv420_lsui"
QUALITY_LEVEL=3
TENSORBOARD_DIR="./pretrained/${MODEL_NAME}/${QUALITY_LEVEL}/tensorboard"
PORT=6006

echo "Starting TensorBoard..."
echo "Model: ${MODEL_NAME}"
echo "Quality Level: ${QUALITY_LEVEL}"
echo "Log Directory: ${TENSORBOARD_DIR}"
echo "Port: ${PORT}"
echo ""
echo "Open in browser: http://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# 启动TensorBoard
tensorboard --logdir=${TENSORBOARD_DIR} --port=${PORT} --bind_all

