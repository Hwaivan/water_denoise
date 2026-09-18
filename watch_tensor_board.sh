#!/usr/bin/env bash

LOG_DIR="/data/huayifan/water_denoise/logs"
LOG_FILE="/tmp/tensorboard.log"
PORT=6006

if pgrep -f "tensorboard.*--port ${PORT}" >/dev/null; then
    echo "TensorBoard已经在端口${PORT}运行。"
    exit 0
fi

nohup /opt/conda/bin/tensorboard \
    --logdir "${LOG_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    > "${LOG_FILE}" 2>&1 &

echo "TensorBoard已启动，PID=$!"
echo "日志文件：${LOG_FILE}"