#!/bin/bash

GPUS_PER_NODE=8
NUM_MACHINES=1
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=${MASTER_PORT:-26005}  # 支持通过环境变量设置，默认26005
MASTER_ADDR=${MASTER_ADDR:-"localhost"}  # 支持通过环境变量设置，默认localhost

RANK=$1

# 简单的端口检测函数（可选，如果端口被占用会报错，建议使用 train.py 的自动端口分配功能）
check_port() {
    local host=$1
    local port=$2
    if command -v nc >/dev/null 2>&1; then
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "警告: 端口 $host:$port 可能已被占用"
            echo "建议: 使用 scripts/multi_node/bagel/train.py 脚本，它支持自动端口分配"
            return 1
        fi
    fi
    return 0
}

# 可选：检测端口（如果 nc 命令可用）
check_port "$MASTER_ADDR" "$MASTER_PORT" || true

accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_machines ${NUM_MACHINES} --num_processes ${NUM_PROCESSES} \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_bagel.py \
    --config config/grpo.py:pickscore_bagel\
