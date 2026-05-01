# 显示帮助信息
show_help() {
    echo "使用方法: $0 --ckpt_dir <ckpt_dir>"
    echo ""
    echo "必需参数:"
    echo "  --ckpt_dir        初始权重存储位置"
    echo "可选参数:"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --ckpt_dir ''"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "CKPT_DIR: ${CKPT_DIR}"

# 从ckpt_path生成output_dir
# 如果ckpt_dir为"results/checkpoints/run_mahjong_ichihime_1014_nebula_steps3001_save100_log10_lr2e-5/0000100/model"
# 则save_path为"results/checkpoints/run_mahjong_ichihime_1014_nebula_steps3001_save100_log10_lr2e-5/0000100/model.safetensors"
SAVE_PATH="${CKPT_DIR}.safetensors"

echo "SAVE_PATH: ${SAVE_PATH}"

#!/bin/bash

# 显示帮助信息
show_help() {
    echo "使用方法: $0 --ckpt_dir <ckpt_dir>"
    echo ""
    echo "必需参数:"
    echo "  --ckpt_dir        初始权重存储位置"
    echo "可选参数:"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --ckpt_dir ''"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数是否提供
if [ -z "$CKPT_DIR" ]; then
    echo "错误: 必须提供 --ckpt_dir 参数"
    show_help
    exit 1
fi

echo "CKPT_DIR: ${CKPT_DIR}"

# 从ckpt_path生成output_dir
SAVE_PATH="${CKPT_DIR}.safetensors"

echo "SAVE_PATH: ${SAVE_PATH}"

# 获取父目录（包含 model, optimizer*, scheduler.pt, data_status 的目录）
PARENT_DIR=$(dirname "$CKPT_DIR")

# 检查 SAVE_PATH 是否已存在
if [ -f "$SAVE_PATH" ]; then
    echo "⚠️  注意: $SAVE_PATH 已存在，跳过模型处理脚本..."
else
    # 执行模型处理脚本
    echo "🚀 开始执行模型转换..."
    python ckpt_processing/ckpt_processing_model.py \
        --ckpt_dir "${CKPT_DIR}" \
        --save_path "${SAVE_PATH}"

    # 检查脚本是否成功执行
    if [ $? -ne 0 ]; then
        echo "❌ 错误：模型处理脚本执行失败，跳过清理步骤。"
        exit 1
    fi
    echo "✅ 模型转换完成。"
fi

# 清理阶段 —— 只有当 SAVE_PATH 存在时才进行清理（说明转换成功或已存在）
if [ -f "$SAVE_PATH" ]; then
    echo "🧹 开始清理临时文件..."

    # 删除 data_status 目录
    if [ -d "$PARENT_DIR/data_status" ]; then
        rm -rf "$PARENT_DIR/data_status"
        echo "已删除: $PARENT_DIR/data_status"
    fi

    # 删除所有 optimizer.*.pt 文件
    for opt_file in "$PARENT_DIR"/optimizer.*.pt; do
        if [ -f "$opt_file" ]; then
            rm -f "$opt_file"
            echo "已删除: $opt_file"
        fi
    done

    # 删除 scheduler.pt 文件
    if [ -f "$PARENT_DIR/scheduler.pt" ]; then
        rm -f "$PARENT_DIR/scheduler.pt"
        echo "已删除: $PARENT_DIR/scheduler.pt"
    fi

    # 删除 model 文件夹（仅当 .safetensors 存在时才删除，确保模型已安全转换）
    if [ -d "$PARENT_DIR/model" ]; then
        rm -rf "$PARENT_DIR/model"
        echo "已删除: $PARENT_DIR/model"
    fi

    echo "🎉 清理完成。最终保留文件: $SAVE_PATH"
else
    echo "❌ 警告: $SAVE_PATH 不存在，未执行任何清理操作。"
    exit 1
fi