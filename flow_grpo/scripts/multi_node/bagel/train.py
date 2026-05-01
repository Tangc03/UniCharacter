#!/usr/bin/env python3
"""
Python脚本，等同于执行 bash scripts/multi_node/bagel/main.sh 0
接受 GPUS_PER_NODE 和 MASTER_PORT 作为命令行参数
支持自动端口检测和分配，避免端口占用问题
"""

import argparse
import subprocess
import sys
import socket


def is_port_in_use(host: str, port: int) -> bool:
    """检查指定 host:port 是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False  # 绑定成功，端口可用
        except OSError:
            return True  # 绑定失败，端口被占用


def get_truly_free_port(host: str = "localhost") -> int:
    """获取一个真正可用的端口，由系统自动分配"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))  # 绑定到端口0，让系统自动分配
        _, port = s.getsockname()
        return port


def find_free_port(host: str, base_port: int, max_tries: int = 50) -> int:
    """从 base_port 开始向上寻找可用端口"""
    port = base_port
    for _ in range(max_tries):
        if not is_port_in_use(host, port):
            return port
        port += 1
    raise RuntimeError(
        f"在 {host} 上从端口 {base_port} 开始连续尝试 {max_tries} 个端口均失败，请手动指定一个空闲端口"
    )


def main():
    parser = argparse.ArgumentParser(
        description="运行多节点训练脚本，等同于 main.sh 0"
    )
    parser.add_argument(
        "--GPUS_PER_NODE",
        type=int,
        required=True,
        help="每个节点的GPU数量"
    )
    parser.add_argument(
        "--NUM_MACHINES",
        type=int,
        default=1,
        help="节点数量，默认1"
    )
    parser.add_argument(
        "--MASTER_PORT",
        type=int,
        default=0,
        help="主进程端口号。设置为0时自动分配可用端口，或指定具体端口号（如果被占用会自动寻找可用端口）"
    )
    parser.add_argument(
        "--MASTER_ADDR",
        type=str,
        default="localhost",
        help="主节点IP地址，单节点使用localhost，多节点使用主节点IP"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/grpo.py:pickscore_bagel",
        help="训练配置，例如: config/grpo.py:personalized_t2i_bagel 或 config/grpo.py:personalized_t2i_bagel_lora"
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        default=None,
        help="YAML配置文件名称（可选），例如: personalized_t2i_bagel_vqa_only.yaml。也可以通过环境变量 PERSONALIZED_T2I_BAGEL_CONFIG 指定"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="自定义checkpoint路径（.safetensors文件），如果为None则使用默认的ema.safetensors。可以是绝对路径或相对于模型目录的路径"
    )
    
    args = parser.parse_args()
    
    # 设置参数
    GPUS_PER_NODE = args.GPUS_PER_NODE
    NUM_MACHINES = args.NUM_MACHINES
    NUM_PROCESSES = NUM_MACHINES * GPUS_PER_NODE
    MASTER_ADDR = args.MASTER_ADDR
    RANK = 0
    
    # 端口处理：支持自动分配（端口0）或手动指定
    if args.MASTER_PORT == 0:
        # 自动分配一个可用端口
        MASTER_PORT = get_truly_free_port(MASTER_ADDR)
        print("=" * 80)
        print("使用自动端口分配模式")
        print(f"系统自动分配端口: {MASTER_PORT}")
        print("如需固定端口，请使用 --MASTER_PORT 指定具体端口号")
        print("=" * 80)
    else:
        # 检查指定端口是否可用
        original_port = args.MASTER_PORT
        print(f"正在检测端口 {MASTER_ADDR}:{original_port} 是否可用...")
        
        if is_port_in_use(MASTER_ADDR, original_port):
            print(f"端口 {MASTER_ADDR}:{original_port} 已被占用，正在寻找可用端口...")
            try:
                MASTER_PORT = find_free_port(MASTER_ADDR, original_port + 1)
                print("=" * 80)
                print(f"检测到端口被占用: {MASTER_ADDR}:{original_port}")
                print(f"自动切换到可用端口: {MASTER_ADDR}:{MASTER_PORT}")
                print("如果需要固定端口，请显式指定一个当前未被占用的 --MASTER_PORT。")
                print("=" * 80)
            except RuntimeError as e:
                print("=" * 80)
                print("错误: 自动寻找可用端口失败")
                print(str(e))
                print("=" * 80)
                sys.exit(1)
        else:
            MASTER_PORT = original_port
            print(f"端口 {MASTER_ADDR}:{MASTER_PORT} 可用")
    
    # 设置环境变量确保PyTorch底层也使用正确的端口
    import os
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["WANDB_API_KEY"] = "<wandb_api_key>"
    print(f"已设置环境变量 MASTER_PORT={MASTER_PORT}")
    
    # 处理 YAML 配置文件
    if args.yaml_config:
        os.environ["PERSONALIZED_T2I_BAGEL_CONFIG"] = args.yaml_config
        print(f"已设置环境变量 PERSONALIZED_T2I_BAGEL_CONFIG={args.yaml_config}")
    elif "PERSONALIZED_T2I_BAGEL_CONFIG" in os.environ:
        print(f"使用环境变量 PERSONALIZED_T2I_BAGEL_CONFIG={os.environ['PERSONALIZED_T2I_BAGEL_CONFIG']}")
    
    # 构建 accelerate launch 命令
    if GPUS_PER_NODE == 8:
        cmd = [
            "accelerate", "launch",
            "--config_file", "scripts/accelerate_configs/fsdp.yaml",
            # "--num_machines", str(NUM_MACHINES),
            "--num_processes", str(NUM_PROCESSES),
            # "--machine_rank", str(RANK),
            "--main_process_ip", MASTER_ADDR,
            "--main_process_port", str(MASTER_PORT),
            "scripts/train_bagel.py",
            "--config", args.config
        ]
    elif GPUS_PER_NODE == 16:
        cmd = [
            "accelerate", "launch",
            "--config_file", "scripts/accelerate_configs/fsdp_16gpu.yaml",
            # "--num_machines", str(NUM_MACHINES),
            "--num_processes", str(NUM_PROCESSES),
            # "--machine_rank", str(RANK),
            "--main_process_ip", MASTER_ADDR,
            "--main_process_port", str(MASTER_PORT),
            "scripts/train_bagel.py",
            "--config", args.config
        ]
    
    # 如果提供了checkpoint_path（且不是字符串"None"），添加到命令中
    if args.checkpoint_path and args.checkpoint_path.lower() != "none":
        cmd.extend(["--checkpoint_path", args.checkpoint_path])
        print(f"使用自定义checkpoint路径: {args.checkpoint_path}")
    
    # 执行命令
    print(f"执行命令: {' '.join(cmd)}")
    print(f"参数: GPUS_PER_NODE={GPUS_PER_NODE}, MASTER_PORT={MASTER_PORT}, RANK={RANK}")
    
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，退出码: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("错误: 找不到 accelerate 命令，请确保已安装 accelerate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

