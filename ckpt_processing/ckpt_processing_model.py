import argparse
import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.format_utils as dist_cp_format_utils
from safetensors.torch import save_file, load_file
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, help='Checkpoint directory path')
    parser.add_argument('--save_path', required=True, help='Output safetensors file path')
    args = parser.parse_args()

    print(f"输入检查点目录: {args.ckpt_dir}")
    print(f"输出文件路径: {args.save_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(args.ckpt_dir):
        print(f"错误: 输入目录不存在: {args.ckpt_dir}")
        return
    
    # 修复：创建保存文件的父目录，而不是将save_path当作目录
    save_dir = os.path.dirname(args.save_path)
    print(f"创建目录: {save_dir}")
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"目录已创建: {save_dir}")
    
    print("正在加载检查点...")
    state_dict = {}
    dist_cp_format_utils._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(args.ckpt_dir),
        planner=dist_cp_format_utils._EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    
    print(f"加载的state_dict键数量: {len(state_dict.keys())}")
    # if len(state_dict.keys()) > 0:
    #     print(f"state_dict键: {list(state_dict.keys())}")
    
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]
        print("提取了嵌套的state_dict")
    
    print(f"最终state_dict张量数量: {len(state_dict)}")
    
    # 修复meta device问题：确保所有张量都在CPU上
    print("检查并修复meta device张量...")
    meta_device_count = 0
    fixed_state_dict = {}
    
    for key, tensor in state_dict.items():
        if hasattr(tensor, 'device') and str(tensor.device) == 'meta':
            meta_device_count += 1
            print(f"发现meta device张量: {key}, shape: {tensor.shape}, dtype: {tensor.dtype}")
            # 创建一个在CPU上的零张量来替换meta device张量
            fixed_tensor = torch.zeros_like(tensor, device='cpu', dtype=tensor.dtype)
            fixed_state_dict[key] = fixed_tensor
            print(f"已修复张量 {key} 到CPU设备")
        else:
            # 确保张量在CPU上
            if hasattr(tensor, 'device') and tensor.device != torch.device('cpu'):
                fixed_state_dict[key] = tensor.cpu()
            else:
                fixed_state_dict[key] = tensor
    
    if meta_device_count > 0:
        print(f"警告: 发现并修复了 {meta_device_count} 个meta device张量")
        print("注意: meta device张量已被零张量替换，这可能影响模型性能")
    else:
        print("未发现meta device张量")
    
    print(f"保存到: {args.save_path}")
    save_file(fixed_state_dict, args.save_path)
    
    # 验证文件是否保存成功
    if os.path.exists(args.save_path):
        file_size = os.path.getsize(args.save_path)
        print(f"文件保存成功! 大小: {file_size / (1024*1024):.2f} MB")
    else:
        print("错误: 文件保存失败!")

if __name__ == "__main__":
    main()