# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
BAGEL 模型推理接口
封装了生成、编辑和理解功能的统一接口
"""

import os
import random
import numpy as np
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer


class BagelInference:
    """
    BAGEL 模型推理类，提供统一的接口进行图像生成、编辑和理解
    
    主要功能：
    1. generation: 纯文本到图像生成
    2. generation_with_thinking: 带思考过程的文本到图像生成
    3. editing: 图像编辑
    4. editing_with_thinking: 带思考过程的图像编辑
    5. understanding: 图像理解
    """
    
    def __init__(
        self, 
        model_path: str = "models/BAGEL-7B-MoT",
        checkpoint_path: Optional[str] = None,
        vit_checkpoint_path: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        seed: int = 42
    ):
        """
        初始化BAGEL推理器
        
        Args:
            model_path: 模型路径
            checkpoint_path: 检查点路径，如果为None则使用model_path下的ema.safetensors
            vit_checkpoint_path: ViT权重的检查点路径，用于加载visual_und=False训练的模型缺失的ViT部分
                               如果为None，则不额外加载ViT权重
            max_mem_per_gpu: 每个GPU的最大内存使用量
            seed: 随机种子
        """
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path or os.path.join(model_path, "ema.safetensors")
        self.vit_checkpoint_path = vit_checkpoint_path
        self.max_mem_per_gpu = max_mem_per_gpu
        
        # 设置随机种子
        self._set_seed(seed)
        
        # 初始化模型组件
        self._init_configs()
        self._init_models()
        self._load_model()
        self._init_inferencer()
        
        print("BAGEL推理器初始化完成")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_configs(self):
        """初始化配置"""
        # LLM配置
        self.llm_config = Qwen2Config.from_json_file(os.path.join(self.model_path, "llm_config.json"))
        self.llm_config.qk_norm = True
        self.llm_config.tie_word_embeddings = False
        self.llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        # ViT配置
        self.vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_path, "vit_config.json"))
        self.vit_config.rope = False
        self.vit_config.num_hidden_layers = self.vit_config.num_hidden_layers - 1
        
        # VAE配置
        self.vae_model, self.vae_config = load_ae(local_path=os.path.join(self.model_path, "ae.safetensors"))
        
        # BAGEL配置
        self.config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=self.llm_config,
            vit_config=self.vit_config,
            vae_config=self.vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
    
    def _init_models(self):
        """初始化模型"""
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(self.llm_config)
            vit_model = SiglipVisionModel(self.vit_config)
            self.model = Bagel(language_model, vit_model, self.config)
            self.model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(self.vit_config, meta=True)
        
        # 初始化分词器
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        
        # 初始化图像变换
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)
    
    def _fix_state_dict_keys(self, state_dict):
        """
        修复state dict的key名称，处理FSDP训练时的命名差异
        
        FSDP训练时保存的格式: model.layers.X... / model.embed_tokens... / lm_head...
        Bagel推理时期望的格式: language_model.model.layers.X... / language_model.model.embed_tokens... / language_model.lm_head...
        """
        fixed_state_dict = {}
        key_mapping_stats = {
            'language_model': 0,
            'vit_model': 0,
            'connector': 0,
            'other': 0
        }
        
        for key, value in state_dict.items():
            new_key = key
            
            # 处理language model的权重
            if key.startswith('model.'):
                # model.layers.X... -> language_model.model.layers.X...
                # model.embed_tokens... -> language_model.model.embed_tokens...
                # model.norm... -> language_model.model.norm...
                new_key = 'language_model.' + key
                key_mapping_stats['language_model'] += 1
            elif key.startswith('lm_head.'):
                # lm_head... -> language_model.lm_head...
                new_key = 'language_model.' + key
                key_mapping_stats['language_model'] += 1
            elif key.startswith('language_model.'):
                key_mapping_stats['language_model'] += 1
            elif key.startswith('vit_model.'):
                key_mapping_stats['vit_model'] += 1
            elif key.startswith('connector.') or key == 'vit_pos_embed':
                key_mapping_stats['connector'] += 1
            else:
                # time_embedder, latent_pos_embed, vae2llm, llm2vae等
                key_mapping_stats['other'] += 1
            
            fixed_state_dict[new_key] = value
        
        print(f"  - language_model相关: {key_mapping_stats['language_model']} 个")
        print(f"  - vit_model相关: {key_mapping_stats['vit_model']} 个")
        print(f"  - connector相关: {key_mapping_stats['connector']} 个")
        print(f"  - 其他模块: {key_mapping_stats['other']} 个")
        
        return fixed_state_dict
    
    def _load_model(self):
        """加载模型权重"""
        # 如果指定了vit_checkpoint_path，需要先合并权重
        missing_gen_modules = []  # 初始化变量
        if self.vit_checkpoint_path is not None:
            print(f"检测到vit_checkpoint_path，正在合并权重...")
            print(f"主checkpoint: {self.checkpoint_path}")
            print(f"ViT checkpoint: {self.vit_checkpoint_path}")
            
            # 加载主checkpoint
            main_state_dict = load_file(self.checkpoint_path)
            print(f"主checkpoint包含 {len(main_state_dict)} 个权重")
            
            # 检查图像生成核心模块是否存在
            gen_modules = ['time_embedder', 'vae2llm', 'llm2vae', 'latent_pos_embed']
            for module_name in gen_modules:
                has_module = any(k.startswith(module_name) for k in main_state_dict.keys())
                if not has_module:
                    missing_gen_modules.append(module_name)
            
            if missing_gen_modules:
                print(f"警告: 主checkpoint缺少图像生成模块: {missing_gen_modules}")
                print(f"这些模块对图像生成至关重要！")
            
            # 加载ViT checkpoint
            vit_state_dict = load_file(self.vit_checkpoint_path)
            
            # 检查是否需要从ViT checkpoint补充图像生成模块
            gen_modules_to_add = []
            if missing_gen_modules:
                print(f"尝试从ViT checkpoint补充缺失的图像生成模块...")
                for module_name in missing_gen_modules:
                    has_in_vit = any(k.startswith(module_name) for k in vit_state_dict.keys())
                    if has_in_vit:
                        gen_modules_to_add.append(module_name + '.')
                        print(f"  找到 {module_name} 在ViT checkpoint中")
            
            # 过滤出ViT相关的权重（以及缺失的图像生成模块）
            vit_keys = ['vit_model.', 'connector.', 'vit_pos_embed'] + gen_modules_to_add
            vit_weights = {k: v for k, v in vit_state_dict.items() 
                          if any(k.startswith(prefix) for prefix in vit_keys)}
            
            if len(vit_weights) > 0:
                print(f"从ViT checkpoint提取了 {len(vit_weights)} 个ViT相关权重")
                # 合并权重（ViT权重会覆盖主checkpoint中的同名权重，如果有的话）
                main_state_dict.update(vit_weights)
                print(f"合并后共 {len(main_state_dict)} 个权重")
            else:
                print(f"警告: 在 {self.vit_checkpoint_path} 中未找到ViT相关权重")
            
            # 修复key名称（处理FSDP保存的命名差异）
            print("修复权重key名称...")
            main_state_dict = self._fix_state_dict_keys(main_state_dict)
            print(f"修复后共 {len(main_state_dict)} 个权重")
            
            # 打印一些示例key用于调试
            sample_keys = list(main_state_dict.keys())[:5]
            print(f"示例key（前5个）: {sample_keys}")
            
            # 保存合并后的checkpoint到临时文件
            import tempfile
            temp_checkpoint = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
            temp_checkpoint_path = temp_checkpoint.name
            temp_checkpoint.close()
            
            from safetensors.torch import save_file
            save_file(main_state_dict, temp_checkpoint_path)
            print(f"已保存合并后的checkpoint到临时文件: {temp_checkpoint_path}")
            
            checkpoint_to_load = temp_checkpoint_path
            cleanup_temp = True
        else:
            checkpoint_to_load = self.checkpoint_path
            cleanup_temp = False
        
        # 计算device map
        device_map = infer_auto_device_map(
            self.model,
            max_memory={i: self.max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        
        # 确保相关模块在同一设备上
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]
        
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        
        # 加载合并后的checkpoint
        print(f"开始加载checkpoint到模型...")
        
        # 先检查模型需要哪些权重
        print("检查模型参数...")
        model_keys = set(dict(self.model.named_parameters()).keys())
        print(f"模型需要 {len(model_keys)} 个参数")
        
        # 检查checkpoint有哪些权重
        if cleanup_temp:
            checkpoint_keys = set(load_file(checkpoint_to_load).keys())
        else:
            checkpoint_keys = set(load_file(self.checkpoint_path).keys())
        print(f"Checkpoint包含 {len(checkpoint_keys)} 个权重")
        
        # 找出缺失的权重
        missing_in_checkpoint = model_keys - checkpoint_keys
        if missing_in_checkpoint:
            print(f"警告: 有 {len(missing_in_checkpoint)} 个参数在checkpoint中缺失")
            # 打印前10个缺失的参数
            for i, key in enumerate(list(missing_in_checkpoint)[:10]):
                print(f"  缺失: {key}")
            if len(missing_in_checkpoint) > 10:
                print(f"  ... 还有 {len(missing_in_checkpoint) - 10} 个")
        
        self.model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=checkpoint_to_load,
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        # self.model = load_checkpoint_and_dispatch(
        #     self.model,
        #     checkpoint=checkpoint_to_load,
        #     device_map=device_map,
        #     dtype=torch.bfloat16
        # )
        print(f"Checkpoint加载完成")
        
        # 清理临时文件
        if cleanup_temp:
            try:
                os.remove(checkpoint_to_load)
                print(f"已清理临时文件: {checkpoint_to_load}")
            except:
                pass
        
        self.model = self.model.eval()
    
    def _init_inferencer(self):
        """初始化推理器"""
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
    
    def generation(
        self,
        text: str,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        **kwargs
    ) -> Dict[str, Any]:
        """
        纯文本到图像生成
        
        Args:
            text: 输入文本提示
            cfg_text_scale: 文本引导强度，控制模型对文本提示的遵循程度
            cfg_img_scale: 图像引导强度（对纯生成任务通常为1.0）
            cfg_interval: CFG应用的时间步区间
            timestep_shift: 时间步偏移，影响去噪过程的分布
            num_timesteps: 去噪步数
            cfg_renorm_min: CFG重归一化最小值
            cfg_renorm_type: CFG重归一化类型 ("global", "channel", "text_channel")
            
        Returns:
            包含生成图像的字典 {'image': PIL.Image}
        """
        inference_params = {
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        return self.inferencer(text=text, **inference_params)
    
    def generation_with_thinking(
        self,
        text: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        **kwargs
    ) -> Dict[str, Any]:
        """
        带思考过程的文本到图像生成
        
        Args:
            text: 输入文本提示
            max_think_token_n: 最大思考token数量
            do_sample: 是否使用采样生成思考过程
            其他参数同generation函数
            
        Returns:
            包含生成图像和思考过程的字典 {'image': PIL.Image, 'text': str}
        """
        inference_params = {
            'max_think_token_n': max_think_token_n,
            'do_sample': do_sample,
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        return self.inferencer(text=text, think=True, **inference_params)
    
    def editing(
        self,
        image: Union[str, Image.Image],
        text: str,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = [0.0, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        **kwargs
    ) -> Dict[str, Any]:
        """
        图像编辑
        
        Args:
            image: 输入图像，可以是PIL.Image对象或图像路径
            text: 编辑指令文本
            cfg_text_scale: 文本引导强度
            cfg_img_scale: 图像引导强度，控制对原图的保持程度
            cfg_interval: CFG应用的时间步区间，编辑通常使用[0.0, 1.0]
            cfg_renorm_type: 编辑推荐使用"text_channel"
            其他参数同generation函数
            
        Returns:
            包含编辑后图像的字典 {'image': PIL.Image}
        """
        # 处理图像输入
        if isinstance(image, str):
            image = Image.open(image)
        
        inference_params = {
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        return self.inferencer(image=image, text=text, **inference_params)
    
    def editing_with_thinking(
        self,
        image: Union[str, Image.Image],
        text: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 2.0,
        cfg_interval: List[float] = [0.0, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "text_channel",
        **kwargs
    ) -> Dict[str, Any]:
        """
        带思考过程的图像编辑
        
        Args:
            image: 输入图像，可以是PIL.Image对象或图像路径
            text: 编辑指令文本
            max_think_token_n: 最大思考token数量
            do_sample: 是否使用采样生成思考过程
            其他参数同editing函数
            
        Returns:
            包含编辑后图像和思考过程的字典 {'image': PIL.Image, 'text': str}
        """
        # 处理图像输入
        if isinstance(image, str):
            image = Image.open(image)
        
        inference_params = {
            'max_think_token_n': max_think_token_n,
            'do_sample': do_sample,
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        return self.inferencer(image=image, text=text, think=True, **inference_params)
    
    def understanding(
        self,
        image: Optional[Union[str, Image.Image]] = None,
        text: str = "",
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        图像理解
        
        Args:
            image: 输入图像，可以是PIL.Image对象或图像路径；若为None则仅文本理解
            text: 问题或指令文本
            max_think_token_n: 最大思考token数量
            do_sample: 是否使用采样生成回答
            
        Returns:
            包含理解结果的字典 {'text': str}
        """
        # 处理图像输入（可选）
        if isinstance(image, str):
            image = Image.open(image)
        
        inference_params = {
            'max_think_token_n': max_think_token_n,
            'do_sample': do_sample,
            **kwargs
        }
        
        if image is None:
            return self.inferencer(
                text=text,
                understanding_output=True,
                **inference_params
            )
        else:
            return self.inferencer(
                image=image,
                text=text,
                understanding_output=True,
                **inference_params
            )


# 便捷函数，用于快速创建推理器实例
def create_bagel_inference(
    model_path: str = "models/BAGEL-7B-MoT",
    checkpoint_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    **kwargs
) -> BagelInference:
    """
    创建BAGEL推理器实例的便捷函数
    
    Args:
        model_path: 模型路径
        checkpoint_path: 检查点路径
        vit_checkpoint_path: ViT权重的检查点路径（可选）
        **kwargs: 其他初始化参数
        
    Returns:
        BagelInference实例
    """
    return BagelInference(
        model_path=model_path, 
        checkpoint_path=checkpoint_path, 
        vit_checkpoint_path=vit_checkpoint_path,
        **kwargs
    )


if __name__ == "__main__":
    # 使用示例
    print("BAGEL推理接口使用示例")
    print("=" * 50)
    
    model_path = "models/BAGEL-7B-MoT"

    # 需要从另一个完整的checkpoint加载ViT权重
    vit_checkpoint_path = "models/BAGEL-7B-MoT/ema.safetensors"
    checkpoint_path = "results/checkpoints/mahjong_ichihime_extension_1010_t2i_vlm/0000300/model.safetensors"

    # 创建推理器
    inference = create_bagel_inference(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        vit_checkpoint_path=vit_checkpoint_path  # 如果checkpoint包含完整权重，设为None即可
    )

    if not os.path.exists("test_images/outputs"):
        os.makedirs("test_images/outputs")
    
    # 示例1: 纯文本生成
    result = inference.generation("一只可爱的小猫坐在花园里")
    result['image'].save("test_images/outputs/cat.png")
    print("生成图像已保存")
    
    # 示例2: 带思考的生成
    result = inference.generation_with_thinking("一辆由小汽车组成的大汽车")
    print(f"思考过程: {result['text']}")
    result['image'].save("test_images/outputs/car.png")
    
    # 示例5: 图像理解
    result = inference.understanding("test_images/meme.jpg", "这个梗图有什么好笑的？")
    print(f"理解结果: {result['text']}")
    
    print("请根据实际需要取消注释相应的代码进行测试")
