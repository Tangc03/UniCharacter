# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
个性化BAGEL推理接口 (GRPO版本)
扩展了基础推理功能，支持角色对话和个性化图像生成
基于 inference_grpo.py，支持 vit_checkpoint_path 参数
"""

import re
import os
from typing import Dict, List, Optional, Union, Any
from PIL import Image

from inference_grpo import BagelInference


# 思考系统提示
THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then generate the image.\n"
    "The planning process must be enclosed within <think></think>.\n"
    "Then provide a concise, standardized generation instruction enclosed within <gen></gen> that can be fed to a text-to-image model.\n"
    "The <gen> instruction should be a single descriptive sentence or a short list of comma-separated attributes, focusing on character, pose/expression, scene context, composition, style, and quality words.\n"
    "Output format (no additional text):\n<think>...your planning process...</think>\n<gen>...your generation instruction...</gen>\n"
)


class PersonalizedBagelInference(BagelInference):
    """
    个性化BAGEL推理类 (GRPO版本)，支持角色对话和个性化图像生成
    继承自 inference_grpo.BagelInference，支持 vit_checkpoint_path 参数
    
    主要功能：
    1. personalized_response_without_thinking: 无思考过程的角色对话+图像生成
    2. personalized_response_with_thinking: 带思考过程的角色对话+图像生成
    """
    
    def __init__(self, *args, **kwargs):
        """初始化个性化推理器"""
        super().__init__(*args, **kwargs)
        print("个性化BAGEL推理器初始化完成 (GRPO版本)")
    
    def extract_thinking_and_instruction(self, text: str) -> Dict[str, str]:
        """
        从文本中提取思考过程和生成指令
        
        Args:
            text: 包含<think>和<gen>标签的文本
            
        Returns:
            包含thinking_process和generation_instruction的字典
        """
        # 提取<think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        thinking_process = think_match.group(1).strip() if think_match else ""
        
        # 提取<gen>...</gen>
        gen_match = re.search(r'<gen>(.*?)</gen>', text, re.DOTALL)
        generation_instruction = gen_match.group(1).strip() if gen_match else ""
        
        return {
            "thinking_process": thinking_process,
            "generation_instruction": generation_instruction
        }
    
    def personalized_response_without_thinking(
        self,
        character_name: str,
        description: str,
        opening: str,
        user_text: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        # 文本生成参数
        max_response_tokens: int = 500,
        do_sample_response: bool = True,
        text_temperature: float = 0.7,
        # 图像生成参数
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
        无思考过程的个性化对话和图像生成
        
        流程：
        1. 生成角色回复
        2. 根据回复生成对应图像
        
        Args:
            character_name: 角色名称
            description: 角色描述
            opening: 角色开场白
            user_text: 用户输入
            reference_image: 参考图像（可选）
            conversation_history: 对话历史（可选）
            max_response_tokens: 最大回复token数
            do_sample_response: 是否采样生成回复
            text_temperature: 文本生成温度
            其他图像生成参数同generation函数
            
        Returns:
            包含角色回复和生成图像的字典 {'response': str, 'image': PIL.Image}
        """
        # 构建对话历史文本
        history_text = ""
        if conversation_history:
            for turn in conversation_history:
                role = turn.get('role', 'unknown')
                content = turn.get('text', '')
                history_text += f"{role}: {content}\n"
        
        # Step 1: 生成角色回复
        response_prompt = (
            f"Character Description: {description}\n\n"
            f"Opening: {opening}\n\n"
        )
        
        if history_text:
            response_prompt += f"Conversation History:\n{history_text}\n"
        
        response_prompt += f"User Input: {user_text}\n\n"
        response_prompt += (
            f"Based on the character description, opening, conversation history, "
            f"current user input, and the reference image, please generate {character_name}'s response."
        )
        
        # 生成角色回复（纯文本输出）
        response_result = self.inferencer(
            text=response_prompt if reference_image is None else None,
            image=reference_image,
            understanding_output=True,
            max_think_token_n=max_response_tokens,
            do_sample=do_sample_response,
            text_temperature=text_temperature
        )
        
        machine_response = response_result.get('text', '')
        
        # Step 2: 根据回复生成图像
        image_prompt = (
            f"Character Description: {description}\n\n"
            f"User Input: {user_text}\n\n"
            f"Character Response: {machine_response}\n\n"
            f"Please generate an image that visualizes this scene with the character "
            f"responding to the user's input based on the character description."
        )
        
        image_params = {
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        image_result = self.inferencer(text=image_prompt, **image_params)
        
        return {
            'response': machine_response,
            'image': image_result['image']
        }
    
    def personalized_response_with_thinking(
        self,
        character_name: str,
        description: str,
        opening: str,
        user_text: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        # 文本生成参数
        max_response_tokens: int = 500,
        do_sample_response: bool = True,
        text_temperature: float = 0.7,
        # 思考过程参数
        max_think_token_n: int = 1000,
        do_sample_thinking: bool = False,
        # 图像生成参数
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
        带思考过程的个性化对话和图像生成
        
        流程：
        1. 生成角色回复
        2. 生成思考过程和生成指令
        3. 根据回复、思考过程和指令生成图像
        
        Args:
            character_name: 角色名称
            description: 角色描述
            opening: 角色开场白
            user_text: 用户输入
            reference_image: 参考图像（可选）
            conversation_history: 对话历史（可选）
            max_response_tokens: 最大回复token数
            do_sample_response: 是否采样生成回复
            text_temperature: 文本生成温度
            max_think_token_n: 最大思考token数
            do_sample_thinking: 是否采样生成思考过程
            其他图像生成参数同generation函数
            
        Returns:
            包含角色回复、思考过程和生成图像的字典 
            {'response': str, 'thinking_process': str, 'generation_instruction': str, 'image': PIL.Image}
        """
        # 构建对话历史文本
        history_text = ""
        if conversation_history:
            for turn in conversation_history:
                role = turn.get('role', 'unknown')
                content = turn.get('text', '')
                history_text += f"{role}: {content}\n"
        
        # Step 1: 生成角色回复
        response_prompt = (
            f"Character Description: {description}\n\n"
            f"Opening: {opening}\n\n"
        )
        
        if history_text:
            response_prompt += f"Conversation History:\n{history_text}\n"
        
        response_prompt += f"User Input: {user_text}\n\n"
        response_prompt += (
            f"Based on the character description, opening, conversation history, "
            f"current user input, and the reference image, please generate {character_name}'s response."
        )
        
        # 生成角色回复（纯文本输出）
        response_result = self.inferencer(
            text=response_prompt if reference_image is None else None,
            image=reference_image,
            understanding_output=True,
            max_think_token_n=max_response_tokens,
            do_sample=do_sample_response,
            text_temperature=text_temperature
        )
        
        machine_response = response_result.get('text', '')
        
        # Step 2: 生成思考过程和生成指令
        thinking_prompt = (
            f"{THINK_SYSTEM_PROMPT}\n\n"
            f"Character Description: {description}\n\n"
            f"User Input: {user_text}\n\n"
            f"Character Response: {machine_response}\n\n"
            f"Please think about how to generate an image for this scene."
        )
        
        thinking_result = self.inferencer(
            text=thinking_prompt,
            understanding_output=True,
            max_think_token_n=max_think_token_n,
            do_sample=do_sample_thinking,
            text_temperature=0.3
        )
        
        thinking_text = thinking_result.get('text', '')
        extracted = self.extract_thinking_and_instruction(thinking_text)
        thinking_process = extracted['thinking_process']
        generation_instruction = extracted['generation_instruction']
        
        # Step 3: 根据回复、思考过程和指令生成图像
        image_prompt = (
            f"Character Description: {description}\n\n"
            f"User Input: {user_text}\n\n"
            f"Character Response: {machine_response}\n\n"
            f"Thinking Process: {thinking_process}\n\n"
            f"Generation Instruction: {generation_instruction}\n\n"
            f"Please generate an image that visualizes this scene with the character "
            f"responding to the user's input based on the character description, "
            f"thinking process, and generation instruction."
        )
        
        image_params = {
            'cfg_text_scale': cfg_text_scale,
            'cfg_img_scale': cfg_img_scale,
            'cfg_interval': cfg_interval,
            'timestep_shift': timestep_shift,
            'num_timesteps': num_timesteps,
            'cfg_renorm_min': cfg_renorm_min,
            'cfg_renorm_type': cfg_renorm_type,
            **kwargs
        }
        
        image_result = self.inferencer(text=image_prompt, **image_params)
        
        return {
            'response': machine_response,
            'thinking_process': thinking_process,
            'generation_instruction': generation_instruction,
            'image': image_result['image']
        }


# 便捷函数
def create_personalized_inference(
    model_path: str = "models/BAGEL-7B-MoT",
    checkpoint_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    **kwargs
) -> PersonalizedBagelInference:
    """
    创建个性化BAGEL推理器实例的便捷函数 (GRPO版本)
    
    Args:
        model_path: 模型路径
        checkpoint_path: 检查点路径
        vit_checkpoint_path: ViT权重的检查点路径（可选，用于合并权重）
        **kwargs: 其他初始化参数
        
    Returns:
        PersonalizedBagelInference实例
    """
    return PersonalizedBagelInference(
        model_path=model_path, 
        checkpoint_path=checkpoint_path,
        vit_checkpoint_path=vit_checkpoint_path,
        **kwargs
    )

