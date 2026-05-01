import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os


class DINOScorer(torch.nn.Module):
    """
    DINO相似度评分器，用于计算生成图像与reference图像的最大相似度
    并实现阈值惩罚机制
    """
    def __init__(self, device="cuda", dtype=torch.float32, model_name="facebook/dinov2-base"):
        super().__init__()
        self.device = device
        self.dtype = dtype

        model_path = "pretrained/dinov2-base"
        if not os.path.exists(model_path):
            model_path = model_name
        print(f"Using model from: {model_path}")
        
        # 加载DINOv2模型
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        
    def _load_image(self, image_path):
        """加载图像"""
        if isinstance(image_path, str):
            if not os.path.isabs(image_path):
                # 相对路径，尝试从当前工作目录或数据集目录查找
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert('RGB')
                else:
                    # 尝试从常见的数据集目录查找
                    possible_paths = [
                        os.path.join(os.getcwd(), image_path),
                        os.path.join(os.getcwd(), "dataset", image_path),
                        os.path.join(os.getcwd(), "personalized_data", image_path),
                    ]
                    img = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            img = Image.open(path).convert('RGB')
                            break
                    if img is None:
                        raise FileNotFoundError(f"Image not found: {image_path}")
            else:
                img = Image.open(image_path).convert('RGB')
            return img
        elif isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_path)}")
    
    def _preprocess_images(self, images):
        """预处理图像"""
        if isinstance(images, torch.Tensor):
            # 假设是 (B, C, H, W) 格式
            if images.max() <= 1.0:
                images = (images * 255).clamp(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy()  # BCHW -> BHWC
            images = [Image.fromarray(img) for img in images]
        elif isinstance(images, np.ndarray):
            if images.dtype != np.uint8:
                images = (images * 255).clip(0, 255).astype(np.uint8)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = [Image.fromarray(img) for img in images]
            else:
                images = [Image.fromarray(img) for img in images]
        
        return images
    
    @torch.no_grad()
    def extract_features(self, images):
        """
        提取DINO特征
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            features: torch.Tensor of shape (N, feature_dim)
        """
        images = self._preprocess_images(images)
        
        # 批量处理
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # 提取特征
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state
        
        # 使用CLS token的特征（第一个token）
        features = features[:, 0, :]
        
        # 归一化
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
        return features
    
    @torch.no_grad()
    def compute_max_similarity(self, gen_images, ref_images):
        """
        计算生成图像与reference图像的最大相似度
        
        Args:
            gen_images: 生成图像列表
            ref_images: reference图像列表（从metadata加载）
            
        Returns:
            max_similarities: torch.Tensor of shape (N,), 每个生成图像的最大相似度
        """
        # 提取特征
        gen_features = self.extract_features(gen_images)
        ref_features = self.extract_features(ref_images)
        
        # 计算相似度矩阵
        similarity_matrix = gen_features @ ref_features.T  # (N_gen, N_ref)
        
        # 获取每个生成图像的最大相似度
        max_similarities, _ = torch.max(similarity_matrix, dim=1)
        
        return max_similarities
    
    @torch.no_grad()
    def compute_penalty(self, max_similarities, threshold_high=0.9, threshold_low=0.3):
        """
        根据阈值计算惩罚分数
        
        Args:
            max_similarities: 最大相似度分数
            threshold_high: 高阈值，超过此值给予惩罚（防止过拟合）
            threshold_low: 低阈值，低于此值给予惩罚（防止偏离太远）
            
        Returns:
            penalties: torch.Tensor of shape (N,), 惩罚分数（负值表示惩罚）
        """
        penalties = torch.zeros_like(max_similarities)
        
        # 超过高阈值：惩罚
        high_mask = max_similarities > threshold_high
        penalties[high_mask] = -(max_similarities[high_mask] - threshold_high) * 2.0
        
        # 低于低阈值：惩罚
        low_mask = max_similarities < threshold_low
        penalties[low_mask] = -(threshold_low - max_similarities[low_mask]) * 2.0
        
        return penalties
    
    def _load_folder_images(self, folder_path):
        """
        从文件夹中加载所有参考图片
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            images: PIL Image列表
        """
        # 处理相对路径
        if not os.path.isabs(folder_path):
            # 尝试相对于当前工作目录
            possible_paths = [
                folder_path,
                os.path.join(os.getcwd(), folder_path),
            ]
            folder_path_resolved = None
            for path in possible_paths:
                if os.path.isdir(path):
                    folder_path_resolved = path
                    break
            if folder_path_resolved is None:
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            folder_path = folder_path_resolved
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # 加载所有图片
        images = []
        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
                try:
                    img = Image.open(file_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        if len(images) == 0:
            raise FileNotFoundError(f"No images found in folder: {folder_path}")
        
        print(f"Loaded {len(images)} reference images from {folder_path}")
        return images
    
    @torch.no_grad()
    def __call__(self, gen_images, reference_folder_path, threshold_high=0.9, threshold_low=0.3):
        """
        计算DINO相似度惩罚
        
        Args:
            gen_images: 生成图像
            reference_folder_path: reference图像文件夹路径（从metadata中的reference_images_path获取）
                                  例如："Bo" 或 "personalized_data/train/Bo"
            threshold_high: 高阈值
            threshold_low: 低阈值
            
        Returns:
            penalties: 惩罚分数
            max_similarities: 最大相似度（用于调试）
        """
        # 加载reference文件夹中的所有图像
        ref_images = self._load_folder_images(reference_folder_path)
        
        # 计算最大相似度
        max_similarities = self.compute_max_similarity(gen_images, ref_images)
        
        # 计算惩罚
        penalties = self.compute_penalty(max_similarities, threshold_high, threshold_low)
        
        return penalties, max_similarities


def main():
    scorer = DINOScorer(device="cuda", dtype=torch.float32)
    
    # 测试
    gen_images = [Image.new('RGB', (256, 256), color='red') for _ in range(2)]
    ref_paths = ["test_ref1.jpg", "test_ref2.jpg"]  # 需要实际存在的路径
    
    try:
        penalties, similarities = scorer(gen_images, ref_paths)
        print("Penalties:", penalties)
        print("Similarities:", similarities)
    except FileNotFoundError as e:
        print(f"Test skipped: {e}")

if __name__ == "__main__":
    main()
