import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import lpips
from transformers import CLIPModel, CLIPProcessor
import os


class DiversityScorer(torch.nn.Module):
    """
    多样性评分器，计算LPIPS和Vendi Score
    """
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex').to(device).eval()
        
        # CLIP模型用于Vendi Score
        clip_path = "pretrained/clip-vit-large-patch14"
        if not os.path.exists(clip_path):
            clip_path = "openai/clip-vit-large-patch14"
        print(f"Using model from: {clip_path}")
        self.clip_model = CLIPModel.from_pretrained(clip_path).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        
    def _preprocess_images(self, images):
        """预处理图像为LPIPS需要的格式"""
        if isinstance(images, torch.Tensor):
            # 假设是 (B, C, H, W) 格式
            if images.max() <= 1.0:
                images = images * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            else:
                images = images / 127.5 - 1.0  # [0, 255] -> [-1, 1]
            return images.to(self.device).to(self.dtype)
        elif isinstance(images, np.ndarray):
            if len(images.shape) == 4:
                # (B, H, W, C) -> (B, C, H, W)
                images = images.transpose(0, 3, 1, 2)
            if images.dtype == np.uint8:
                images = images.astype(np.float32) / 127.5 - 1.0
            else:
                images = images * 2.0 - 1.0
            images = torch.from_numpy(images).to(self.device).to(self.dtype)
            return images
        else:
            # PIL Images
            images = [np.array(img) for img in images]
            images = np.array(images)
            if len(images.shape) == 4:
                images = images.transpose(0, 3, 1, 2)
            images = images.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(images).to(self.device).to(self.dtype)
    
    def compute_lpips_diversity(self, images):
        """
        计算LPIPS多样性分数
        
        Args:
            images: torch.Tensor of shape (G, C, H, W), G是组内图像数量
            
        Returns:
            diversity_score: 标量，组内图像的平均LPIPS距离
        """
        G = images.shape[0]
        if G < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 计算所有图像对之间的LPIPS距离
        total_distance = 0.0
        pair_count = 0
        
        for i in range(G):
            for j in range(i + 1, G):
                dist = self.lpips_model(images[i:i+1], images[j:j+1])
                total_distance += dist.item()
                pair_count += 1
        
        # 平均距离
        avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
        
        # 公式: R_div(X) = 1/(G(G-1)) * sum(LPIPS(x_i, x_j))
        # 这里直接返回平均距离，因为分母已经在计算中考虑了
        return torch.tensor(avg_distance, device=self.device)
    
    def compute_vendi_score(self, images):
        """
        计算Vendi Score（基于核矩阵特征值的多样性度量）
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            vendi_score: 标量，多样性分数
        """
        # 转换为PIL Images
        if isinstance(images, torch.Tensor):
            images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            images = (images * 255).clamp(0, 255).to(torch.uint8)
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(img) for img in images]
        elif isinstance(images, np.ndarray):
            if images.dtype != np.uint8:
                images = (images * 255).clip(0, 255).astype(np.uint8)
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = [Image.fromarray(img) for img in images]
            else:
                images = [Image.fromarray(img) for img in images]
        
        # 使用CLIP提取图像特征
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # 构建核矩阵（使用余弦相似度）
        kernel_matrix = image_features @ image_features.T
        
        # 计算特征值
        eigenvalues = torch.linalg.eigvals(kernel_matrix).real
        
        # Vendi Score = exp(entropy of normalized eigenvalues)
        # 归一化特征值
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # 计算熵（避免log(0)）
        epsilon = 1e-10
        eigenvalues = eigenvalues + epsilon
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        entropy = -(eigenvalues * torch.log(eigenvalues + epsilon)).sum()
        vendi_score = torch.exp(entropy)
        
        # 归一化到[0, 1]范围（除以最大可能值，即图像数量）
        G = len(images)
        vendi_score = vendi_score / G if G > 0 else torch.tensor(0.0, device=self.device)
        
        return vendi_score
    
    @torch.no_grad()
    def __call__(self, images, return_components=False):
        """
        计算多样性分数
        
        Args:
            images: 图像组，可以是torch.Tensor (G, C, H, W) 或 List of PIL Images
            return_components: 是否返回各个组件的分数
            
        Returns:
            diversity_scores: dict包含'lpips'和'vendi'分数，或单个综合分数
        """
        # 预处理图像用于LPIPS
        images_lpips = self._preprocess_images(images)
        
        # 计算LPIPS多样性
        lpips_score = self.compute_lpips_diversity(images_lpips)
        
        # 计算Vendi Score
        vendi_score = self.compute_vendi_score(images)
        
        if return_components:
            return {
                'lpips': lpips_score.item(),
                'vendi': vendi_score.item()
            }
        else:
            # 返回综合分数（简单平均）
            combined = (lpips_score + vendi_score) / 2.0
            return combined


def main():
    scorer = DiversityScorer(device="cuda", dtype=torch.float32)
    
    # 创建测试图像
    images = torch.randn(4, 3, 256, 256).to("cuda")
    images = (images + 1.0) / 2.0  # 归一化到[0, 1]
    
    scores = scorer(images, return_components=True)
    print("Diversity Scores:", scores)

if __name__ == "__main__":
    main()
