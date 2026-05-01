from cgi import print_arguments
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import numpy as np
import os

class VQAScorer(torch.nn.Module):
    """
    BLIP-VQA评分器，用于评估图像与文本的对齐质量
    使用较小的BLIP-VQA模型以提高效率
    """
    def __init__(self, device="cuda", dtype=torch.float32, model_name="Salesforce/blip-vqa-base"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        model_path = "pretrained/blip-vqa-base"
        if not os.path.exists(model_path):
            model_path = model_name
        print(f"Using model from: {model_path}")

        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForQuestionAnswering.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        
    @torch.no_grad()
    def __call__(self, images, prompts, metadata=None):
        """
        计算VQA分数
        
        Args:
            images: List of PIL Images or numpy arrays
            prompts: List of prompt strings
            metadata: List of metadata dicts, 每个dict可能包含'vqa_questions'字段
            
        Returns:
            scores: torch.Tensor of shape (batch_size,), 范围在0-1之间
        """
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            # 假设是 (B, C, H, W) 格式，值在[0, 1]或[0, 255]
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
        
        # 批量处理
        batch_size = len(images)
        scores = []
        
        for i in range(batch_size):
            # 检查metadata中是否有vqa_questions
            vqa_questions = None
            if metadata is not None and i < len(metadata):
                vqa_questions = metadata[i].get('vqa_questions', None)
            
            if vqa_questions and len(vqa_questions) > 0:
                # 使用metadata中的多个VQA问题
                question_scores = []
                total_weight = 0.0
                
                for q_item in vqa_questions:
                    question = q_item.get('question', '')
                    expected_answer = q_item.get('answer', '')
                    weight = q_item.get('weight', 1.0)
                    
                    if not question or not expected_answer:
                        continue
                    
                    # 处理输入
                    inputs = self.processor(
                        images=images[i],
                        text=question,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # 获取模型生成的答案
                    out = self.model.generate(**inputs, max_length=50)
                    predicted_answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
                    
                    # 将答案转换为小写进行严格比较
                    predicted_answer_lower = predicted_answer.lower()
                    expected_answer_lower = str(expected_answer).lower()

                    # print 生图的prompt，predicted和expected answer来debug
                    print(f"Prompt: {prompts[i]}, Question: {question}, Predicted: {predicted_answer}, Expected: {expected_answer}")
                    
                    # 严格匹配：答案完全相同（小写）时给分
                    if predicted_answer_lower == expected_answer_lower:
                        question_scores.append(weight)
                    else:
                        question_scores.append(0.0)
                    
                    total_weight += weight
                
                # 计算加权平均分数
                if total_weight > 0:
                    score = sum(question_scores) / total_weight
                else:
                    score = 0.0
            else:
                # 回退到原来的方法：将prompt转换为VQA问题
                question = f"Does this image accurately depict: {prompts[i]}?"
                
                # 处理输入
                inputs = self.processor(
                    images=images[i],
                    text=question,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取答案
                out = self.model.generate(**inputs, max_length=10)
                answer = self.processor.decode(out[0], skip_special_tokens=True).lower()
                
                # 将答案转换为分数
                # 如果答案是肯定的（yes, true, correct等），给高分
                positive_words = ['yes', 'true', 'correct', 'accurate', 'depicts', 'shows']
                negative_words = ['no', 'false', 'incorrect', 'not', 'doesn\'t', 'does not']
                
                if any(word in answer for word in positive_words):
                    score = 1.0
                elif any(word in answer for word in negative_words):
                    score = 0.0
                else:
                    # 默认中等分数
                    score = 0.5
            
            scores.append(score)
        
        return torch.tensor(scores, device=self.device, dtype=self.dtype)


def main():
    scorer = VQAScorer(device="cuda", dtype=torch.float32)
    
    images = [
        Image.open("nasa.jpg") if hasattr(Image, 'open') else None
    ]
    prompts = [
        'A astronaut\'s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    
    if images[0] is not None:
        scores = scorer(images, prompts)
        print("VQA Scores:", scores)

if __name__ == "__main__":
    main()
