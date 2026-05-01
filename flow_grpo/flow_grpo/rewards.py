from PIL import Image
import numpy as np
import torch


def personalized_t2i_reward(device, reward_weights=None):
    """
    个性化T2I奖励函数，组合所有奖励组件
    
    Args:
        device: 计算设备
        reward_weights: 奖励权重字典，包含：
            - vqa: VQA分数权重
            - clip_t: CLIP-T分数权重
            - diversity_lpips: LPIPS多样性权重
            - dino_penalty: DINO相似度惩罚权重
            - dino_threshold_high: DINO高阈值（默认0.9）
            - dino_threshold_low: DINO低阈值（默认0.3）
    """
    from flow_grpo.vqa_scorer import VQAScorer
    from flow_grpo.diversity_scorer import DiversityScorer
    from flow_grpo.dino_scorer import DINOScorer
    from flow_grpo.clip_scorer import ClipScorer
    import numpy as np
    from PIL import Image
    
    # 默认权重
    if reward_weights is None:
        reward_weights = {
            "vqa": 0.3,
            "clip_t": 0.2,
            "diversity_lpips": 0.2,
            "dino_penalty": 0.1,
            "dino_threshold_high": 0.9,
            "dino_threshold_low": 0.3,
        }
    
    # 只初始化需要的scorers（根据权重判断）
    vqa_scorer = VQAScorer(device=device) if reward_weights.get("vqa", 0) > 0 else None
    diversity_scorer = (DiversityScorer(device=device) 
                       if reward_weights.get("diversity_lpips", 0) > 0
                       else None)
    dino_scorer = DINOScorer(device=device) if reward_weights.get("dino_penalty", 0) != 0 else None
    clip_scorer = ClipScorer(device=device) if reward_weights.get("clip_t", 0) > 0 else None
    
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        """
        计算个性化T2I奖励
        
        Args:
            images: 生成图像，torch.Tensor (N, C, H, W) 或 numpy array
            prompts: prompt列表
            metadata: metadata列表，每个元素包含'image'字段（reference图像路径）
            ref_images: 可选，预加载的reference图像
            only_strict: 未使用，保持接口一致性
            
        Returns:
            score_details: 包含各组件分数和总分的字典
            reward_metadata: 额外的元数据
        """
        # 转换图像格式
        if isinstance(images, torch.Tensor):
            # 保持tensor格式用于后续处理
            images_tensor = images
            if images.max() <= 1.0:
                images_for_pil = (images * 255).clamp(0, 255).to(torch.uint8)
            else:
                images_for_pil = images.to(torch.uint8)
            images_for_pil = images_for_pil.permute(0, 2, 3, 1).cpu().numpy()
            images_pil = [Image.fromarray(img) for img in images_for_pil]
        else:
            images_tensor = None
            images_pil = images
        
        num_images = len(images_pil)
        score_details = {}
        
        # 1. VQA分数
        if vqa_scorer is not None:
            vqa_scores = vqa_scorer(images_pil, prompts, metadata)
            score_details['vqa'] = vqa_scores.cpu().numpy().tolist()
        else:
            vqa_scores = torch.zeros(num_images, device=device)
            score_details['vqa'] = [0.0] * num_images
        
        # 2. CLIP-T分数
        if clip_scorer is not None:
            if images_tensor is None:
                # 转换为tensor
                images_array = np.array([np.array(img) for img in images_pil])
                images_array = images_array.transpose(0, 3, 1, 2)  # NHWC -> NCHW
                images_tensor = torch.tensor(images_array, dtype=torch.float32) / 255.0
            clip_t_scores = clip_scorer(images_tensor, prompts)
            score_details['clip_t'] = clip_t_scores.cpu().numpy().tolist()
        else:
            clip_t_scores = torch.zeros(num_images, device=device)
            score_details['clip_t'] = [0.0] * num_images
        
        # 3. 多样性奖励（需要按组计算）
        # 假设相同prompt的图像属于同一组
        diversity_lpips_scores = torch.zeros(num_images, device=device)
        
        if reward_weights.get("diversity_lpips", 0) > 0:
            # 按prompt分组
            prompt_to_indices = {}
            for i, prompt in enumerate(prompts):
                if prompt not in prompt_to_indices:
                    prompt_to_indices[prompt] = []
                prompt_to_indices[prompt].append(i)
            
            # 对每个组计算多样性
            for prompt, indices in prompt_to_indices.items():
                if len(indices) < 2:
                    # 组内只有一张图像，多样性为0
                    continue
                
                # 提取该组的图像
                group_images = [images_pil[i] for i in indices]
                
                # 计算多样性
                diversity_result = diversity_scorer(group_images, return_components=True)
                
                # 将多样性分数分配给组内每个图像
                lpips_score = diversity_result['lpips']
                
                for idx in indices:
                    diversity_lpips_scores[idx] = lpips_score
        
        score_details['diversity_lpips'] = diversity_lpips_scores.cpu().numpy().tolist()
        
        # 4. DINO相似度惩罚
        if reward_weights.get("dino_penalty", 0) > 0:
            # 从metadata中提取reference_images_path
            reference_folder = None
            for meta in metadata:
                if 'reference_images_path' in meta:
                    reference_folder = meta['reference_images_path']
                    break
            
            if reference_folder is not None:
                try:
                    dino_penalties, dino_similarities = dino_scorer(
                        images_pil,
                        reference_folder,
                        threshold_high=reward_weights.get("dino_threshold_high", 0.9),
                        threshold_low=reward_weights.get("dino_threshold_low", 0.3)
                    )
                    
                    score_details['dino_penalty'] = dino_penalties.cpu().numpy().tolist()
                    score_details['dino_similarity'] = dino_similarities.cpu().numpy().tolist()
                    all_dino_penalties = dino_penalties
                except Exception as e:
                    # 如果加载reference图像失败，不给予惩罚
                    print(f"Warning: DINO scorer failed: {e}")
                    score_details['dino_penalty'] = [0.0] * num_images
                    score_details['dino_similarity'] = [0.0] * num_images
                    all_dino_penalties = torch.zeros(num_images, device=device)
            else:
                score_details['dino_penalty'] = [0.0] * num_images
                score_details['dino_similarity'] = [0.0] * num_images
                all_dino_penalties = torch.zeros(num_images, device=device)
        else:
            all_dino_penalties = torch.zeros(num_images, device=device)
            score_details['dino_penalty'] = [0.0] * num_images
            score_details['dino_similarity'] = [0.0] * num_images
        
        # 5. 组合所有奖励
        total_scores = torch.zeros(num_images, device=device)
        
        if reward_weights.get("vqa", 0) > 0:
            print(f"vqa_scores: {vqa_scores}")
            total_scores += reward_weights["vqa"] * vqa_scores
        
        if reward_weights.get("clip_t", 0) > 0:
            print(f"clip_t_scores: {clip_t_scores}")
            total_scores += reward_weights["clip_t"] * clip_t_scores
        
        if reward_weights.get("diversity_lpips", 0) > 0:
            print(f"diversity_lpips_scores: {diversity_lpips_scores}")
            total_scores += reward_weights["diversity_lpips"] * diversity_lpips_scores
        
        if reward_weights.get("dino_penalty", 0) > 0:
            print(f"all_dino_penalties: {all_dino_penalties}")
            total_scores += reward_weights["dino_penalty"] * all_dino_penalties
        
        print(f"total_scores: {total_scores}")
        score_details['avg'] = total_scores.cpu().numpy().tolist()
        
        return score_details, {}
    
    return _fn


def multi_score(device, score_dict, reward_weights=None):
    """
    多任务奖励函数组合器
    
    Args:
        device: 计算设备
        score_dict: 奖励函数字典，格式为 {"reward_name": weight}
        reward_weights: personalized_t2i奖励函数的权重配置（仅当使用personalized_t2i时需要）
    """
    score_functions = {
        "personalized_t2i": personalized_t2i_reward,
    }
    
    score_fns={}
    for score_name, weight in score_dict.items():
        if score_name == "personalized_t2i":
            # personalized_t2i需要reward_weights参数
            score_fns[score_name] = score_functions[score_name](device, reward_weights)
        else:
            raise ValueError(f"Unknown score function: {score_name}. Only 'personalized_t2i' is supported.")

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "personalized_t2i":
                # personalized_t2i已经返回了组合后的分数
                score_details_dict, _ = score_fns[score_name](images, prompts, metadata, ref_images, only_strict)
                # 合并到score_details
                for key, value in score_details_dict.items():
                    score_details[key] = value
                # 使用'avg'作为总分数
                scores = score_details_dict.get('avg', [0.0] * len(images))
                
                # personalized_t2i已经包含了权重，直接使用
                if not total_scores:
                    total_scores = list(scores)
                else:
                    total_scores = [total + score for total, score in zip(total_scores, scores)]
        
        if 'avg' not in score_details:
            score_details['avg'] = total_scores
        return score_details, {}

    return _fn
