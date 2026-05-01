"""
从YAML文件加载个性化T2I BAGEL配置的辅助模块
"""
import os
import yaml


def load_personalized_t2i_bagel_config(use_lora=False, config_file_name=None):
    """
    从YAML文件加载配置
    
    Args:
        use_lora: 是否使用LoRA版本
        config_file_name: 配置文件名（可选），如果不指定则使用默认文件名
                         也可以通过环境变量 PERSONALIZED_T2I_BAGEL_CONFIG 指定
        
    Returns:
        dict: 配置字典，如果文件不存在则返回None
    """
    # 优先使用环境变量，然后是函数参数，最后是默认值
    if config_file_name is None:
        config_file_name = os.environ.get('PERSONALIZED_T2I_BAGEL_CONFIG', 'personalized_t2i_bagel.yaml')
    
    config_file = os.path.join(os.path.dirname(__file__), config_file_name)
    
    if not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    return yaml_config


def apply_yaml_config_to_config_object(config, yaml_config, use_lora=False):
    """
    将YAML配置应用到config对象
    
    Args:
        config: ml_collections.ConfigDict对象
        yaml_config: 从YAML加载的配置字典
        use_lora: 是否使用LoRA版本
    """
    variant = 'lora' if use_lora else 'full'
    gpu_number = yaml_config.get('gpu_number', 8)
    
    # 数据集路径
    dataset_path = yaml_config.get('dataset', 'Bo_pure_t2i.jsonl')
    config.dataset = os.path.join(os.getcwd(), dataset_path)
    
    # Run名称
    run_name_cfg = yaml_config.get('run_name', {})
    config.run_name = run_name_cfg.get(variant, f"[bagel-personalized-t2i-{variant}]-8gpu")
    
    # 模型路径配置
    model_cfg = yaml_config.get('model', {})
    local_model_path = os.path.join(os.getcwd(), model_cfg.get('local_path', 'models/BAGEL-7B-MoT'))
    remote_oss_path = model_cfg.get('remote_oss_path', 'models/BAGEL-7B-MoT')
    hf_id = model_cfg.get('huggingface_id', 'ByteDance-Seed/BAGEL-7B-MoT')
    
    if os.path.exists(local_model_path):
        config.pretrained.model = local_model_path
        print(f"使用本地模型: {local_model_path}")
    elif os.path.exists(remote_oss_path):
        config.pretrained.model = remote_oss_path
        print(f"使用远程OSS模型: {remote_oss_path}")
    else:
        config.pretrained.model = hf_id
        print(f"使用 Hugging Face 仓库: {config.pretrained.model}")
    
    # 采样配置
    sample_cfg = yaml_config.get('sample', {})
    config.sample.num_steps = sample_cfg.get('num_steps', 15)
    config.sample.eval_num_steps = sample_cfg.get('eval_num_steps', 50)
    config.sample.guidance_scale = sample_cfg.get('guidance_scale', 4.0)
    config.sample.eval_guidance_scale = sample_cfg.get('eval_guidance_scale', 4.0)
    config.sample.train_batch_size = sample_cfg.get('train_batch_size', 6)
    config.sample.num_image_per_prompt = sample_cfg.get('num_image_per_prompt', 4)
    config.sample.test_batch_size = sample_cfg.get('test_batch_size', 1)
    config.sample.same_latent = sample_cfg.get('same_latent', False)
    config.sample.global_std = sample_cfg.get('global_std', False)
    config.sample.noise_level = sample_cfg.get('noise_level', 1.3)
    
    # SDE窗口配置：full版本使用3，lora版本使用2
    if use_lora:
        config.sample.sde_window_size = sample_cfg.get('sde_window_size_lora', 2)
    else:
        config.sample.sde_window_size = sample_cfg.get('sde_window_size', 3)
    
    sde_range = sample_cfg.get('sde_window_range', [0, None])
    if sde_range[1] is None:
        sde_range[1] = config.sample.num_steps // 2
    config.sample.sde_window_range = tuple(sde_range)
    
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    
    # 训练配置
    train_cfg = yaml_config.get('train', {})
    config.train.cfg = train_cfg.get('cfg', True)
    config.train.ema = train_cfg.get('ema', False)
    config.train.num_inner_epochs = train_cfg.get('num_inner_epochs', 1)
    # 处理科学计数法字符串，转换为float
    clip_range_lt = train_cfg.get('clip_range_lt', 1e-5)
    clip_range_gt = train_cfg.get('clip_range_gt', 1e-5)
    config.train.clip_range_lt = float(clip_range_lt) if isinstance(clip_range_lt, str) else clip_range_lt
    config.train.clip_range_gt = float(clip_range_gt) if isinstance(clip_range_gt, str) else clip_range_gt
    config.train.beta = train_cfg.get('beta', 0)
    # 处理科学计数法字符串，转换为float
    learning_rate = train_cfg.get('learning_rate', 1e-4)
    config.train.learning_rate = float(learning_rate) if isinstance(learning_rate, str) else learning_rate
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    
    # 其他配置
    config.resolution = yaml_config.get('resolution', 512)
    config.mixed_precision = yaml_config.get('mixed_precision', 'bf16')
    config.use_lora = use_lora  # 根据函数参数设置
    config.activation_checkpointing = yaml_config.get('activation_checkpointing', True)
    config.fsdp_optimizer_offload = yaml_config.get('fsdp_optimizer_offload', True)
    config.save_freq = yaml_config.get('save_freq', 30)
    config.eval_freq = yaml_config.get('eval_freq', 5)

    # 总训练epoch数（如果不设置，将使用base.py中的默认值100000）
    if 'num_epochs' in yaml_config:
        config.num_epochs = yaml_config.get('num_epochs')
    
    # 保存目录
    save_dir_cfg = yaml_config.get('save_dir', {})
    config.save_dir = save_dir_cfg.get(variant, f'flow_grpo/logs/personalized_t2i/bagel-{variant}')
    
    # Prompt函数
    config.prompt_fn = yaml_config.get('prompt_fn', 'personalized_t2i')
    config.per_prompt_stat_tracking = yaml_config.get('per_prompt_stat_tracking', True)
    
    # 奖励函数配置
    reward_fn_cfg = yaml_config.get('reward_fn', {})
    config.reward_fn = reward_fn_cfg if reward_fn_cfg else {"personalized_t2i": 1.0}
    
    # 奖励权重配置
    reward_weights_cfg = yaml_config.get('reward_weights', {})
    # 直接赋值字典，与 grpo.py 中的方式一致
    if reward_weights_cfg:
        config.reward_weights = reward_weights_cfg
    else:
        # 使用默认值（如果 YAML 中没有配置）
        config.reward_weights = {
            "vqa": 0.3,
            "clip_t": 0.2,
            "diversity_lpips": 0.2,
            "dino_penalty": -0.1,
            "dino_threshold_high": 0.9,
            "dino_threshold_low": 0.3,
        }

