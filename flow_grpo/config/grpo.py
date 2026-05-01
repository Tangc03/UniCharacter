import ml_collections
import imp
import os
import yaml

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))
# Import personalized T2I configuration loading module
try:
    from .load_personalized_t2i_config import load_personalized_t2i_bagel_config, apply_yaml_config_to_config_object
except ImportError:
    # If import fails (e.g. running directly), try relative import
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_personalized_t2i_config import load_personalized_t2i_bagel_config, apply_yaml_config_to_config_object

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def personalized_t2i():
    """
    Personalized T2I training configuration, using generation instruction and image correspondence for training
    """
    config = base.get_config()
    
    # Dataset path (jsonl format, containing text, image, character, scene_id fields)
    config.dataset = os.path.join(os.getcwd(), "Bo_pure_t2i.jsonl")
    
    # Model configuration
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    
    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 4  # GRPO generates one group of images at a time
    config.sample.num_batches_per_epoch = 2
    config.sample.test_batch_size = 2
    
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    
    config.save_freq = 60
    config.eval_freq = 30
    config.save_dir = 'logs/personalized_t2i'
    
    # Reward function configuration
    config.reward_fn = {
        "personalized_t2i": 1.0,
    }
    
    # Reward weight configuration
    config.reward_weights = {
        "vqa": 0.3,                    # VQA score weight
        "clip_t": 0.2,                # CLIP-T score weight
        "diversity_lpips": 0.2,       # LPIPS diversity weight
        "dino_penalty": 0.1,          # DINO similarity penalty
        "dino_threshold_high": 0.9,   # DINO high threshold (punish if over this value, prevent overfitting)
        "dino_threshold_low": 0.3,    # DINO low threshold (punish if below this value, prevent diverging too far)
    }
    
    config.prompt_fn = "personalized_t2i"  # Need to implement the corresponding prompt function
    config.per_prompt_stat_tracking = True
    
    return config


def personalized_t2i_bagel():
    """
    Personalized T2I + BAGEL training configuration
    Use BAGEL model for personalized image generation training
    Configuration can be read from config/personalized_t2i_bagel.yaml file
    """
    config = compressibility()
    
    # Try to load configuration from YAML file
    try:
        yaml_config = load_personalized_t2i_bagel_config(use_lora=False)
        if yaml_config:
            apply_yaml_config_to_config_object(config, yaml_config, use_lora=False)
            print("✓ Successfully loaded configuration from YAML file")
            return config
        else:
            print("Warning: YAML configuration file does not exist or is empty, using default configuration")
    except Exception as e:
        import traceback
        print(f"Warning: Failed to load configuration from YAML file, using default configuration")
        print(f"Error details: {type(e).__name__}: {e}")
        print(f"Error location: {traceback.format_exc()}")
    
    # If YAML file does not exist or loading fails, use default configuration (for backward compatibility)
    gpu_number = 8
    # Dataset path (jsonl format, containing text, image, character, scene_id fields)
    config.dataset = os.path.join(os.getcwd(), "Bo_pure_t2i.jsonl")
    
    config.run_name = "[bagel-personalized-t2i]-8gpu"
    
    # Use local model if it exists, otherwise use Hugging Face repository identifier
    local_model_path = os.path.join(os.getcwd(), "models/BAGEL-7B-MoT")
    remote_oss_path = "models/BAGEL-7B-MoT"
    
    if os.path.exists(local_model_path):
        config.pretrained.model = local_model_path
        print(f"Using local model: {local_model_path}")
    elif os.path.exists(remote_oss_path):
        config.pretrained.model = remote_oss_path
        print(f"Using remote OSS model: {remote_oss_path}")
    else:
        # If local and remote paths do not exist, use Hugging Face repository identifier
        config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"
        print(f"Using Hugging Face repository: {config.pretrained.model}")
    
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.train.cfg = True     # No effect for BAGEL, always use cfg in code.
    config.train.ema = False
    config.use_lora = False  # Can be set to True to use LoRA training

    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 4  # GRPO generates one group of images at a time
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 1 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2

    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    config.sample.same_latent = False
    config.sample.global_std = False
    config.sample.noise_level = 1.3

    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)

    config.save_freq = 30 # epoch
    config.eval_freq = 5
    config.save_dir = 'flow_grpo/logs/personalized_t2i/bagel-full'
    
    # Reward function configuration
    config.reward_fn = {
        "personalized_t2i": 1.0,
    }
    
    # 奖励权重配置
    config.reward_weights = {
        "vqa": 0.3,                    # VQA score weight
        "clip_t": 0.2,                # CLIP-T score weight
        "diversity_lpips": 0.2,       # LPIPS diversity weight
        "dino_penalty": 0.1,          # DINO similarity penalty
        "dino_threshold_high": 0.9,   # DINO high threshold (punish if over this value, prevent overfitting)
        "dino_threshold_low": 0.3,    # DINO low threshold (punish if below this value, prevent diverging too far)
    }
    
    config.prompt_fn = "personalized_t2i"
    config.per_prompt_stat_tracking = True

    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    return config


def get_config(name):
    return globals()[name]()
