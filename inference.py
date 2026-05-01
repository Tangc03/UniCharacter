"""
Unified inference interface (UniCharacter Inference)

Built on the GRPO-based base inference and personalized inference, and
providing the following four high-level capabilities:
1. Character image generation (Role T2I Generation)
2. Visual understanding / VQA (Visual Understanding)
3. Knowledge question answering (Knowledge QA)
4. Multimodal role-play (Multimodal Role-play)
"""

from typing import Dict, List, Optional, Union, Any
import os
import json

from PIL import Image

from personalize_inference_grpo import PersonalizedBagelInference


class UniCharacterInference:
    """
    UniCharacter unified inference class.

    It is recommended to use the four high-level capability interfaces
    provided by this class directly, instead of calling the underlying
    BagelInference.
    """

    def __init__(
        self,
        model_path: str = "models/BAGEL-7B-MoT",
        checkpoint_path: Optional[str] = None,
        vit_checkpoint_path: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        seed: int = 42,
    ):
        """
        Args:
            model_path: path to the base model
            checkpoint_path: GRPO / SFT checkpoint path
            vit_checkpoint_path: optional path for merging additional ViT
                or generation related weights from SFT
            max_mem_per_gpu: maximum memory per GPU
            seed: random seed
        """
        # Personalized inference inherits from BagelInference and uses the same weights
        self._base = PersonalizedBagelInference(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            vit_checkpoint_path=vit_checkpoint_path,
            max_mem_per_gpu=max_mem_per_gpu,
            seed=seed,
        )

        # Root directory of character metadata (for auto-binding description / opening)
        self._data_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "personalized_data",
            "train",
        )

        # Mapping from character name to data directory
        # (directory names come from subfolders under data/personalized_data/train)
        self._character_dir_map: Dict[str, str] = {
            # Directly matching directory names
            "Adrien Brody": "Adrien Brody",
            "Bo": "Bo",
            "Butin": "Butin",
            "Coco": "Coco",
            "Leonardo": "Leonardo",
            "Mam": "Mam",
            "Mydieu": "Mydieu",
            "Wukong": "Wukong",
            # Directory names with prefixes
            "Chandler": "Friends-Chandler",
            "Joey": "Friends-Joey",
            "Gao Qiqiang": "Gao Qiqiang",
            "Furina": "Genshin-Furina",
            "Hermione": "Harry Potter-Hermione",
            "Ichihime": "Mahjong Soul-Ichihime",
            "Miki Nikaidou": "Mahjong Soul-Miki Nikaidou",
            "Rin Tohsaka": "Mahjong Soul-Rin Tohsaka",
            "Saber": "Mahjong Soul-Saber",
            "YuiYagi": "Mahjong Soul-YuiYagi",
            "Pikachu": "Pokemon-Pikachu",
            "Will in Vietnam": "Will In Vietnam",
        }

    # ------------------------------------------------------------------
    # 0. Character metadata loading (auto-binding description / opening)
    # ------------------------------------------------------------------
    def _load_character_meta(
        self, character_name: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Automatically load description and opening from
        data/personalized_data/train/*/annotation.json according to
        character_name.

        Returns None if the file is not found or parsing fails.
        """
        name = (character_name or "").strip()
        if not name:
            return None

        # Find the corresponding data directory based on the mapping table
        dir_name = self._character_dir_map.get(name, name)
        ann_path = os.path.join(self._data_root, dir_name, "annotation.json")

        if not os.path.isfile(ann_path):
            return None

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        description = str(data.get("description", "") or "").strip()
        opening = str(data.get("opening", "") or "").strip()

        return {"description": description, "opening": opening}

    # ------------------------------------------------------------------
    # 1. Character image generation (Role T2I Generation)
    # ------------------------------------------------------------------
    def generate_image(
        self,
        text: str,
        with_thinking: bool = False,
        max_think_token_n: int = 1000,
        do_sample_thinking: bool = False,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: Optional[List[float]] = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Text-to-image generation (optionally with thinking process).

        Returns:
            - with_thinking=False: {'image': PIL.Image}
            - with_thinking=True:  {'image': PIL.Image, 'text': str}
        """
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        if with_thinking:
            return self._base.generation_with_thinking(
                text=text,
                max_think_token_n=max_think_token_n,
                do_sample=do_sample_thinking,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                **kwargs,
            )
        else:
            return self._base.generation(
                text=text,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                **kwargs,
            )

    # ------------------------------------------------------------------
    # 2. Visual understanding / VQA
    # ------------------------------------------------------------------
    def visual_understanding(
        self,
        image: Union[str, Image.Image],
        question: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Visual understanding / VQA.

        Args:
            image: image path or PIL.Image
            question: question regarding the image
        Returns:
            {'text': str}
        """
        return self._base.understanding(
            image=image,
            text=question,
            max_think_token_n=max_think_token_n,
            do_sample=do_sample,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 3. Knowledge question answering (Knowledge QA)
    # ------------------------------------------------------------------
    def knowledge_qa(
        self,
        question: str,
        max_think_token_n: int = 1000,
        do_sample: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Pure text knowledge question answering
        (can be viewed as VQA without images).

        Args:
            question: user question
        Returns:
            {'text': str}
        """
        return self._base.understanding(
            image=None,
            text=question,
            max_think_token_n=max_think_token_n,
            do_sample=do_sample,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 4. Multimodal role-play
    # ------------------------------------------------------------------
    def role_play(
        self,
        character_name: str,
        description: str,
        opening: str,
        user_text: str,
        reference_image: Optional[Union[str, Image.Image]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        with_thinking: bool = False,
        # text generation parameters
        max_response_tokens: int = 500,
        do_sample_response: bool = True,
        text_temperature: float = 0.7,
        # thinking process parameters
        max_think_token_n: int = 1000,
        do_sample_thinking: bool = False,
        # image generation parameters
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: Optional[List[float]] = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Multimodal role-play: character response + character image.

        Returns:
            - with_thinking=False:
                {'response': str, 'image': PIL.Image}
            - with_thinking=True:
                {'response': str, 'thinking_process': str,
                 'generation_instruction': str, 'image': PIL.Image}
        """
        # If the character exists in the dataset, prefer the description / opening
        # from the dataset.
        meta = self._load_character_meta(character_name)
        if meta is not None:
            if meta.get("description"):
                description = meta["description"] or ""
            # opening can be an empty string, but we still follow the dataset
            # to ensure binding between character_name and opening.
            if "opening" in meta:
                opening = meta["opening"] or ""

        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        if with_thinking:
            return self._base.personalized_response_with_thinking(
                character_name=character_name,
                description=description,
                opening=opening,
                user_text=user_text,
                reference_image=reference_image,
                conversation_history=conversation_history,
                max_response_tokens=max_response_tokens,
                do_sample_response=do_sample_response,
                text_temperature=text_temperature,
                max_think_token_n=max_think_token_n,
                do_sample_thinking=do_sample_thinking,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                **kwargs,
            )
        else:
            return self._base.personalized_response_without_thinking(
                character_name=character_name,
                description=description,
                opening=opening,
                user_text=user_text,
                reference_image=reference_image,
                conversation_history=conversation_history,
                max_response_tokens=max_response_tokens,
                do_sample_response=do_sample_response,
                text_temperature=text_temperature,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                **kwargs,
            )


def create_unicharacter_inference(
    model_path: str = "models/BAGEL-7B-MoT",
    checkpoint_path: Optional[str] = None,
    vit_checkpoint_path: Optional[str] = None,
    **kwargs,
) -> UniCharacterInference:
    """
    Helper function: create a UniCharacterInference instance.
    """
    return UniCharacterInference(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        vit_checkpoint_path=vit_checkpoint_path,
        **kwargs,
    )


if __name__ == "__main__":
    # Simple self-test example (please modify paths according to your environment)
    print("UniCharacter unified inference interface quick self-test")
    from pathlib import Path

    default_model_path = "models/BAGEL-7B-MoT"
    default_vit_ckpt = "results/checkpoints/mahjong_ichihime_extension_1010_t2i_vlm/0000300/model.safetensors"
    default_ckpt = "PersonalizedBAGEL/results/checkpoints/mahjong_ichihime_extension_1010_t2i_vlm/0000300/model.safetensors"

    inference = create_unicharacter_inference(
        model_path=default_model_path,
        checkpoint_path=default_ckpt,
        vit_checkpoint_path=default_vit_ckpt,
    )

    out_dir = Path("test_images/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) T2I
    res = inference.generate_image("Ichihime chasing a butterfly")
    res["image"].save(out_dir / "t2i_ichihime.png")

    # 2) VQA
    res = inference.visual_understanding("data/personalized_data/train/Mahjong Soul-Ichihime/1.png", "What's the color of Ichihime's hair?")
    print("VQA:", res["text"])

    # 3) Knowledge QA
    res = inference.knowledge_qa("When do you born?")
    print("Knowledge QA:", res["text"])

    # 4) Role-play
    res = inference.role_play(
        character_name="Ichihime",
        description = "",
        opening = "",
        user_text="Hi, Ichihime. How are you?"
    )
    print("Role-play:", res["response"])

    print("Please modify the example paths in __main__ as needed and run the test.")

