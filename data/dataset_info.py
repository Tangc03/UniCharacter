# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        # Adrien Brody T2I dataset
        'adrien_brody_t2i': {
            'data_dir': 'data/personalized_data/train/Adrien Brody/processed/t2i/Adrien Brody_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Adrien Brody-With-Thinking T2I dataset
        'adrien_brody_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Adrien Brody/processed/t2i/Adrien Brody_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Adrien Brody-Generation-Instruction-Only T2I dataset
        'adrien_brody_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Adrien Brody/processed/t2i/Adrien Brody_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Adrien Brody-Pure T2I dataset
        'adrien_brody_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Adrien Brody/processed/t2i/Adrien Brody_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Bo T2I dataset
        'bo_t2i': {
            'data_dir': 'data/personalized_data/train/Bo/processed/t2i/Bo_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Bo-With-Thinking T2I dataset
        'bo_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Bo/processed/t2i/Bo_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Bo-Generation-Instruction-Only T2I dataset
        'bo_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Bo/processed/t2i/Bo_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Bo-Pure T2I dataset
        'bo_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Bo/processed/t2i/Bo_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Bo-Full T2I dataset (merge Bo_t2i and Bo_t2i_with_thinking, each repeated 10 times)
        'bo_t2i_full': {
            'data_dir': 'data/personalized_data/train/Bo/processed/t2i/Bo_t2i_full_parquet',
            'num_files': 1,
            'num_total_samples': 240,
        },

        # Butin T2I dataset
        'butin_t2i': {
            'data_dir': 'data/personalized_data/train/Butin/processed/t2i/Butin_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Butin-With-Thinking T2I dataset
        'butin_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Butin/processed/t2i/Butin_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Butin-Generation-Instruction-Only T2I dataset
        'butin_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Butin/processed/t2i/Butin_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Butin-Pure T2I dataset
        'butin_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Butin/processed/t2i/Butin_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },

        # Coco T2I dataset
        'coco_t2i': {
            'data_dir': 'data/personalized_data/train/Coco/processed/t2i/Coco_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Coco-With-Thinking T2I dataset
        'coco_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Coco/processed/t2i/Coco_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Coco-Generation-Instruction-Only T2I dataset
        'coco_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Coco/processed/t2i/Coco_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Coco-Pure T2I dataset
        'coco_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Coco/processed/t2i/Coco_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },

        # Friends-Chandler T2I dataset
        'friends_chandler_t2i': {
            'data_dir': 'data/personalized_data/train/Friends-Chandler/processed/t2i/Chandler_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Friends-Chandler-With-Thinking T2I dataset
        'friends_chandler_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Friends-Chandler/processed/t2i/Chandler_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Friends-Chandler-Generation-Instruction-Only T2I dataset
        'friends_chandler_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Friends-Chandler/processed/t2i/Chandler_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Friends-Chandler-Pure T2I dataset
        'friends_chandler_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Friends-Chandler/processed/t2i/Chandler_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Friends-Joey T2I dataset
        'friends_joey_t2i': {
            'data_dir': 'data/personalized_data/train/Friends-Joey/processed/t2i/Joey_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Friends-Joey-With-Thinking T2I dataset
        'friends_joey_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Friends-Joey/processed/t2i/Joey_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Friends-Joey-Generation-Instruction-Only T2I dataset
        'friends_joey_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Friends-Joey/processed/t2i/Joey_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },
        # Friends-Joey-Pure T2I dataset
        'friends_joey_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Friends-Joey/processed/t2i/Joey_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 11,
        },

        # Gao Qiqiang T2I dataset
        'gao_qiqiang_t2i': {
            'data_dir': 'data/personalized_data/train/Gao Qiqiang/processed/t2i/Gao Qiqiang_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Gao Qiqiang-With-Thinking T2I dataset
        'gao_qiqiang_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Gao Qiqiang/processed/t2i/Gao Qiqiang_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Gao Qiqiang-Generation-Instruction-Only T2I dataset
        'gao_qiqiang_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Gao Qiqiang/processed/t2i/Gao Qiqiang_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Gao Qiqiang-Pure T2I dataset
        'gao_qiqiang_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Gao Qiqiang/processed/t2i/Gao Qiqiang_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Genshin-Furina T2I dataset
        'genshin_furina_t2i': {
            'data_dir': 'data/personalized_data/train/Genshin-Furina/processed/t2i/Furina_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 9,
        },
        # Genshin-Furina-With-Thinking T2I dataset
        'genshin_furina_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Genshin-Furina/processed/t2i/Furina_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 9,
        },
        # Genshin-Furina-Generation-Instruction-Only T2I dataset
        'genshin_furina_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Genshin-Furina/processed/t2i/Furina_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 9,
        },
        # Genshin-Furina-Pure T2I dataset
        'genshin_furina_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Genshin-Furina/processed/t2i/Furina_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 9,
        },

        # Harry Potter-Hermione T2I dataset
        'harry_potter_hermione_t2i': {
            'data_dir': 'data/personalized_data/train/Harry Potter-Hermione/processed/t2i/Hermione_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Harry Potter-Hermione-With-Thinking T2I dataset
        'harry_potter_hermione_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Harry Potter-Hermione/processed/t2i/Hermione_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Harry Potter-Hermione-Generation-Instruction-Only T2I dataset
        'harry_potter_hermione_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Harry Potter-Hermione/processed/t2i/Hermione_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Harry Potter-Hermione-Pure T2I dataset
        'harry_potter_hermione_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Harry Potter-Hermione/processed/t2i/Hermione_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Leonardo T2I dataset
        'leonardo_t2i': {
            'data_dir': 'data/personalized_data/train/Leonardo/processed/t2i/Leonardo_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Leonardo-With-Thinking T2I dataset
        'leonardo_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Leonardo/processed/t2i/Leonardo_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Leonardo-Generation-Instruction-Only T2I dataset
        'leonardo_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Leonardo/processed/t2i/Leonardo_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Leonardo-Pure T2I dataset
        'leonardo_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Leonardo/processed/t2i/Leonardo_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Mahjong Soul-Ichihime T2I dataset
        'mahjong_ichihime_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/t2i/Ichihime_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Ichihime-With-Thinking T2I dataset
        "mahjong_ichihime_t2i_with_thinking": {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/t2i/Ichihime_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Ichihime-Generation-Instruction-Only T2I dataset
        "mahjong_ichihime_t2i_generation_instruction_only": {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/t2i/Ichihime_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Ichihime-Pure T2I dataset
        'mahjong_ichihime_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/t2i/Ichihime_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },

        # Mahjong Soul-Miki Nikaidou T2I dataset
        'mahjong_miki_nikaidou_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/t2i/Miki Nikaidou_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 6,
        },
        # Mahjong Soul-Miki Nikaidou-With-Thinking T2I dataset
        'mahjong_miki_nikaidou_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/t2i/Miki Nikaidou_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 6,
        },
        # Mahjong Soul-Miki Nikaidou-Generation-Instruction-Only T2I dataset
        'mahjong_miki_nikaidou_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/t2i/Miki Nikaidou_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 6,
        },
        # Mahjong Soul-Miki Nikaidou-Pure T2I dataset
        'mahjong_miki_nikaidou_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/t2i/Miki Nikaidou_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 6,
        },

        # Mahjong Soul-Rin Tohsaka T2I dataset
        'mahjong_rin_tohsaka_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/t2i/Rin Tohsaka_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Rin Tohsaka-With-Thinking T2I dataset
        'mahjong_rin_tohsaka_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/t2i/Rin Tohsaka_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Rin Tohsaka-Generation-Instruction-Only T2I dataset
        'mahjong_rin_tohsaka_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/t2i/Rin Tohsaka_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Rin Tohsaka-Pure T2I dataset
        'mahjong_rin_tohsaka_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/t2i/Rin Tohsaka_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },

        # Mahjong Soul-Saber T2I dataset
        'mahjong_saber_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Saber/processed/t2i/Saber_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Saber-With-Thinking T2I dataset
        'mahjong_saber_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Saber/processed/t2i/Saber_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Saber-Generation-Instruction-Only T2I dataset
        'mahjong_saber_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Saber/processed/t2i/Saber_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },
        # Mahjong Soul-Saber-Pure T2I dataset
        'mahjong_saber_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-Saber/processed/t2i/Saber_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 10,
        },

        # Mahjong Soul-YuiYagi T2I dataset
        'mahjong_yuiyagi_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/t2i/YuiYagi_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mahjong Soul-YuiYagi-With-Thinking T2I dataset
        "mahjong_yuiyagi_t2i_with_thinking": {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/t2i/YuiYagi_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mahjong Soul-YuiYagi-Generation-Instruction-Only T2I dataset
        "mahjong_yuiyagi_t2i_generation_instruction_only": {
            'data_dir': 'data/ours/Mahjong Soul-YuiYagi/processed/t2i/YuiYagi_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mahjong Soul-YuiYagi-Pure T2I dataset
        'mahjong_yuiyagi_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/t2i/YuiYagi_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },

        # Mam T2I dataset
        'mam_t2i': {
            'data_dir': 'data/personalized_data/train/Mam/processed/t2i/Mam_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Mam-With-Thinking T2I dataset
        'mam_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Mam/processed/t2i/Mam_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Mam-Generation-Instruction-Only T2I dataset
        'mam_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Mam/processed/t2i/Mam_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Mam-Pure T2I dataset
        'mam_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mam/processed/t2i/Mam_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Mydieu T2I dataset
        'mydieu_t2i': {
            'data_dir': 'data/personalized_data/train/Mydieu/processed/t2i/Mydieu_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mydieu-With-Thinking T2I dataset
        'mydieu_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Mydieu/processed/t2i/Mydieu_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mydieu-Generation-Instruction-Only T2I dataset
        'mydieu_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Mydieu/processed/t2i/Mydieu_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Mydieu-Pure T2I dataset
        'mydieu_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Mydieu/processed/t2i/Mydieu_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },

        # Pokemon-Pikachu T2I dataset
        'pokemon_pikachu_t2i': {
            'data_dir': 'data/personalized_data/train/Pokemon-Pikachu/processed/t2i/Pikachu_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Pokemon-Pikachu-With-Thinking T2I dataset
        'pokemon_pikachu_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Pokemon-Pikachu/processed/t2i/Pikachu_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Pokemon-Pikachu-Generation-Instruction-Only T2I dataset
        'pokemon_pikachu_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Pokemon-Pikachu/processed/t2i/Pikachu_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Pokemon-Pikachu-Pure T2I dataset
        'pokemon_pikachu_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Pokemon-Pikachu/processed/t2i/Pikachu_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },

        # Will In Vietnam T2I dataset
        'will_in_vietnam_t2i': {
            'data_dir': 'data/personalized_data/train/Will In Vietnam/processed/t2i/Will In Vietnam_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Will In Vietnam-With-Thinking T2I dataset
        'will_in_vietnam_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Will In Vietnam/processed/t2i/Will In Vietnam_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Will In Vietnam-Generation-Instruction-Only T2I dataset
        'will_in_vietnam_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Will In Vietnam/processed/t2i/Will In Vietnam_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },
        # Will In Vietnam-Pure T2I dataset
        'will_in_vietnam_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Will In Vietnam/processed/t2i/Will In Vietnam_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 12,
        },

        # Wukong T2I dataset
        'wukong_t2i': {
            'data_dir': 'data/personalized_data/train/Wukong/processed/t2i/Wukong_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Wukong-With-Thinking T2I dataset
        'wukong_t2i_with_thinking': {
            'data_dir': 'data/personalized_data/train/Wukong/processed/t2i/Wukong_t2i_with_thinking_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },
        # Wukong-Generation-Instruction-Only T2I dataset
        'wukong_t2i_generation_instruction_only': {
            'data_dir': 'data/personalized_data/train/Wukong/processed/t2i/Wukong_t2i_generation_instruction_only_parquet',
            'num_files': 1,
            'num_total_samples': 7
        },
        # Wukong-Pure T2I dataset
        'wukong_pure_t2i': {
            'data_dir': 'data/personalized_data/train/Wukong/processed/t2i/Wukong_pure_t2i_parquet',
            'num_files': 1,
            'num_total_samples': 7,
        },

        # Xenoblade2-Pyra T2I dataset
        'xenoblade_pyra_t2i': {
            'data_dir': 'data/ours/Xenoblade2-Pyra/processed/t2i_parquet',
            'num_files': 1,
            'num_total_samples': 6,
        },
    },

    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'data/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'data/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },

    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'data/bagel_example/vlm/images',
			'jsonl_path': 'data/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
        'test_vlm': {
            'data_dir': 'data/test/vlm',
            'jsonl_path': 'data/test/vlm/test_vlm.jsonl',
            'num_total_samples': 2
        },

        # Adrien Brody VLM dataset
        'adrien_brody_vlm': {
            'data_dir': 'data',  # base data directory, because the JSONL contains the complete relative path
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_vlm.jsonl',
            'num_total_samples': 12
        },
        # Adrien Brody-Extension VLM dataset
        'adrien_brody_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_vlm_extension.jsonl',
            'num_total_samples': 188
        },
        # Adrien Brody-Thinking VLM dataset
        'adrien_brody_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Adrien Brody-Knowledge-QA VLM dataset
        'adrien_brody_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Adrien Brody-VQA VLM dataset
        'adrien_brody_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_vqa.jsonl',
            'num_total_samples': 240
        },
        # Adrien Brody-Full VLM dataset（merge all Adrien Brody's VLM data）
        'adrien_brody_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_full.jsonl',
            'num_total_samples': 543
        },

        # Bo VLM dataset
        'bo_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_vlm.jsonl',
            'num_total_samples': 12
        },
        # Bo-Extension VLM dataset
        'bo_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Bo-Thinking VLM dataset
        'bo_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Bo-Knowledge-QA VLM dataset
        'bo_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Bo-VQA VLM dataset
        'bo_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_vqa.jsonl',
            'num_total_samples': 240
        },
        # Bo-Full VLM dataset（merge all Bo's VLM data）
        'bo_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_full.jsonl',
            'num_total_samples': 544
        },
        # Bo-Full-New VLM dataset（add black image for no image samples and add <image> in human message）
        'bo_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Bo/processed/vlm/Bo_full_new.jsonl',
            'num_total_samples': 544
        },

        # Butin VLM dataset
        'butin_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_vlm.jsonl',
            'num_total_samples': 7
        },
        # Butin-Extension VLM dataset
        'butin_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_vlm_extension.jsonl',
            'num_total_samples': 187
        },
        # Butin-Thinking VLM dataset
        'butin_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_vlm_thinking.jsonl',
            'num_total_samples': 7
        },
        # Butin-Knowledge-QA VLM dataset
        'butin_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Butin-VQA VLM dataset
        'butin_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_vqa.jsonl',
            'num_total_samples': 140
        },
        # Butin-Full VLM dataset（merge all Butin's VLM data）
        'butin_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_full.jsonl',
            'num_total_samples': 434
        },

        # Coco VLM dataset
        'coco_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_vlm.jsonl',
            'num_total_samples': 11
        },
        # Coco-Extension VLM dataset
        'coco_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_vlm_extension.jsonl',
            'num_total_samples': 191
        },
        # Coco-Thinking VLM dataset
        'coco_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_vlm_thinking.jsonl',
            'num_total_samples': 11
        },
        # Coco-Knowledge-QA VLM dataset
        'coco_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Coco-VQA VLM dataset
        'coco_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_vqa.jsonl',
            'num_total_samples': 220
        },
        # Coco-Full VLM dataset（merge all Coco's VLM data）
        'coco_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_full.jsonl',
            'num_total_samples': 522
        },

        # Friends-Chandler VLM dataset
        'friends_chandler_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Chandler_vlm.jsonl',
            'num_total_samples': 12
        },
        # Friends-Chandler-Extension VLM dataset
        'friends_chandler_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Chandler_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Friends-Chandler-Thinking VLM dataset
        'friends_chandler_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Chandler_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Friends-Chandler-Knowledge-QA VLM dataset
        'friends_chandler_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Chandler_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Friends-Chandler-VQA VLM dataset
        'friends_chandler_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Chandler_vqa.jsonl',
            'num_total_samples': 239
        },
        # Friends-Chandler-Full VLM dataset（merge all Friends-Chandler's VLM data）
        'friends_chandler_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Friends-Chandler_full.jsonl',
            'num_total_samples': 543
        },

        # Friends-Joey VLM dataset
        'friends_joey_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Joey_vlm.jsonl',
            'num_total_samples': 11
        },
        # Friends-Joey-Extension VLM dataset
        'friends_joey_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Joey_vlm_extension.jsonl',
            'num_total_samples': 191
        },
        # Friends-Joey-Thinking VLM dataset
        'friends_joey_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Joey_vlm_thinking.jsonl',
            'num_total_samples': 11
        },
        # Friends-Joey-Knowledge-QA VLM dataset
        'friends_joey_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Joey_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Friends-Joey-VQA VLM dataset
        'friends_joey_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Joey_vqa.jsonl',
            'num_total_samples': 219
        },
        # Friends-Joey-Full VLM dataset（merge all Friends-Joey's VLM data）
        'friends_joey_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Friends-Joey_full.jsonl',
            'num_total_samples': 521
        },

        # Gao Qiqiang VLM dataset
        'gao_qiqiang_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_vlm.jsonl',
            'num_total_samples': 12
        },
        # Gao Qiqiang-Extension VLM dataset
        'gao_qiqiang_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Gao Qiqiang-Thinking VLM dataset
        'gao_qiqiang_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Gao Qiqiang-Knowledge-QA VLM dataset
        'gao_qiqiang_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Gao Qiqiang-VQA VLM dataset
        'gao_qiqiang_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_vqa.jsonl',
            'num_total_samples': 240
        },
        # Gao Qiqiang-Full VLM dataset（merge all Gao Qiqiang's VLM data）
        'gao_qiqiang_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_full.jsonl',
            'num_total_samples': 544
        },

        # Genshin-Furina VLM dataset
        'genshin_furina_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Furina_vlm.jsonl',
            'num_total_samples': 9
        },
        # Genshin-Furina-Extension VLM dataset
        'genshin_furina_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Furina_vlm_extension.jsonl',
            'num_total_samples': 189
        },
        # Genshin-Furina-Thinking VLM dataset
        'genshin_furina_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Furina_vlm_thinking.jsonl',
            'num_total_samples': 9
        },
        # Genshin-Furina-Knowledge-QA VLM dataset   
        'genshin_furina_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Furina_knowledge_qa.jsonl',
            'num_total_samples': 100
        }, 
        # Genshin-Furina-VQA VLM dataset
        'genshin_furina_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Furina_vqa.jsonl',
            'num_total_samples': 180
        },
        # Genshin-Furina-Full VLM dataset（merge all Genshin-Furina's VLM data）
        'genshin_furina_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Genshin-Furina_full.jsonl',
            'num_total_samples': 478
        },

        # Harry Potter-Hermione VLM dataset
        'harry_potter_hermione_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Hermione_vlm.jsonl',
            'num_total_samples': 12
        },
        # Harry Potter-Hermione-Extension VLM dataset
        'harry_potter_hermione_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Hermione_vlm_extension.jsonl',
            'num_total_samples': 191
        },
        # Harry Potter-Hermione-Thinking VLM dataset
        'harry_potter_hermione_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Hermione_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Harry Potter-Hermione-Knowledge-QA VLM dataset
        'harry_potter_hermione_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Hermione_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Harry Potter-Hermione-VQA VLM dataset
        'harry_potter_hermione_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Hermione_vqa.jsonl',
            'num_total_samples': 228
        },
        # Harry Potter-Hermione-Full VLM dataset（merge all Harry Potter-Hermione's VLM data）
        'harry_potter_hermione_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Harry Potter-Hermione_full.jsonl',
            'num_total_samples': 531
        },

        # Leonardo VLM dataset
        'leonardo_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_vlm.jsonl',
            'num_total_samples': 12
        },
        # Leonardo-Extension VLM dataset
        'leonardo_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Leonardo-Thinking VLM dataset
        'leonardo_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Leonardo-Knowledge-QA VLM dataset
        'leonardo_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Leonardo-VQA VLM dataset
        'leonardo_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_vqa.jsonl',
            'num_total_samples': 240
        },
        # Leonardo-Full VLM dataset（merge all Leonardo's VLM data）
        'leonardo_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_full.jsonl',
            'num_total_samples': 544
        },

        # Mahjong Soul-Ichihime VLM dataset
        'mahjong_ichihime_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Ichihime_vlm.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Ichihime-Extension VLM dataset
        'mahjong_ichihime_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Ichihime_vlm_extension.jsonl',
            'num_total_samples': 190
        },
        # Mahjong Soul-Ichihime-Thinking VLM dataset
        'mahjong_ichihime_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Ichihime_vlm_thinking.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Ichihime-Knowledge-QA VLM dataset
        'mahjong_ichihime_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Ichihime_knowledge_qa.jsonl',
            'num_total_samples': 50
        },
        # Mahjong Soul-Ichihime-VQA VLM dataset
        'mahjong_ichihime_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Ichihime_vqa.jsonl',
            'num_total_samples': 200
        },
        # Mahjong Soul-Ichihime-Full VLM dataset（merge all Mahjong Soul-Ichihime's VLM data）
        'mahjong_ichihime_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Mahjong Soul-Ichihime_full.jsonl',
            'num_total_samples': 450
        },

        # Mahjong Soul-Miki Nikaidou VLM dataset
        'mahjong_miki_nikaidou_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Miki Nikaidou_vlm.jsonl',
            'num_total_samples': 6
        },
        # Mahjong Soul-Miki Nikaidou-Extension VLM dataset
        'mahjong_miki_nikaidou_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Miki Nikaidou_vlm_extension.jsonl',
            'num_total_samples': 186
        },
        # Mahjong Soul-Miki Nikaidou-Thinking VLM dataset
        'mahjong_miki_nikaidou_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Miki Nikaidou_vlm_thinking.jsonl',
            'num_total_samples': 6
        },
        # Mahjong Soul-Miki Nikaidou-Knowledge-QA VLM dataset
        'mahjong_miki_nikaidou_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Miki Nikaidou_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Mahjong Soul-Miki Nikaidou-VQA VLM dataset
        'mahjong_miki_nikaidou_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Miki Nikaidou_vqa.jsonl',
            'num_total_samples': 120
        },
        # Mahjong Soul-Miki Nikaidou-Full VLM dataset（merge all Mahjong Soul-Miki Nikaidou's VLM data）
        'mahjong_miki_nikaidou_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Mahjong Soul-Miki Nikaidou_full.jsonl',
            'num_total_samples': 412
        },

        # Mahjong Soul-Rin Tohsaka VLM dataset
        'mahjong_rin_tohsaka_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Rin Tohsaka_vlm.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Rin Tohsaka-Extension VLM dataset
        'mahjong_rin_tohsaka_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Rin Tohsaka_vlm_extension.jsonl',
            'num_total_samples': 190
        },
        # Mahjong Soul-Rin Tohsaka-Thinking VLM dataset
        'mahjong_rin_tohsaka_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Rin Tohsaka_vlm_thinking.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Rin Tohsaka-Knowledge-QA VLM dataset
        'mahjong_rin_tohsaka_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Rin Tohsaka_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Mahjong Soul-Rin Tohsaka-VQA VLM dataset
        'mahjong_rin_tohsaka_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Rin Tohsaka_vqa.jsonl',
            'num_total_samples': 183
        },
        # Mahjong Soul-Rin Tohsaka-Full VLM dataset（merge all Mahjong Soul-Rin Tohsaka's VLM data）
        'mahjong_rin_tohsaka_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Mahjong Soul-Rin Tohsaka_full.jsonl',
            'num_total_samples': 483
        },

        # Mahjong Soul-Saber VLM dataset
        'mahjong_saber_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Saber_vlm.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Saber-Extension VLM dataset
        'mahjong_saber_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Saber_vlm_extension.jsonl',
            'num_total_samples': 190
        },
        # Mahjong Soul-Saber-Thinking VLM dataset
        'mahjong_saber_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Saber_vlm_thinking.jsonl',
            'num_total_samples': 10
        },
        # Mahjong Soul-Saber-Knowledge-QA VLM dataset
        'mahjong_saber_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Saber_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Mahjong Soul-Saber-VQA VLM dataset
        'mahjong_saber_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Saber_vqa.jsonl',
            'num_total_samples': 200
        },
        # Mahjong Soul-Saber-Full VLM dataset（merge all Mahjong Soul-Saber's VLM data）
        'mahjong_saber_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Mahjong Soul-Saber_full.jsonl',
            'num_total_samples': 500
        },

        # Mahjong Soul-YuiYagi VLM dataset
        'mahjong_yuiyagi_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/YuiYagi_vlm.jsonl',
            'num_total_samples': 7
        },
        # Mahjong Soul-YuiYagi-Extension VLM dataset
        'mahjong_yuiyagi_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/YuiYagi_vlm_extension.jsonl',
            'num_total_samples': 187
        },
        # Mahjong Soul-YuiYagi-Thinking VLM dataset
        'mahjong_yuiyagi_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/YuiYagi_vlm_thinking.jsonl',
            'num_files': 1,
            'num_total_samples': 7
        },
        # Mahjong Soul-YuiYagi-Knowledge-QA VLM dataset
        'mahjong_yuiyagi_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/YuiYagi_knowledge_qa.jsonl',
            'num_total_samples': 81
        },
        # Mahjong Soul-YuiYagi-VQA VLM dataset
        'mahjong_yuiyagi_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/YuiYagi_vqa.jsonl',
            'num_total_samples': 140
        },
        # Mahjong Soul-YuiYagi-Full VLM dataset（merge all Mahjong Soul-YuiYagi's VLM data）
        'mahjong_yuiyagi_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/Mahjong Soul-YuiYagi_full.jsonl',
            'num_total_samples': 415
        },

        # Mam VLM dataset
        'mam_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_vlm.jsonl',
            'num_total_samples': 12
        },
        # Mam-Extension VLM dataset
        'mam_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Mam-Thinking VLM dataset
        'mam_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Mam-Knowledge-QA VLM dataset
        'mam_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Mam-VQA VLM dataset
        'mam_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_vqa.jsonl',
            'num_total_samples': 238
        },
        # Mam-Full VLM dataset（merge all Mam's VLM data）
        'mam_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_full.jsonl',
            'num_total_samples': 542
        },

        # Mydieu VLM dataset
        'mydieu_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_vlm.jsonl',
            'num_total_samples': 7
        },
        # Mydieu-Extension VLM dataset
        'mydieu_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_vlm_extension.jsonl',
            'num_total_samples': 187
        },
        # Mydieu-Thinking VLM dataset
        'mydieu_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_vlm_thinking.jsonl',
            'num_total_samples': 7
        },
        # Mydieu-Knowledge-QA VLM dataset
        'mydieu_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Mydieu-VQA VLM dataset
        'mydieu_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_vqa.jsonl',
            'num_total_samples': 140
        },
        # Mydieu-Full VLM dataset（merge all Mydieu's VLM data）
        'mydieu_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_full.jsonl',
            'num_total_samples': 434
        },

        # Pokemon-Pikachu VLM dataset
        'pokemon_pikachu_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pikachu_vlm.jsonl',
            'num_total_samples': 7
        },
        # Pokemon-Pikachu-Extension VLM dataset
        'pokemon_pikachu_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pikachu_vlm_extension.jsonl',
            'num_total_samples': 187
        },
        # Pokemon-Pikachu-Thinking VLM dataset
        'pokemon_pikachu_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pikachu_vlm_thinking.jsonl',
            'num_total_samples': 7
        },
        # Pokemon-Pikachu-Knowledge-QA VLM dataset
        'pokemon_pikachu_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pikachu_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Pokemon-Pikachu-VQA VLM dataset
        'pokemon_pikachu_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pikachu_vqa.jsonl',
            'num_total_samples': 139
        },
        # Pokemon-Pikachu-Full VLM dataset（merge all Pokemon-Pikachu's VLM data）
        'pokemon_pikachu_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pokemon-Pikachu_full.jsonl',
            'num_total_samples': 433
        },

        # Will In Vietnam VLM dataset
        'will_in_vietnam_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_vlm.jsonl',
            'num_total_samples': 12
        },
        # Will In Vietnam-Extension VLM dataset
        'will_in_vietnam_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_vlm_extension.jsonl',
            'num_total_samples': 192
        },
        # Will In Vietnam-Thinking VLM dataset
        'will_in_vietnam_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_vlm_thinking.jsonl',
            'num_total_samples': 12
        },
        # Will In Vietnam-Knowledge-QA VLM dataset
        'will_in_vietnam_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Will In Vietnam-VQA VLM dataset
        'will_in_vietnam_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_vqa.jsonl',
            'num_total_samples': 239
        },
        # Will In Vietnam-Full VLM dataset（merge all Will In Vietnam's VLM data）
        'will_in_vietnam_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_full.jsonl',
            'num_total_samples': 543
        },

        # Wukong VLM dataset
        'wukong_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_vlm.jsonl',
            'num_total_samples': 7
        },
        # Wukong-Extension VLM dataset
        'wukong_vlm_extension': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_vlm_extension.jsonl',
            'num_total_samples': 187
        },
        # Wukong-Thinking VLM dataset
        'wukong_vlm_thinking': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_vlm_thinking.jsonl',
            'num_total_samples': 7
        },
        # Wukong-Knowledge-QA VLM dataset
        'wukong_knowledge_qa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_knowledge_qa.jsonl',
            'num_total_samples': 100
        },
        # Wukong-VQA VLM dataset
        'wukong_vqa': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_vqa.jsonl',
            'num_total_samples': 139
        },
        # Wukong-Full VLM dataset（merge all Wukong's VLM data）
        'wukong_full': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_full.jsonl',
            'num_total_samples': 433
        },

        # Xenoblade2-Pyra VLM dataset
        'xenoblade_pyra_vlm': {
            'data_dir': 'data',
            'jsonl_path': 'data/ours/Xenoblade2-Pyra/processed/vlm/xenoblade_pyra_vlm.jsonl',
            'num_total_samples': 6
        },

        'adrien_brody_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Adrien Brody/processed/vlm/Adrien Brody_full_new.jsonl',
            'num_total_samples': 543
        },
        'butin_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Butin/processed/vlm/Butin_full_new.jsonl',
            'num_total_samples': 434
        },
        'coco_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Coco/processed/vlm/Coco_full_new.jsonl',
            'num_total_samples': 522
        },
        'friends_chandler_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Chandler/processed/vlm/Friends-Chandler_full_new.jsonl',
            'num_total_samples': 543
        },
        'friends_joey_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Friends-Joey/processed/vlm/Friends-Joey_full_new.jsonl',
            'num_total_samples': 521
        },
        'gao_qiqiang_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Gao Qiqiang/processed/vlm/Gao Qiqiang_full_new.jsonl',
            'num_total_samples': 544
        },
        'genshin_furina_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Genshin-Furina/processed/vlm/Genshin-Furina_full_new.jsonl',
            'num_total_samples': 478
        },
        'harry_potter_hermione_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Harry Potter-Hermione/processed/vlm/Harry Potter-Hermione_full_new.jsonl',
            'num_total_samples': 531
        },
        'leonardo_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Leonardo/processed/vlm/Leonardo_full_new.jsonl',
            'num_total_samples': 544
        },
        'mahjong_soul_ichihime_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Ichihime/processed/vlm/Mahjong Soul-Ichihime_full_new.jsonl',
            'num_total_samples': 450
        },
        'mahjong_soul_miki_nikaidou_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Miki Nikaidou/processed/vlm/Mahjong Soul-Miki Nikaidou_full_new.jsonl',
            'num_total_samples': 412
        },
        'mahjong_soul_rin_tohsaka_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Rin Tohsaka/processed/vlm/Mahjong Soul-Rin Tohsaka_full_new.jsonl',
            'num_total_samples': 483
        },
        'mahjong_soul_saber_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-Saber/processed/vlm/Mahjong Soul-Saber_full_new.jsonl',
            'num_total_samples': 500
        },
        'mahjong_soul_yuiyagi_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mahjong Soul-YuiYagi/processed/vlm/Mahjong Soul-YuiYagi_full_new.jsonl',
            'num_total_samples': 415
        },
        'mam_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mam/processed/vlm/Mam_full_new.jsonl',
            'num_total_samples': 542
        },
        'mydieu_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Mydieu/processed/vlm/Mydieu_full_new.jsonl',
            'num_total_samples': 434
        },
        'pokemon_pikachu_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Pokemon-Pikachu/processed/vlm/Pokemon-Pikachu_full_new.jsonl',
            'num_total_samples': 433
        },
        'will_in_vietnam_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Will In Vietnam/processed/vlm/Will In Vietnam_full_new.jsonl',
            'num_total_samples': 543
        },
        'wukong_full_new': {
            'data_dir': 'data',
            'jsonl_path': 'data/personalized_data/train/Wukong/processed/vlm/Wukong_full_new.jsonl',
            'num_total_samples': 433
        },
},
}