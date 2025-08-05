from dataclasses import dataclass
from typing import Optional
from nanotron.config.models_config import NanotronConfigs
from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
from nanotron.models.llama import LlamaConfig
from transformers import PretrainedConfig

@dataclass 
class SmolVLM2Config:
    """Configuration for SmolVLM2 with Nanotron integration"""
    
    # Text model config (based on SmolLM)
    vocab_size: int = 49152
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 16384
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    
    # Vision config
    vision_config: Optional[dict] = None
    
    # Multimodal config
    image_token_id: int = 32000
    
    # Infini-attention config
    use_infini_attention: bool = True
    segment_length: int = 512
    
    # def __post_init__(self):
    #     if self.vision_config is None:
    #         # self.vision_config = {
    #         #     "hidden_size": 768,
    #         #     "image_size": 224,
    #         #     "patch_size": 16,
    #         #     "num_hidden_layers": 12,
    #         #     "num_attention_heads": 12,
    #         #     "intermediate_size": 3072,
    #         # }
    #         from transformers import Idefics3Config

    #         self.vision_config = Idefics3Config.from_dict({
    #             "model_type": "idefics3",
    #             "vision_config": {
    #                 "hidden_size": 768,
    #                 "image_size": 224,
    #                 "patch_size": 16,
    #                 "num_hidden_layers": 12,
    #                 "num_attention_heads": 12,
    #                 "intermediate_size": 3072
    #             },
    #             "perceiver_config": {
    #                 "resampler_n_latents": 64,
    #                 "resampler_head_dim": 256
    #             }
    #         })
        
    #     # Ensure compatibility with Nanotron
    #     self.is_llama_config = True  # For Nanotron compatibility
    def __post_init__(self):
        self.vision_config = Idefics3VisionConfig(
            hidden_size=768,
            image_size=224,
            patch_size=16,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )

        self.text_config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
        )

        self.is_llama_config = True

        
# from dataclasses import dataclass
# from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
# from nanotron.models.llama import LlamaConfig

# @dataclass
# class PerceiverConfig:
#     """Minimal replacement for Idefics3PerceiverConfig."""
#     resampler_n_latents: int = 64
#     resampler_head_dim: int = 256

# @dataclass
# class SmolVLM2Config:
#     """Configuration for SmolVLM2 with Nanotron integration"""

#     # Core text config
#     vocab_size: int = 49152
#     hidden_size: int = 512
#     intermediate_size: int = 1536
#     num_hidden_layers: int = 12
#     num_attention_heads: int = 8
#     num_key_value_heads: int = 4
#     max_position_embeddings: int = 16384
#     rope_theta: float = 10000.0
#     rms_norm_eps: float = 1e-5

#     # Multimodal
#     image_token_id: int = 32000

#     # Infini-attention
#     use_infini_attention: bool = True
#     segment_length: int = 512

#     # Derived configs (set in __post_init__)
#     vision_config: Idefics3VisionConfig = None
#     perceiver_config: PerceiverConfig = None
#     text_config: LlamaConfig = None

#     def __post_init__(self):
#         # Vision encoder (SigLIP / ViT)
#         self.vision_config = Idefics3VisionConfig(
#             hidden_size=768,
#             image_size=224,
#             patch_size=16,
#             num_hidden_layers=12,
#             num_attention_heads=12,
#             intermediate_size=3072,
#         )

#         # Perceiver for resampling + projection to text dim
#         self.perceiver_config = PerceiverConfig(
#             resampler_n_latents=64,
#             resampler_head_dim=256,
#         )

#         # LLaMA-style text config with Infini-Attention
#         self.text_config = LlamaConfig(
#             vocab_size=self.vocab_size,
#             hidden_size=self.hidden_size,
#             intermediate_size=self.intermediate_size,
#             num_hidden_layers=self.num_hidden_layers,
#             num_attention_heads=self.num_attention_heads,
#             num_key_value_heads=self.num_key_value_heads,
#             max_position_embeddings=self.max_position_embeddings,
#             rope_theta=self.rope_theta,
#             rms_norm_eps=self.rms_norm_eps,
#             use_infini_attention=self.use_infini_attention,
#             segment_length=self.segment_length,
#         )

#         self.is_llama_config = True  # For Nanotron compatibility
