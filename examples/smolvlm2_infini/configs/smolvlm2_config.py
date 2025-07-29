from dataclasses import dataclass
from typing import Optional
from nanotron.config.models_config import NanotronConfigs

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
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = {
                "hidden_size": 768,
                "image_size": 224,
                "patch_size": 16,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
            }
        
        # Ensure compatibility with Nanotron
        self.is_llama_config = True  # For Nanotron compatibility