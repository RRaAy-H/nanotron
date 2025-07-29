import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
from transformers import Idefics3Config, Idefics3Model, Idefics3ForConditionalGeneration

from nanotron.models.base import NanotronModel
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelRowLinear
from nanotron.models.llama import LlamaModel  # For infini-attention integration

class SmolVLM2NanotronModel(NanotronModel):
    """SmolVLM2 model adapted for Nanotron with infini-attention support"""
    
    def __init__(self, config, parallel_context: ParallelContext):
        super().__init__()
        self.config = config
        self.parallel_context = parallel_context
        
        # Initialize base Idefics3 model structure
        self.vision_model = self._build_vision_model(config)
        self.connector = self._build_connector(config)
        
        # Use Nanotron's LlamaModel with infini-attention for text processing
        self.text_model = LlamaModel(
            config=config.text_config,
            parallel_context=parallel_context
        )
        
        # Language modeling head
        self.lm_head = TensorParallelColumnLinear(
            in_features=config.text_config.hidden_size,
            out_features=config.text_config.vocab_size,
            pg=parallel_context.tp_pg,
            bias=False
        )
        
        # Image token for multimodal fusion
        self.image_token_id = getattr(config, 'image_token_id', 32000)
    
    def _build_vision_model(self, config):
        """Build vision encoder based on original SmolVLM2"""
        # Use the vision config from Idefics3
        vision_config = config.vision_config
        
        # Simple vision model implementation
        # In practice, load from HuggingFace SigLIP
        from transformers import SiglipVisionModel
        return SiglipVisionModel(vision_config)
    
    def _build_connector(self, config):
        """Build vision-language connector"""
        return nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)
        )
    
    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        """Merge vision and text inputs (adapted from original SmolVLM2)"""
        if image_hidden_states is None:
            return inputs_embeds
        
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        num_images, num_patches, vision_hidden_size = image_hidden_states.shape
        
        # Find image token positions
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        
        if len(image_positions[0]) == 0:
            return inputs_embeds
        
        # Simple replacement strategy - replace image tokens with vision features
        merged_embeds = inputs_embeds.clone()
        
        for batch_idx in range(batch_size):
            batch_positions = image_positions[1][image_positions[0] == batch_idx]
            if len(batch_positions) > 0:
                # Take first num_patches positions and replace with vision features
                for i, pos in enumerate(batch_positions[:num_patches]):
                    if i < num_patches:
                        merged_embeds[batch_idx, pos] = image_hidden_states[batch_idx, i]
        
        return merged_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multimodal processing"""
        
        # Process vision inputs if provided
        image_hidden_states = None
        if pixel_values is not None:
            # Process images through vision encoder
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            vision_features = vision_outputs.last_hidden_state
            
            # Project to language model dimension
            image_hidden_states = self.connector(vision_features)
        
        # Get text embeddings
        inputs_embeds = self.text_model.token_position_embeddings.pp_block.token_embedding(input_ids)
        
        # Merge vision and text features
        if image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
        
        # Forward through language model with infini-attention
        hidden_states = self.text_model(
            input_ids=None,  # Use inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        
        # Generate logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states
        }
    
    def get_tied_parameters(self):
        """Return tied parameters for nanotron"""
        return []