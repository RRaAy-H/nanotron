import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union, List
# Update imports to use available classes
from transformers.models.idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer, Idefics3PreTrainedModel

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
        
        # Use Idefics3VisionModel directly (contains SigLIP) - same as SmolVLM2
        self.vision_model = Idefics3VisionTransformer(config.vision_config)
        
        # Tensor parallel connector for vision-language fusion
        self.connector = self._build_tensor_parallel_connector(config)
        
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
        
        # Image token and sequence length (from Idefics3 config)
        self.image_token_id = getattr(config, 'image_token_id', 32000)
        self.image_seq_len = config.perceiver_config.resampler_n_latents
    
    def _build_tensor_parallel_connector(self, config):
        """Build tensor parallel version of connector for Idefics3"""
        class TensorParallelIdefics3Connector(nn.Module):
            def __init__(self, config, parallel_context):
                super().__init__()
                # Use the perceiver directly from idefics3.modeling_idefics3
                from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector
                self.perceiver = Idefics3Connector(config.perceiver_config)
                
                # Replace the projection with tensor parallel version
                self.modality_projection = TensorParallelRowLinear(
                    in_features=config.perceiver_config.resampler_head_dim,
                    out_features=config.text_config.hidden_size,
                    pg=parallel_context.tp_pg,
                    bias=False
                )
        
            def forward(self, image_hidden_states, attention_mask=None):
                image_hidden_states = self.perceiver(
                    context=image_hidden_states,
                    attention_mask=attention_mask
                )
                image_hidden_states = self.modality_projection(image_hidden_states)
                return image_hidden_states
    
        return TensorParallelIdefics3Connector(config, self.parallel_context)
    
    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge text embeddings with image embeddings - exact same logic as SmolVLM2
        """
        
        # Basic shape checks
        B, T, D_text = inputs_embeds.shape
        N, S, D_img  = image_hidden_states.shape
        if D_text != D_img:
            raise ValueError(
                f"Text embedding dim {D_text} != image embedding dim {D_img}"
            )
    
        # Track how many images we've used so far across the entire batch
        image_offset = 0
    
        # Store one merged tensor per batch sample
        merged_outputs: List[torch.Tensor] = []
    
        # Iterate through each sample
        for b_idx, (cur_ids, cur_embeds) in enumerate(zip(input_ids, inputs_embeds)):
            # Find positions of <image> tokens in the text
            image_positions = (cur_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_positions)
    
            # If no <image> => text-only
            if num_image_tokens == 0:
                # We do not consume any row from image_hidden_states; 
                # but we do a zero-length slice so the image encoder is in the graph.
                empty_slice = image_hidden_states[0][:0, :]  # shape (0, D)
                # Concatenate text plus that empty slice.
                merged_text_only = torch.cat([cur_embeds, empty_slice], dim=0)
                merged_outputs.append(merged_text_only)
                continue
    
            # Otherwise, we have at least one <image> token.
            if num_image_tokens % S != 0:
                raise ValueError(
                    f"Sample {b_idx} has {num_image_tokens} <image> tokens, not a multiple of S={S}. "
                    "Cannot map them to blocks of shape (S, D)."
                )
    
            # Chunk image_positions into groups of size S
            positions_list = image_positions.tolist()
            chunks = [
                positions_list[i : i + S]
                for i in range(0, num_image_tokens, S)
            ]
    
            # Build segments: text, then image row(s), text, etc.
            segments = []
            text_start = 0
    
            # For each chunk (each chunk => 1 image)
            for chunk in chunks:
                # image_hidden_states[image_offset] => shape (S, D)
                cur_block = image_hidden_states[image_offset]
                image_offset += 1
    
                # Iterate over the S positions in ascending order
                for i_s, pos in enumerate(chunk):
                    # Add text from [text_start..pos)
                    if pos > text_start:
                        segments.append(cur_embeds[text_start:pos])
                    # Then add one row from cur_block => shape (1, D)
                    row_of_block = cur_block[i_s : i_s + 1, :]
                    segments.append(row_of_block)
                    # skip the <image> token
                    text_start = pos + 1
    
            # leftover text after the final <image> token
            if text_start < T:
                segments.append(cur_embeds[text_start:])
    
            # cat them into a single (T_b, D) tensor
            merged_sample = torch.cat(segments, dim=0)
            merged_outputs.append(merged_sample)
            
        merged_outputs = torch.stack(merged_outputs)
        return merged_outputs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multimodal processing - matches SmolVLM2 logic"""
        
        # Get text embeddings
        inputs_embeds = self.text_model.token_position_embeddings.pp_block.token_embedding(input_ids)
        
        # START VISUAL INPUTS INTEGRATION (same as SmolVLM2)
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            
            if not any(real_images_inds):
                # no images, leave one empty image.
                real_images_inds[0] = True
                
            pixel_values = pixel_values[real_images_inds].contiguous()
            
            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder (SigLIP)
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
            
            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=inputs_embeds.dtype, device=input_ids.device)

        # Merge vision and text features if we have image states
        if inputs_embeds is not None and image_hidden_states is not None:
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )
        
        # Forward through language model with infini-attention
        outputs = self.text_model(
            input_ids=None,  # Use inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        
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
            "hidden_states": hidden_states,
            "image_hidden_states": image_hidden_states
        }
    
    def get_tied_parameters(self):
        """Return tied parameters for nanotron"""
        return []
    
    # Add this method to the SmolVLM2NanotronModel class
    def init_model_randomly(self, config):
        """Initialize the model weights randomly (required abstract method)"""
        # Initialize vision model
        if hasattr(self.vision_model, "init_weights"):
            self.vision_model.init_weights()
        
        # Initialize text model
        if hasattr(self.text_model, "init_model_randomly"):
            self.text_model.init_model_randomly(config.text_config)
        
        # Initialize connector
        if hasattr(self.connector, "apply"):
            def _init_weights(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            self.connector.apply(_init_weights)
        
        # Initialize LM head
        if hasattr(self.lm_head, "weight"):
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        return self