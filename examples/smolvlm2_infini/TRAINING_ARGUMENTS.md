# Training Arguments for SmolVLM2 Infini-Attention

## Currently Implemented Arguments

### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--model_name_or_path` | str | Path to pretrained model or HuggingFace model ID |
| `--data_mixture` | str | Path to data mixture YAML file |
| `--output_dir` | str | Directory where model checkpoints will be saved |

### Model Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--trust_remote_code` | True | Allow loading custom model code from HuggingFace |
| `--use_infini_attention` | True | Replace standard attention with infini-attention layers |
| `--segment_length` | 512 | Segment size for infini-attention memory mechanism |

### Data Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--max_seq_length` | 2048 | Maximum token sequence length |
| `--train_data_path` | "data/datasets_nanotron/*_nanotron.json" | Path to training data JSON files (uses glob pattern for multiple nanotron files) |
| `--eval_data_path` | None | Path to evaluation data JSON file for validation during training |
| `--image_dir` | "./images" | Base directory for images (NOT USED in actual pipeline - paths are absolute) |

### Training Configuration
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--per_device_train_batch_size` | 8 | Training batch size per GPU | ✅ Fully implemented |
| `--per_device_eval_batch_size` | 8 | Evaluation batch size per GPU | ✅ Fully implemented |
| `--num_train_epochs` | 3.0 | Number of training epochs | ✅ Fully implemented |
| `--max_steps` | -1 | Maximum training steps (overrides epochs if > 0) | ✅ Fully implemented (NEW) |
| `--learning_rate` | 5e-5 | Initial learning rate | ✅ Fully implemented |
| `--weight_decay` | 0.01 | L2 regularization weight | ✅ Fully implemented |
| `--warmup_steps` | 0 | Number of warmup steps for learning rate scheduler | ✅ Fully implemented |
| `--lr_scheduler_type` | "linear" | Type of learning rate scheduler ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup") | ✅ Fully implemented |
| `--max_grad_norm` | 1.0 | Maximum gradient norm for clipping (0 = no clipping) | ✅ Fully implemented |
| `--gradient_accumulation_steps` | 1 | Number of steps to accumulate gradients | ❌ NOT implemented |

### Checkpointing & Evaluation
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--save_steps` | 500 | Save checkpoint every N steps | ✅ Fully implemented |
| `--save_strategy` | "steps" | When to save checkpoints ("steps", "epoch", "no") | ✅ Fully implemented |
| `--eval_steps` | 500 | Run evaluation every N steps during training | ✅ Fully implemented |
| `--do_train` | False | Whether to run training | ✅ Fully implemented |
| `--do_eval` | False | Whether to run evaluation after training | ✅ Fully implemented |

### Performance & Precision
| Argument | Default | Description | Implementation Status |
|----------|---------|-------------|----------------------|
| `--bf16` | False | Use bfloat16 precision (saves memory, faster training) | ✅ Fully implemented |
| `--fp16` | False | Use float16 precision | ❌ NOT implemented |
| `--gradient_checkpointing` | False | Trade compute for memory by recomputing activations | ✅ Fully implemented |

## Usage Examples

### Minimal Command (Required Arguments Only)
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "configs/data_mixture.yaml" \
    --output_dir "./output"
```

### Training with Custom Batch Size and BF16
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "configs/data_mixture.yaml" \
    --output_dir "./output" \
    --train_data_path "data/datasets_nanotron/*_nanotron.json" \
    --per_device_train_batch_size 4 \
    --bf16 True \
    --do_train
```

### Training with Evaluation and Max Steps
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "configs/data_mixture.yaml" \
    --output_dir "./output" \
    --train_data_path "data/datasets_nanotron/*_nanotron.json" \
    --eval_data_path "data/datasets_nanotron/eval_nanotron.json" \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --eval_steps 100 \
    --save_steps 200 \
    --bf16 True \
    --gradient_checkpointing True \
    --do_train \
    --do_eval
```

### Full Training Configuration
```bash
python train_smolvlm2_infini.py \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-1.7B-Instruct" \
    --data_mixture "configs/data_mixture.yaml" \
    --output_dir "./checkpoints" \
    --train_data_path "data/datasets_nanotron/*_nanotron.json" \
    --eval_data_path "data/datasets_nanotron/eval_nanotron.json" \
    --use_infini_attention True \
    --segment_length 512 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --max_steps 5000 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --save_strategy "steps" \
    --eval_steps 250 \
    --bf16 True \
    --gradient_checkpointing True \
    --do_train \
    --do_eval
```

### GPU Selection
To select specific GPUs, use the `CUDA_VISIBLE_DEVICES` environment variable:
```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python train_smolvlm2_infini.py ...

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python train_smolvlm2_infini.py ...

# Use GPUs 0 and 1 (multi-GPU not fully supported in current implementation)
CUDA_VISIBLE_DEVICES=0,1 python train_smolvlm2_infini.py ...
```

## Notes

1. **Data Pipeline**: The actual training pipeline uses pre-processed `*_nanotron.json` files that contain absolute paths to images/videos in `/data1/yihao/`. The `--image_dir` argument is not used in practice.

2. **Precision**: If `--bf16` is not set, training uses full float32 precision which requires more memory.

3. **Max Steps vs Epochs**: When `--max_steps` is set and > 0, it overrides `--num_train_epochs`. The training will stop at the specified number of steps regardless of epochs.

4. **Not Implemented**: 
   - `--gradient_accumulation_steps`: Currently updates weights every batch
   - `--fp16`: Only bf16 is implemented, not fp16
   - Multi-GPU training: Current implementation is for single GPU only

5. **Memory Optimization**: For limited GPU memory, use:
   - `--bf16 True` (reduces memory by ~50%)
   - `--gradient_checkpointing True` (trades compute for memory)
   - Smaller `--per_device_train_batch_size`