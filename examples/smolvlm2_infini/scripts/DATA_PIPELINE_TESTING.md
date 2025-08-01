# SmolVLM2-Infini Data Pipeline Testing Guide

This guide explains how to validate and debug the SmolVLM2-Infini training data pipeline using the provided testing tools.

## Overview

The data pipeline consists of:
1. **Data Preparation**: `prepare_training_data.py` - Processes local datasets into training-ready JSON files
2. **Data Mixture Config**: `smolvlm2_256m_mixture.yaml` - Defines dataset composition for training
3. **SmolVLM2 Dataset Loading**: Integration with SmolVLM2's dataset classes
4. **Data Collation**: Batching and padding for training

## Testing Tools

### 1. Comprehensive Pipeline Test (`test_data_pipeline.py`)

**Purpose**: Complete validation of the entire data pipeline

**Usage**:
```bash
# Basic test
python scripts/test_data_pipeline.py

# Custom paths
python scripts/test_data_pipeline.py \
    --data_dir data/datasets \
    --mixture_path data/smolvlm2_256m_mixture.yaml \
    --num_samples 20 \
    --verbose
```

**What it tests**:
- ✅ Prepared JSON data files structure and format
- ✅ Data mixture YAML configuration validity
- ✅ SmolVLM2 dataset loading with actual processor
- ✅ Data collation and batching
- ✅ Multi-modal data handling (text, image, video, multi-image)
- ✅ Sample validation and format compliance

**Output**: 
- Detailed console logs with test results
- `data_pipeline_validation_report.json` - Detailed validation report
- `test_data_pipeline.log` - Full test log

### 2. Quick Debug Tool (`debug_data_pipeline.py`)

**Purpose**: Fast debugging of specific data issues

**Usage Examples**:

```bash
# Inspect a specific dataset
python scripts/debug_data_pipeline.py \
    --dataset data/datasets/magpie_pro_l3_80b_mt.json \
    --inspect

# Show statistics for all datasets
python scripts/debug_data_pipeline.py \
    --data_dir data/datasets \
    --stats

# Debug a specific sample
python scripts/debug_data_pipeline.py \
    --dataset data/datasets/llava_video_1_2m.json \
    --sample_id 0

# Validate mixture configuration
python scripts/debug_data_pipeline.py \
    --mixture data/smolvlm2_256m_mixture.yaml \
    --validate_mixture
```

**Features**:
- 🔍 Dataset inspection with sample structure analysis
- 📊 Statistics dashboard for all datasets
- 🐛 Individual sample debugging
- ⚙️ Mixture configuration validation
- 🚨 Common issue detection

## Step-by-Step Testing Workflow

### Step 1: Prepare Training Data

First, prepare your training data from local storage:

```bash
# Navigate to the smolvlm2_infini directory
cd examples/smolvlm2_infini

# Prepare datasets from local storage
python scripts/prepare_training_data.py \
    --output_dir data/datasets \
    --base_path /data1/yihao \
    --seed 42
```

### Step 2: Quick Validation

Check that datasets were created correctly:

```bash
# Show overview of all prepared datasets
python scripts/debug_data_pipeline.py --data_dir data/datasets --stats
```

Expected output:
```
Dataset Statistics for: data/datasets
================================================================================

┌─────────────────────────┬─────────┬───────────┬──────┬───────┬─────────────┬───────┐
│ Dataset                 │ Samples │ Size (MB) │ Text │ Image │ Multi-Image │ Video │
├─────────────────────────┼─────────┼───────────┼──────┼───────┼─────────────┼───────┤
│ magpie_pro_l3_80b_mt    │  68,000 │    45.23  │ 68,000│   0   │      0      │   0   │
│ llava_video_1_2m        │  73,000 │    89.45  │   0  │   0   │      0      │73,000 │
│ ...                     │   ...   │    ...    │ ...  │  ...  │     ...     │  ...  │
└─────────────────────────┴─────────┴───────────┴──────┴───────┴─────────────┴───────┘

📊 Summary:
  Total datasets: 32
  Total samples: 1,000,000

📈 Modality Totals:
  - image: 344,000 (34.4%)
  - video: 330,000 (33.0%)
  - text: 202,000 (20.2%)
  - multi-image: 123,000 (12.3%)
```

### Step 3: Inspect Individual Datasets

Check specific datasets for issues:

```bash
# Inspect a text dataset
python scripts/debug_data_pipeline.py \
    --dataset data/datasets/magpie_pro_l3_80b_mt.json \
    --inspect

# Inspect a video dataset
python scripts/debug_data_pipeline.py \
    --dataset data/datasets/llava_video_1_2m.json \
    --inspect
```

### Step 4: Validate Data Mixture Configuration

Ensure the mixture config is properly configured:

```bash
python scripts/debug_data_pipeline.py \
    --mixture data/smolvlm2_256m_mixture.yaml \
    --validate_mixture
```

### Step 5: Comprehensive Pipeline Test

Run the full pipeline validation:

```bash
python scripts/test_data_pipeline.py \
    --data_dir data/datasets \
    --mixture_path data/smolvlm2_256m_mixture.yaml \
    --num_samples 10 \
    --verbose
```

## Interpreting Test Results

### Success Indicators

✅ **Pipeline Ready for Training**:
```
✅ DATA PIPELINE IS READY FOR TRAINING!
All critical tests passed. The data pipeline appears to be properly configured.
```

### Common Issues and Solutions

❌ **Missing Dataset Files**:
```
❌ Dataset xyz.json not found
```
**Solution**: Re-run `prepare_training_data.py` or check file paths

⚠️ **Sample Format Issues**:
```
⚠️ Found 5 problematic samples in dataset xyz
```
**Solution**: Check sample structure - conversations should have 'from' and 'value' fields

❌ **Mixture Config Errors**:
```
❌ JSON file not found for dataset xyz
```
**Solution**: Update paths in `smolvlm2_256m_mixture.yaml` or regenerate missing datasets

⚠️ **Modality Distribution Issues**:
```
⚠️ Expected image samples: 344,000, found: 300,000
```
**Solution**: Check sampling strategies and regenerate affected datasets

## Advanced Debugging

### Debug Specific Samples

If training fails on specific samples, debug them individually:

```bash
# Find the problematic sample
python scripts/debug_data_pipeline.py \
    --dataset data/datasets/problematic_dataset.json \
    --sample_id 1234
```

### Check Data Loading with Actual Processor

Test integration with SmolVLM2's processor:

```bash
# This tests actual tokenization and image processing
python scripts/test_data_pipeline.py --verbose
```

### Memory and Performance Testing

For large datasets, monitor memory usage:

```bash
# Test with limited samples to avoid memory issues
python scripts/test_data_pipeline.py --num_samples 5
```

## Expected Dataset Structure

### Valid Sample Format

```json
{
  "id": "sample_001",
  "conversations": [
    {
      "from": "human",
      "value": "What do you see in this image?"
    },
    {
      "from": "gpt", 
      "value": "I see a beautiful landscape with mountains."
    }
  ],
  "image": "path/to/image.jpg"  // Optional: for image/video modalities
}
```

### Modality-Specific Fields

- **Text-only**: Only `id` and `conversations`
- **Image**: Add `image` field with string path or list of paths
- **Video**: Add `video` field with string path
- **Multi-image**: Add `image` field with list of paths

## Troubleshooting

### Import Errors

```bash
# Make sure SmolVLM dependencies are installed
pip install transformers>=4.35.0 pillow>=10.0.0

# Add nanotron to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nanotron/src"
```

### Path Issues

- Ensure all paths in mixture config are relative to the config file location
- Image/video paths should be relative to the base data path (`/data1/yihao`)
- Use forward slashes (/) even on Windows

### Memory Issues

- Test with smaller `--num_samples` values first
- Ensure sufficient RAM for large datasets
- Monitor disk space for temporary files

## Integration with Training

Once validation passes, you can proceed with training:

```bash
# Training with validated data pipeline
python scripts/train_smolvlm2_infini.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
    --data_mixture data/smolvlm2_256m_mixture.yaml \
    --output_dir checkpoints/smolvlm2_infini \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 1
```

## Files Generated by Testing

- `test_data_pipeline.log` - Complete test execution log
- `data_pipeline_validation_report.json` - Detailed validation results
- Console output with real-time test results and recommendations