# Dataset Composition and Distribution

This document outlines the multimodal dataset composition with specific data sampling strategies for training.

## Overview: Data Split by Modality

| Modality | Percentage | Notes |
|----------|------------|-------|
| Image | 34.4% | Unchanged |
| Video | 33.0% | Unchanged |
| Text | 20.2% | Unchanged |
| Multi-image | 12.3% | **Modified**: M4-Instruct-Data skipped, Mammoth upsampled 6.5x |

**Total Dataset Size**: ~896K samples (reduced from ~1M due to M4-Instruct-Data skip)

## Dataset Skipping and Upsampling Strategy

The current configuration demonstrates a flexible approach to dataset management:

### Skip Functionality
- **M4-Instruct-Data**: Uses `sampling_strategy: skip` in the YAML configuration
- **Benefits**: Avoids dependency issues, reduces training time, maintains pipeline integrity
- **Re-enabling**: Simply change `sampling_strategy: skip` to `sampling_strategy: all`

### Compensation Strategy  
- **Mammoth Multi-image**: Upsampled from 19K to 123K samples (6.5x increase)
- **Method**: Uses `sampling_strategy: random:650%` for balanced upsampling
- **Result**: Maintains 12.3% multi-image representation in the overall dataset

## Detailed Dataset Breakdown

### Text Datasets (20.2% total)

All text datasets are in parquet format and located at `/data1/yihao/LLaVA-OneVision-Data`.

| Dataset | Percentage | 
|---------|------------|
| llava-onevision/magpie_pro(l3_80b_mt) | 6.8% | 
| llava-onevision/magpie_pro(l3_80b_st) | 6.8% |
| llava-onevision/magpie_pro(qwen2_72b_st) | 5.8% |
| llava-onevision/mathqa | 0.9% |

### Multi-image Datasets (12.3% total)

| Dataset | Percentage | Format | Path | Status |
|---------|------------|--------|------|--------|
| m4-instruct-data/ | 0% (SKIPPED) | ZIP | `/data1/yihao/M4-Instruct-Data` | **Configurable** - set `sampling_strategy: all` to re-enable |
| mammoth/multi_image_data/shard_1.tar.gz | 12.3% (UPSAMPLED) | TAR.GZ | `/data1/yihao/MAmmoTH-VL-Instruct-12M/multi_image_data` | **Compensates for M4-Instruct-Data skip** (was 1.9%, now 6.5x upsampled) |

**Note**: M4-Instruct-Data is currently skipped using `sampling_strategy: skip` in the YAML configuration. This allows for easy re-enabling in the future while maintaining the overall dataset balance through mammoth upsampling.

### Image Datasets (34.4% total)

All image datasets are in parquet format and located at `/data1/yihao/LLaVA-OneVision-Data`.

| Dataset | Percentage | Sampling Strategy |
|---------|------------|-------------------|
| llava-onevision/other | 17.4% | **Composite sampling**: 70% from figureqa(cauldron,llava_format) + raven(cauldron), 30% from remaining image datasets (avoid duplicates) |
| llava-onevision/vision_flan(filtered) | 3.9% | Direct sampling |
| llava-onevision/mavis_math_metagen | 2.6% | Direct sampling |
| llava-onevision/mavis_math_rule_geo | 2.5% | Direct sampling |
| llava-onevision/sharegpt4o | 1.7% | Direct sampling |
| llava-onevision/sharegpt4v(coco) | 1.5% | Direct sampling |
| llava-onevision/image_textualization | 1.3% | Direct sampling |
| llava-onevision/sharegpt4v(llava) | 0.9% | Direct sampling |
| llava-onevision/MAPQA(MathV360K) | 0.9% | Direct sampling |
| llava-onevision/qa | 0.8% | **Alternative sampling**: Use figureqa or MAPQA(MathV360K) as substitute (avoid duplicates) |
| llava-onevision/textocr(gpt4v) | 0.8% | Direct sampling |

### Video Datasets (33.0% total)

| Dataset | Percentage | Format | Path | Sampling Strategy |
|---------|------------|--------|------|-------------------|
| llava-video-178k/1-2m | 7.3% | Video (MP4/MKV+) | `/data1/yihao/LLaVA-OneVision-Data` | Direct sampling |
| llava-video-178k/2-3m | 7.0% | Video (MP4/MKV+) | `/data1/yihao/LLaVA-OneVision-Data` | Direct sampling |
| other-video/combined | 5.7% | Video | **Alternative sampling** | Random sampling from llava-video datasets (avoid duplicates) |
| llava-video-178k/hound | 4.4% | Video (MP4/MKV+) | `/data1/yihao/LLaVA-OneVision-Data` | Direct sampling |
| llava-video-178k/0-30s | 2.4% | Video (MP4/MKV+) | `/data1/yihao/LLaVA-OneVision-Data` | Direct sampling |
| video-star/starb | 2.2% | Video | **Alternative sampling** | Random sampling from llava-video datasets (avoid duplicates) |
| vista-400k/combined | 2.2% | TAR | `/data1/yihao/VISTA-400K/two_needle_niah_qa` | Direct sampling |
| vript/long | 1.0% | Video | **Alternative sampling** | Random sampling from vista-400k/combined (avoid duplicates) |
| ShareGPT4Video/all | 0.8% | Video | **Alternative sampling** | Random sampling from vista-400k/combined (avoid duplicates) |

## Sampling Notes

- **Direct sampling**: Use original dataset as specified
- **Alternative sampling**: Substitute with specified alternatives when original is unavailable
- **Composite sampling**: Combine multiple sources with defined proportions
- **Deduplication**: All sampling strategies require avoiding duplicate samples across datasets

## Dataset Structure Notes (Updated based on server analysis)

**Important Path Corrections:**
- MammoTH dataset is located at `/data1/yihao/MAmmoTH-VL-Instruct-12M` (note the capitalization)
- Video data (llava-video-178k) is located at `/data1/yihao/llava-video/` (separate from LLaVA-OneVision-Data)
- Video subdirectories follow naming pattern: `{duration}_{source}` (e.g., `1_2_m_academic_v0_1`)

**Video Directory Structure:**
```
/data1/yihao/llava-video/
├── 0_30_s_academic_v0_1/
├── 0_30_s_activitynetqa/
├── 1_2_m_academic_v0_1/
├── 2_3_m_academic_v0_1/
└── ... (other time-based subdirectories)
```

**Verified Dataset Locations:**
- ✅ LLaVA-OneVision-Data: `/data1/yihao/LLaVA-OneVision-Data`
- ✅ M4-Instruct-Data: `/data1/yihao/M4-Instruct-Data`  
- ✅ VISTA-400K: `/data1/yihao/VISTA-400K/two_needle_niah_qa`
- ✅ MAmmoTH-VL-Instruct-12M: `/data1/yihao/MAmmoTH-VL-Instruct-12M/multi_image_data`
- ✅ llava-video: `/data1/yihao/llava-video/` (various time-based subdirs)
