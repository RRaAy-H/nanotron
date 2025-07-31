# Dataset Composition and Distribution

This document outlines the multimodal dataset composition with specific data sampling strategies for training.

## Overview: Data Split by Modality

| Modality | Percentage | 
|----------|------------|
| Image | 34.4% | 
| Video | 33.0% | 
| Text | 20.2% | 
| Multi-image | 12.3% |

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

| Dataset | Percentage | Format | Path |
|---------|------------|--------|------|
| m4-instruct-data/ | 10.4% | ZIP | `/data1/yihao/M4-Instruct-Data` |
| mammoth/multi_image_data/shard_1.tar.gz | 1.9% | TAR.GZ | `/data1/yihao/MammoTH-VL_Instruct-12M/multi_image_data` |

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
| ShareGPT4Video/all | 0.8% | Video | `/data1/yihao/ShareGPTVideo/train_300k` | Direct sampling |

## Sampling Notes

- **Direct sampling**: Use original dataset as specified
- **Alternative sampling**: Substitute with specified alternatives when original is unavailable
- **Composite sampling**: Combine multiple sources with defined proportions
- **Deduplication**: All sampling strategies require avoiding duplicate samples across datasets