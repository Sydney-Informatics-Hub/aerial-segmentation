# aerial-segmentation
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SIH/building-segmentation)
Open source aerial imagery segmentation model fine tuning, evaluation, and prediction tools. Part of https://github.com/Sydney-Informatics-Hub/PIPE-3956-aerial-segmentation

## Setup

### Local (or interactive VM)

```bash
conda create -n aerial-segmentation python==3.9

conda activate aerial-segmentation

pip install -r requirements.txt

pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Input Data Format

Images and annotations in COCO JSON.

## Output Data Format

Images and annotations in COCO JSON.
