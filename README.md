# aerial-segmentation
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

## Example notebooks

A google colab notebook demonstrating use of a detectron2 model for segmentation can be viewed at:
https://colab.research.google.com/github/Sydney-Informatics-Hub/aerial-segmentation/blob/main/notebooks/detectron2_fine_tuning.ipynb

