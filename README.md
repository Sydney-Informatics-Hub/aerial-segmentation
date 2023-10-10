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

Jupyter notebooks demonstrating use of a detectron2 model for segmentation are in the `notebooks` dir.

The google colab notebook can be run [here](https://colab.research.google.com/github/Sydney-Informatics-Hub/aerial-segmentation/blob/main/notebooks/detectron2_fine_tuning_colab.ipynb)


## Dataset

A toy dataset has been uploaded to Roboflow. It is a small subset, containing Chatswood region, available [here](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200).

There are multiple versions of this dataset. Please ignore the first version. Version 2 and later versions are the ones that are being used. The main difference of version 2 and 3 is that [version 2](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/2) contains 90 degree augmentaions, while [version 3](https://universe.roboflow.com/sih-vpfnf/gis-hd-200x200/3) does not.

For implementing this in your code, you can use the following code snippet:

```python
from roboflow import Roboflow
 
rf = Roboflow(api_key= 'your_roboflow_api_key' )
workspace_name = "sih-vpfnf" 
dataset_version = 3 
project_name = "gis-hd-200x200" 
dataset_download_name = "coco-segmentation" 

project = rf.workspace(workspace_name).project(project_name)
dataset = project.version(dataset_version).download(dataset_download_name)
```
<!-- 
# Register the dataset
from detectron2.data.datasets import register_coco_instances
dataset_name = "chatswood-dataset" #@param {type:"string"}
dataset_folder = "gis-hd-200x200" #@param {type:"string"}
register_coco_instances(f"{dataset_name}_train", {}, f"{dataset_folder}/train/_annotations.coco.json", f"/content/{dataset_folder}/train/")
register_coco_instances(f"{dataset_name}_val", {}, f"{dataset_folder}/valid/_annotations.coco.json", f"/content/{dataset_folder}/valid/")
register_coco_instances(f"{dataset_name}_test", {}, f"{dataset_folder}/test/_annotations.coco.json", f"/content/{dataset_folder}/test/")

# Use the dataset
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
cfg.DATASETS.TEST = (f"{dataset_name}_test",)
# then do the other configs

``` -->

## Usage

### Fine-tuning

TBC

### Prediction

The following code snippet can be used to predict on a directory of tiles (batch prediction):


```bash
python -m scripts.prediction_batch_detectron2 -i "path/to/tiles" -c "path/to/config.yml" -w "path/to/weights/model.pth" --coco "path/to/coco.json" --simplify-tolerance 0.3  threshold 0.7 --force-cpu 

```

For more information about the script, you may run:

```bash
python scripts.prediction_batch_detectron2  --help

```

For prediction and visualisation on a single image, you can use the following script:

```bash
python -m scripts.scripts/prediction_detectron2 --image "path/to/image" --config "path/to/config.yml" --weights "path/to/weights/model.pth" --threshold 0.7 --coco "path/to/coco.json"

```


## Contributing to the Project

Please make sure to install all the required libraries in the [requirements.txt](https://github.com/Sydney-Informatics-Hub/aerial-segmentation/tree/main/requirements.txt) file for development.


### Commit rules:

In this project, `pre-commit` is being used. Hence, please make sure you have it in your
environment by installing it with `pip install pre-commit`.

Make sure to run pre-commit on each commit. You can run it before commit on all files in the
repository using `pre-commit run --all-files`. Otherwise, you can run it via `pre-commit run`
which will only run it on files staged for commit.

Alternatively, to add the hook, after installing pre-commit, run:

```
pre-commit install
```

this will run the pre-commit hooks every time you commit changes to the repository.
