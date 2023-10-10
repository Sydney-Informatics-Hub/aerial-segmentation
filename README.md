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

## Deploy benchmark script to Jetson Nano

There is a Docker image available with a GPU enabled version of PyTorch and Detectron2 compiled on a
Jetson Nano using Python 3.8. This can be used to deploy scripts from this repository on these devices.
The image can be retrieved to the local Docker repository on a Jetson Nano using:

```
sudo docker pull sydneyinformaticshub/aerialseg:det2-py38-jetson
```

In order to run the Docker image first make sure the Jetson Nano has the lastest compatible version of
Jetpack installed. You will need at least 4.6 to run the image, 5.xand greater are not compatible. Check
[here](https://developer.nvidia.com/embedded/jetpack-archive) for a list of available Jetpack versions
compatible with the Nano.

Once Jetpack is updated you can log into the Docker image and mount the `aerial-segmentation` repository
from the current directory inside it using:

```
sudo docker run -v ${PWD}/aerial-segmentation:/root/aerial-segmentation -it --runtime nvidia sydneyinformaticshub/aerialseg:det2-py38-jetson /bin/bash
```

The `sydneyinformaticshub/aerialseg:det2-py38-jetson` image can also be used as a base image to install
further packages and scripts as required.
