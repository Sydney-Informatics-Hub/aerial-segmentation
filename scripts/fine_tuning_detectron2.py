#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import wandb
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from roboflow import Roboflow

setup_logger()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune Detectron2 weights using annotated Roboflow dataset."
    )
    roboflow_group = parser.add_argument_group("Roboflow options")
    roboflow_group.add_argument(
        "--workspace",
        type=str,
        default="sih-vpfnf",
        help="Roboflow workspace name. (Default: %(default)s)",
    )
    roboflow_group.add_argument(
        "--project",
        type=str,
        default="gis-hd-200x200",
        help="Roboflow project name. (Default: %(default)s)",
    )
    roboflow_group.add_argument(
        "--project-version",
        type=int,
        default=0,
        help="Roboflow project version number to download. (Default: Highest available).",
    )
    roboflow_group.add_argument(
        "--roboflow-api-key", "-r", type=str, help="Roboflow API key.", required=True
    )
    wandb_group = parser.add_argument_group("Weights & Biases options")
    wandb_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Initialise Weights & Biases (wandb) for this run.",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="gis-segmentation",
        help="The name of the wandb project being sent the new run. (Default: %(default)s)",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        type=str,
        default="sih",
        help="wandb username or team name to whose UI the run will be sent.",
    )
    return parser


def get_roboflow_dataset(
    api_key: str, workspace: str, project: str, version_number: int = 0
):
    """Download a dataset from the Roboflow server.

    Parameters
    ----------
    api_key : str
        A Roboflow API key required to access the data.
    workspace : str
        Roboflow workspace name.
    project : str
        Roboflow project name.
    version_number : int, optional
        The version number of the Roboflow project.
        Values < 1 mean find the highest available version.

    Returns
    -------
    :class:`Dataset`
        A Dataset instance pointing to the data downloaded from Roboflow
    """

    # This script only supports 'coco-segmentation' format.
    # I'm making it explicit here in case we generalise it in future.
    roboflow_dataset_format = "coco-segmentation"

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version_list = project.versions()
    all_versions = [os.path.basename(version.id) for version in version_list]
    # if version_number is 0 get the highest available otherwise
    # try to pick the specified version
    if version_number < 1:
        version_number = max(all_versions)
    version = project.version(version_number)
    dataset = version.download(roboflow_dataset_format)
    return dataset


def register_coco_json_from_roboflow(
    name: str, location: str, instance_type: str = "train"
):
    """Register a COCO JSON format file obtained from Roboflow for Detectron2
    instance detection.

    Parameters
    ----------
    name : str
        The name of the dataset obtained from Roboflow.
    location : str
        The root path of the dataset downloaded from Roboflow.
    instance_type : str, optional
        The instance type to register, defaults to 'train'.
        It is assumed the COCO JSON and related images are contained in
        the sub-directory `instance_type` of `location`.

    Returns
    -------
    str
        The name of the registered instance - which can be used by Detectron2.
    """
    # Roboflow always saves its COCO annotation file in the dir containing the
    # images with this filename.
    roboflow_coco_json_filename = "_annotations.coco.json"

    instance_name = name + "_" + instance_type
    image_root = os.path.join(location, instance_type)
    coco_json_file = os.path.join(location, instance_type, roboflow_coco_json_filename)
    register_coco_instances(instance_name, {}, coco_json_file, image_root)
    return instance_name


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)

    # Shall we use wandb?
    if args.use_wandb:
        # For now just accept API keys from the TTY - but we may want to change this later
        # so that it can run noninteractively.
        wandb.login()
        # There are loads of options to wandb.init, some of which we might like to expose
        # further down the line - for now I'm just using the options from the colab notebook.
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, sync_tensorboard=True
        )

    # Download the roboflow dataset and register it in detectron2
    dataset = get_roboflow_dataset(
        args.roboflow_api_key,
        args.workspace,
        args.project,
        version_number=args.project_version,
    )

    # Register the training dataset with detectron2
    register_coco_json_from_roboflow(
        dataset.name, dataset.location, instance_type="train"
    )

    # Shutdown wandb run if we were using it in this process.
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
