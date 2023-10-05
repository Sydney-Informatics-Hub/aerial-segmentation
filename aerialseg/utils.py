# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from aerial_conversion import coco
from PIL import Image
from tqdm import tqdm


def save_images_as_gif(input_folder, output_gif_path, duration=100):
    """Save a folder of images as an animated GIF.

    Args:
    - input_folder (str): Path to the folder containing image files (e.g., JPEG or PNG).
    - output_gif_path (str): Path to save the animated GIF file.
    - duration (int, optional): Duration (in milliseconds) for each frame in the GIF. Default is 100ms.

    Returns:
    - None
    """
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".png", ".jpeg", ".gif"))
    ]

    if not image_files:
        print("No image files found in the input folder.")
        return

    images = []
    for image_file in sorted(image_files):
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save the animated GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def extract_output_annotations(output, flatten: bool = False):
    """Extracts polygons, bounding boxes, and binary masks from prediction
    ouputs.

    Args:
        output: Detectron2 prediction output
        flatten (bool): If true, will flatten polygons, as such used in coco segmentations.

    #TODO: Simplify shapes
    """
    mask_array = output["instances"].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    # scores = output['instances'].scores.to("cpu").numpy()
    labels = output["instances"].pred_classes.to("cpu").numpy()
    bbox = output["instances"].pred_boxes.to("cpu").tensor.numpy()
    # print(mask_array.shape)
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_arrays = []
    polygons = []
    for i in range(num_instances):
        # img = np.zeros_like(image)
        mask_array_instance = mask_array[:, :, i : (i + 1)]
        mask_arrays.append(mask_array_instance)
        # img = np.where(mask_array_instance[i] == True, 255, img)
        polygon = sv.mask_to_polygons(mask_array_instance)[0]

        if flatten:
            polygons.append(polygon.flatten().tolist())
        else:
            polygons.append(polygon.tolist())

    return mask_arrays, polygons, bbox, labels


def extract_tile_annotations_df(image_path, image_id, predictor):
    """Reads through tiles, predicts, and extracts annnotations as a dataframe.

    Args:
        image_path (str): path to the tile png file
        image_id (int): an id for the image tile. Usually a unique int
        predictor: Detectron2 predictor object
    """
    image = cv2.imread(image_path)
    output = predictor(image)
    _, polygons, _, labels = extract_output_annotations(output)
    annotations = pd.DataFrame(
        {"pixel_polygon": polygons, "image_id": image_id, "class_id": labels}
    )  # "annot_id" should be added later

    return annotations


def extract_all_annotations_df(images_list: list, predictor):
    """Extract and combine tile annotations into a single dataframe.

    Args:
        images_list (list): A list of image paths
        predictor: Detectron2 predictor object
    """

    all_annotations = []
    for image_index, image in tqdm(enumerate(images_list), total=len(images_list)):
        all_annotations.append(
            extract_tile_annotations_df(image, image_index, predictor)
        )

    all_annotations = pd.concat(all_annotations)
    all_annotations = all_annotations.reset_index(drop=True)
    all_annotations = all_annotations.reset_index()
    all_annotations.columns = ["annot_id", "pixel_polygon", "image_id", "class_id"]

    return all_annotations


def assemble_coco_json(
    annotations, images, license: str = "", info: str = "", type: str = "instances"
):
    """Generate a coco json object.

    Args:
        annotations (Pandas.DataFrame): a dataframe of annotations, usually generated via extract_all_annotations_df function.
        images (list): a list of image paths
        license (str): license of the dataset
        info (str): info of the dataset
        type (str, optional): type of the segmentation. Defaults to "instances"
    """
    coco_json = coco.coco_json()
    coco_json.images = coco.create_coco_images_object_png(images).images
    coco_json.annotations = coco.coco_polygon_annotations(
        annotations
    )  # [tmp2]#[annots_tmp[0]]#
    coco_json.license = license
    coco_json.type = type
    coco_json.info = info
    coco_json.categories = [
        coco.make_category(class_name=str(cat), class_id=cat)
        for cat in annotations.groupby("class_id").groups.keys()
    ]

    return coco_json
