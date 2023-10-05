#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from aerialseg.utils import assemble_coco_json, extract_all_annotations_df


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run Detectron2 prediction on aerial imagery and show image with results."
    )
    parser.add_argument(
        "--indir",
        "-i",
        type=str,
        required=True,
        help="Path to a directory containing the input images.",
    )
    parser.add_argument(
        "--in-pattern",
        "-p",
        type=str,
        default="*.png",
        help="Glob pattern to match the input images. Default: %(default)s.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Model configuration YAML file from Detectron2.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="Path to a weights .pth file output from Detectron2 training.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Detection threshold. Default: %(default)s.",
    )
    parser.add_argument(
        "--coco",
        type=str,
        default=None,
        help="Path to a COCO JSON containg the annotation categories. "
        "These will be printed onto the image if provided.",
    )
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        help="If set, will force CPU inference.",
    )
    parser.add_argument(
        "--coco-out",
        "-o",
        type=str,
        default=None,
        help="Path to a COCO JSON file to save the predictions to. By default will save to the same directory as the input image with 'coco-out.json' name.",
    )

    return parser


def main(args=None):
    parser = create_parser()
    args = parser.parse_args()
    cfg = get_cfg()

    config_file = args.config
    weights_file = args.weights
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights_file

    # If we have the COCO JSON then we can set up the class names for the prediction.
    if args.coco is not None:
        with open(args.coco, "r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        thing_classes = [c["name"] for c in categories]
        meta = MetadataCatalog.get("predict")
        meta.thing_classes = thing_classes

    # scan the input directory for images based on the pattern given
    images = glob.glob(os.path.join(args.indir, args.in_pattern))
    assert (
        len(images) > 0
    ), f"No images found in the input directory given the pattern {args.in_pattern}."

    # If not many images are given, this should be okay to go with CPU inference.
    if args.force_cpu:
        cfg.MODEL.DEVICE = "cpu"
    else:
        args.n_jobs = 1  # force to 1 job if using GPU

    predictor = DefaultPredictor(cfg)

    all_annotations = extract_all_annotations_df(
        images, predictor
    )  # , threshold=args.threshold)
    coco_json = assemble_coco_json(
        all_annotations, images, license="", info="", type="instances"
    )
    if args.coco_out is None:
        args.coco_out = os.path.join(args.indir, "coco-out.json")

    with open(args.coco_out, "w") as f:
        json.dump(coco_json.toJSON(), f)


if __name__ == "__main__":
    main()