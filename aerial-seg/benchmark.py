import os
import time
import torch
import torchvision
import argparse
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

def parse_arguments():
    parser = argparse.ArgumentParser(description="Segment buildings in images using Detectron2")
    parser.add_argument("--input-dir", required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", required=True, help="Output directory to save annotated images")
    parser.add_argument("--config-yaml", required=True, help="Path to the Detectron2 config YAML file")
    parser.add_argument("--model-weights", required=True, help="Path to the model weights file")
    parser.add_argument("--roi-score-thresh", type=float, default=0.5, help="ROI score threshold for detection")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set up Detectron2 configuration and model
    cfg = get_cfg()
    cfg.merge_from_file(args.config_yaml)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.roi_score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    predictor = DefaultPredictor(cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    iteration_times = []

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            start_time = time.time()
            im = Image.open(input_path)
            annotated_im = segment_buildings(im, predictor)
            annotated_im.save(output_path)
            end_time = time.time()

            iteration_time = end_time - start_time
            iteration_times.append(iteration_time)

            print(f"Processed: {filename} | Time: {iteration_time:.2f} s")

    # Calculate and print performance benchmarks
    avg_iteration_time = sum(iteration_times) / len(iteration_times)
    total_images = len(iteration_times)
    print("\nPerformance Benchmarks:")
    print(f"Total Images Processed: {total_images}")
    print(f"Average Iteration Time: {avg_iteration_time:.2f} s")

def segment_buildings(im, predictor):
    im = np.array(im)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    print(len(outputs["instances"]), "buildings detected.")
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return Image.fromarray(out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()
