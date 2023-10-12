"""
building-segmentation
Proof of concept showing effectiveness of a fine tuned instance segmentation model for deteting buildings.
"""
import os
import cv2
os.system("pip install 'git+https://github.com/facebookresearch/detectron2.git'")
from transformers import DetrFeatureExtractor, DetrForSegmentation
from PIL import Image
import gradio as gr
import numpy as np
import torch
import torchvision
import detectron2

# import some common detectron2 utilities
import itertools
import seaborn as sns
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer

cfg = get_cfg()
cfg.merge_from_file("model_weights/buildings_poc_cfg.yml")
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.WEIGHTS = "model_weights/chatswood_buildings_poc.pth"  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
predictor = DefaultPredictor(cfg)

def segment_buildings(im):

    """
    Segment buildings in an image using a pre-trained instance segmentation model.

    Args:
    im (PIL.Image): The input image to segment buildings from.

    Returns:
    PIL.Image: An image with buildings segmented and highlighted.
    """

    im = np.array(im)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )
    # Print out information to appear in logs for diagnostics
    print(len(outputs["instances"])," buildings detected.")
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    return Image.fromarray(out.get_image()[:, :, ::-1])

# gradio components 
"""
gr_slider_confidence = gr.inputs.Slider(0,1,.1,.7,
                                        label='Set confidence threshold % for masks')
"""
# gradio outputs
inputs = gr.inputs.Image(type="pil", label="Input Image")
outputs = gr.outputs.Image(type="pil", label="Output Image")

title = "Building Segmentation"
description = "An instance segmentation demo for identifying boundaries of buildings in aerial images using DETR (End-to-End Object Detection) model with MaskRCNN-101 backbone."

# Create user interface and launch
gr.Interface(segment_buildings, 
                inputs = inputs,
                outputs = outputs,
                description = description).launch(debug=True)
