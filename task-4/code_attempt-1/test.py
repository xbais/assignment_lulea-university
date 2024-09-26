#!/DATA/phd-interview_stage-2/task-4_thermal-object-detection/code/pyvenv_3.10.12/bin/python
from headers import *
import config

import os
import cv2 # opencv-python
import torch
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

'''
def test_model(test_dataset_dir, test_annotation_file, output_dir, model_path):
    # Register your custom test dataset
    register_coco_instances("custom_test", {}, test_annotation_file, test_dataset_dir)
    
    # Load configuration and model weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("custom_test",)
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.MODEL.WEIGHTS = model_path  # Load the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes  # Set to number of classes in your dataset
    cfg.OUTPUT_DIR = output_dir  # Directory to save the output
    
    # Create a predictor using the configuration and model
    predictor = DefaultPredictor(cfg)
    
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the test dataset
    test_loader = build_detection_test_loader(cfg, "custom_test")

    # Run evaluation on the test dataset
    #evaluator = COCOEvaluator("custom_test", cfg, False, output_dir=output_dir)
    #metrics = inference_on_dataset(predictor.model, test_loader, evaluator)
    #print("Evaluation metrics:", metrics)
    
    # Save output images with bounding boxes and class labels
    metadata = MetadataCatalog.get("custom_test")
    
    for idx, inputs in enumerate(test_loader.dataset):
        img = cv2.imread(inputs["file_name"])
        outputs = predictor(img)  # Perform object detection
        
        # Create visualizer for bounding box drawing
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save image with bounding boxes and class labels
        out_img = out.get_image()[:, :, ::-1]
        output_image_path = os.path.join(output_dir, f"output_{idx}.jpg")
        cv2.imwrite(output_image_path, out_img)
        print(f"Saved output image: {output_image_path}")
'''

'''
def test_model(test_dataset_dir, test_annotation_file, output_dir, model_path):
    # Register your custom test dataset
    register_coco_instances("custom_test", {}, test_annotation_file, test_dataset_dir)
    
    # Load configuration and model weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("custom_test",)
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.MODEL.WEIGHTS = model_path  # Load the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes  # Set to number of classes in your dataset
    cfg.OUTPUT_DIR = output_dir  # Directory to save the output
    
    # Create a predictor using the configuration and model
    predictor = DefaultPredictor(cfg)
    
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the test dataset
    test_loader = build_detection_test_loader(cfg, "custom_test")

    # Metadata for custom test set
    metadata = MetadataCatalog.get("custom_test")
    
    # Iterate over the test dataset loader
    for idx, inputs in enumerate(test_loader):
        # Retrieve the image from 'inputs' correctly
        img = inputs[0]["image"].permute(1, 2, 0).cpu().numpy()  # Converts from Tensor format to numpy
        
        # Get the file name (or load image if necessary)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Perform object detection
        outputs = predictor(img_bgr)  
        
        # Create visualizer for bounding box drawing
        v = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save image with bounding boxes and class labels
        out_img = out.get_image()[:, :, ::-1]  # Convert back to BGR format
        output_image_path = os.path.join(output_dir, f"output_{idx}.jpg")
        cv2.imwrite(output_image_path, out_img)
        print(f"Saved output image: {output_image_path}")
'''

def test_model(test_dataset_dir, test_annotation_file, output_dir, model_path):
    # Register your custom test dataset
    register_coco_instances("custom_test", {}, test_annotation_file, test_dataset_dir)
    
    # Load configuration and model weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("custom_test",)
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.MODEL.WEIGHTS = model_path  # Load the trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # Set a lower threshold for testing
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes  # Set to number of classes in your dataset
    cfg.OUTPUT_DIR = output_dir  # Directory to save the output
    
    # Create a predictor using the configuration and model
    predictor = DefaultPredictor(cfg)
    
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Metadata for custom test set
    metadata = MetadataCatalog.get("custom_test")
    
    # Iterate over the test dataset loader
    test_loader = build_detection_test_loader(cfg, "custom_test")
    for idx, inputs in enumerate(test_loader):
        img = inputs[0]["image"].permute(1, 2, 0).cpu().numpy()  # Converts from Tensor format to numpy
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Perform object detection
        outputs = predictor(img_bgr)

        # Debugging: Check if any instances were detected
        instances = outputs["instances"]
        if len(instances) == 0:
            print(f"No instances detected in image {idx}.")
            continue

        print(f"Detected {len(instances)} instances in image {idx}.")

        # Create visualizer for bounding box drawing
        v = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, scale=0.5)
        out = v.draw_instance_predictions(instances.to("cpu"))

        # Save image with bounding boxes and class labels
        out_img = out.get_image()[:, :, ::-1]  # Convert back to BGR format
        output_image_path = os.path.join(output_dir, f"output_{idx}.jpg")
        cv2.imwrite(output_image_path, out_img)
        print(f"Saved output image: {output_image_path}")
        
if __name__ == "__main__":
    # Set paths to the test dataset and annotation files
    test_dataset_dir = config.test_data_path
    test_annotation_file = config.test_data_annotations
    output_dir = "./test_outputs"  # Directory to save output images and results
    model_path = config.pretrained_model_for_testing  # Path to the trained model

    # Run the testing script
    test_model(test_dataset_dir, test_annotation_file, output_dir, model_path)
