#!/DATA/phd-interview_stage-2/task-4_thermal-object-detection/code/pyvenv_3.10.12/bin/python
import json
import cv2
import os
import config
from headers import *

def check_annotations(annotation_file, image_dir):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Loop over each annotation and check the bounding box
    for img_info in tqdm(coco_data['images']):
        image_id = img_info['id']
        img_path = os.path.join(image_dir, img_info['file_name'])

        # Check if the image exists
        if not os.path.exists(img_path):
            print(f"Image {img_info['file_name']} not found in {image_dir}.")
            continue

        # Get the image dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_path}.")
            continue

        img_height, img_width = img.shape[:2]

        # Get all annotations for this image
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        for ann in img_annotations:
            bbox = ann['bbox']  # [x, y, width, height]

            x, y, w, h = bbox
            # Check if bbox is valid
            if w <= 0 or h <= 0:
                print(f"Invalid bounding box (zero or negative dimensions) in image {img_info['file_name']}: {bbox}")
            if x < 0 or y < 0 or (x + w) > img_width or (y + h) > img_height:
                print(f"Bounding box out of image bounds in {img_info['file_name']}: {bbox}")

if __name__ == "__main__":
    annotation_file = config.train_data_annotations  # Path to your annotation file
    image_dir = config.train_data_path  # Directory containing your images
    check_annotations(annotation_file, image_dir)
