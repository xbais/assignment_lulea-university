import os
import json

def coco_to_yolo(coco_json_path, images_dir, output_dir):
    # Load COCO JSON annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image information from COCO JSON
    images = {img['id']: img for img in coco_data['images']}
    
    # List of images in the directory (without extensions)
    available_images = {os.path.splitext(img)[0] for img in os.listdir(images_dir) if img.endswith(('.jpg', '.png'))}

    missing_images = []  # To track images missing from the dataset
    created_labels = 0  # Track how many labels have been created

    # Process each annotation in coco.json
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image_info = images[image_id]
        img_width = image_info['width']
        img_height = image_info['height']

        # COCO bbox format: [x_min, y_min, width, height]
        bbox = annotation['bbox']
        x_min, y_min, bbox_width, bbox_height = bbox

        # Convert COCO bbox to YOLO format (x_center, y_center, width, height)
        x_center = (x_min + bbox_width / 2) / img_width
        y_center = (y_min + bbox_height / 2) / img_height
        bbox_width /= img_width
        bbox_height /= img_height

        # YOLO annotation format: [class_id, x_center, y_center, bbox_width, bbox_height]
        class_id = annotation['category_id'] - 1  # YOLO expects classes starting from 0

        # Get the image file name without extension
        image_file_name = os.path.splitext(image_info['file_name'])[0]

        # Check if the image exists in the dataset directory
        if image_file_name not in available_images:
            # Log the missing image
            missing_images.append(image_info['file_name'])
            continue

        # Prepare the corresponding YOLO label file path
        label_file_path = os.path.join(output_dir, f"{image_file_name}.txt")

        # Ensure the labels directory exists
        label_dir = os.path.dirname(label_file_path)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Write the YOLO label to the file
        with open(label_file_path, 'a') as label_file:
            label_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            created_labels += 1

    # Log missing images
    if missing_images:
        print(f"{len(missing_images)} images referenced in coco.json are missing from {images_dir}.")
        with open('missing_images_report.txt', 'w') as report_file:
            for img in missing_images:
                report_file.write(f"{img}\n")
        print(f"Missing images list saved to missing_images_report.txt")

    print(f"Created {created_labels} YOLO labels in {output_dir}")

# Set paths
coco_json_path = "/home/aakash/Aakash/object_detection/task-4_thermal-object-detection/data/FLIR_ADAS_v2/video_thermal_test/coco_updated.json"
images_dir = "/home/aakash/Aakash/object_detection/task-4_thermal-object-detection/data/FLIR_ADAS_v2/video_thermal_test/data"  # Path to your images directory
output_dir = "/home/aakash/Aakash/object_detection/task-4_thermal-object-detection/data/FLIR_ADAS_v2/video_thermal_test/labels"

# Convert COCO annotations to YOLO format
coco_to_yolo(coco_json_path, images_dir, output_dir)
