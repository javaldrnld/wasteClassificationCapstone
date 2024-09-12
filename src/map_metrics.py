import os
import pathlib
import xml.etree.ElementTree as ET  # for parsing XML files

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.metrics import coco_evaluation
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

matplotlib.use("TkAgg")
plt.ion()
# Define paths to config files and checkpoint directories
MODELS = {
    "MobileNetV2 SSD 320x320": {
        "pipeline_config": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/ssd_v2_1/pipeline.config",
        "checkpoint_dir": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/ssd_v2_1/checkpoint",
    },
    "MobileNetV2 V2 FPNLite 640x640": {
        "pipeline_config": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/fpn_v1/pipeline.config",
        "checkpoint_dir": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/fpn_v1/checkpoint",
    },
    "EfficientDet D0 512x512": {
        "pipeline_config": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/efficient_do/pipeline.config",
        "checkpoint_dir": "/home/bokuto/Documents/Capstone/wasteClassification/outputs/efficient_do/checkpoint",
    },
    # Add more models as needed
}

# Define the COCO categories (class definitions from your label map)
categories = [
    {"id": 1, "name": "cardboard"},
    {"id": 2, "name": "dirt"},
    {"id": 3, "name": "glass"},
    {"id": 4, "name": "metal"},
    {"id": 5, "name": "paper"},
    {"id": 6, "name": "plastic"},
    {"id": 7, "name": "rock"},
    {"id": 8, "name": "trash"},
]


# Function to load model from config and checkpoint
def load_model(pipeline_config_path, checkpoint_dir):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint_dir, "ckpt-0")).expect_partial()

    return detection_model


# Function to run inference and get predictions
def run_inference(model, image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference and extract detections
    detections = model(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }

    # Detection classes should be integers
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    return detections


# Function to calculate mAP
def calculate_map(predictions, ground_truths, categories):
    coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)

    # Add ground truth and predictions
    for image_id, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        coco_evaluator.add_single_ground_truth_image_info(
            image_id=image_id, groundtruth_dict=gt
        )
        coco_evaluator.add_single_detected_image_info(
            image_id=image_id, detections_dict=pred
        )

    metrics = coco_evaluator.evaluate()
    return metrics["DetectionBoxes_Precision/mAP"]


# Function to load test dataset from XML (PASCAL VOC) annotations
def load_test_dataset():
    image_dir = "/home/bokuto/Documents/Capstone/wasteClassification/data/images/test"
    annotation_dir = (
        "/home/bokuto/Documents/Capstone/wasteClassification/data/annotations/xml"
    )

    image_paths = []
    ground_truths = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            image_paths.append(str(pathlib.Path(image_path)))

            # Get the corresponding XML annotation file
            annotation_file = os.path.join(
                annotation_dir, filename.replace(".jpg", ".xml")
            )
            if os.path.exists(annotation_file):
                boxes, classes = [], []

                # Parse the XML file
                tree = ET.parse(annotation_file)
                root = tree.getroot()

                for obj in root.findall("object"):
                    # Get class label
                    class_name = obj.find("name").text
                    class_id = next(
                        cat["id"] for cat in categories if cat["name"] == class_name
                    )

                    # Get bounding box coordinates
                    bbox = obj.find("bndbox")
                    x_min = int(bbox.find("xmin").text)
                    y_min = int(bbox.find("ymin").text)
                    x_max = int(bbox.find("xmax").text)
                    y_max = int(bbox.find("ymax").text)

                    boxes.append([x_min, y_min, x_max, y_max])
                    classes.append(class_id)

                # Append the ground truth boxes and classes
                ground_truths.append(
                    {"boxes": np.array(boxes), "classes": np.array(classes)}
                )

    images = [np.array(Image.open(img_path)) for img_path in image_paths]
    return images, ground_truths


def load_label_map(label_map_path):
    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path, use_display_name=True
    )
    return category_index


def main():
    # Load the test dataset
    PATH_TO_LABELS = "/home/bokuto/Documents/Capstone/wasteClassification/data/annotations/label_map.pbtxt"
    category_index = load_label_map(PATH_TO_LABELS)
    images, ground_truths = load_test_dataset()

    model_map_results = {}

    for model_name, paths in MODELS.items():
        # Load model
        detection_model = load_model(paths["pipeline_config"], paths["checkpoint_dir"])

        all_predictions = []
        all_ground_truths = []

        # Run inference on test images
        for image_np, ground_truth in zip(images, ground_truths):
            detections = run_inference(detection_model, image_np)
            all_predictions.append(detections)
            all_ground_truths.append(ground_truth)

        # Calculate mAP
        mAP = calculate_map(all_predictions, all_ground_truths, categories)
        model_map_results[model_name] = mAP
        print(f"{model_name} mAP: {mAP:.4f}")

    # Plot mAP comparison

    map_values = list(model_map_results.values())
    model_names = list(model_map_results.keys())

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, map_values, color="orange")
    plt.xlabel("Models")
    plt.ylabel("mAP")
    plt.title("Model mAP Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        "/home/bokuto/Documents/Capstone/wasteClassification/outputs/mAP_comparison.png"
    )
    plt.show(block=True)


if __name__ == "__main__":
    main()
