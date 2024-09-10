import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import pathlib
import random
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

matplotlib.use("TkAgg")
plt.ion()

tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_local_images():
    base_dir = "/home/bokuto/Documents/Capstone/wasteClassification/data/images/test"
    image_paths = []

    # Get all images from the test directory
    for filename in os.listdir(base_dir):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            image_path = os.path.join(base_dir, filename)
            image_path = pathlib.Path(image_path)
            image_paths.append(str(image_path))

    return random.sample(image_paths, 5) if len(image_paths) > 5 else image_paths


IMAGE_PATHS = load_local_images()

# Load pipeline config and build a detection model

PATH_TO_CFG = (
    "/home/bokuto/Documents/Capstone/wasteClassification/outputs/pipeline.config"
)
PATH_TO_CKPT = "/home/bokuto/Documents/Capstone/wasteClassification/outputs/checkpoint/"

print("Loading model... ", end="")
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs["model"]
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore Checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, "ckpt-0")).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))

# Load Label Map Data for plotting
PATH_TO_LABELS = "/home/bokuto/Documents/Capstone/wasteClassification/data/annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)

# Putting everything together

warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


# Process only the first 5 images

plt.figure(figsize=(15, 12))
for i, image_path in enumerate(IMAGE_PATHS, 1):
    print(f"Processing image {i} of 5: {image_path}")

    image_np = load_image_into_numpy_array(image_path)
    print(f"Image shape: {image_np.shape}")

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    print(f"Number of detections: {num_detections}")
    print(f"Detection classes: {detections['detection_classes']}")
    print(f"Detection scores: {detections['detection_scores']}")

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.5,  # Lowered threshold for debugging
        agnostic_mode=False,
    )

    plt.subplot(2, 3, i)
    plt.imshow(image_np_with_detections)
    plt.title(f"Image {i}: {os.path.basename(image_path)}")
    plt.axis("off")
    print(f"Finished processing image {i}\n")

plt.tight_layout()
plt.show(block=True)
print("All images processed. Check if a plot window has opened.")
