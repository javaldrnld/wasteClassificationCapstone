import os

import cv2
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils

tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


PATH_TO_CFG = (
    "/home/bokuto/Documents/Capstone/wasteClassification/outputs/fpn_v2/pipeline.config"
)
PATH_TO_CKPT = (
    "/home/bokuto/Documents/Capstone/wasteClassification/outputs/fpn_v2/checkpoint/"
)

print("Loading model... ", end="")

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


PATH_TO_LABELS = "/home/bokuto/Documents/Capstone/wasteClassification/data/annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)


# Initialize webcam
cap = cv2.VideoCapture(2)  # 0 for default camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    image_np = np.array(frame)

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))

    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Apply Non-Maximum Suppression
    nms_indices = tf.image.non_max_suppression(
        detections["detection_boxes"],
        detections["detection_scores"],
        max_output_size=20,
        iou_threshold=0.5,
        score_threshold=0.80,
    )

    # Filter boxes, scores, and classes
    nms_boxes = tf.gather(detections["detection_boxes"], nms_indices).numpy()
    nms_scores = tf.gather(detections["detection_scores"], nms_indices).numpy()
    nms_classes = tf.gather(detections["detection_classes"], nms_indices).numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        nms_boxes,
        nms_classes + label_id_offset,
        nms_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.80,
        agnostic_mode=False,
    )

    cv2.imshow("object detection", cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        break
