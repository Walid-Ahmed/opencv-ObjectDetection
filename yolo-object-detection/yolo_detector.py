"""Shared YOLOv3 helpers used by yolo.py (images) and yolo_video.py (videos).

Loads the Darknet YOLOv3 model with OpenCV's DNN module, runs it on a frame,
filters weak detections, applies non-maxima suppression, and draws the boxes.
"""

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_DIR = os.path.join(HERE, "yolo-coco")
INPUT_SIZE = (416, 416)  # YOLOv3 was trained on 416x416 inputs


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x, y, width, height) in pixels


class YoloDetector:
    def __init__(self, yolo_dir=DEFAULT_YOLO_DIR, confidence=0.5, threshold=0.3):
        if not hasattr(cv2.dnn, "readNetFromDarknet"):
            sys.exit(
                f"[ERROR] OpenCV {cv2.__version__} cannot load Darknet models "
                "(the importer was removed in OpenCV 5). Install OpenCV 4: "
                'pip install "opencv-python>=4.8,<5"'
            )

        weights = os.path.join(yolo_dir, "yolov3.weights")
        config = os.path.join(yolo_dir, "yolov3.cfg")
        names = os.path.join(yolo_dir, "coco.names")
        if not os.path.isfile(weights):
            sys.exit(
                f"[ERROR] {weights} not found. Download it first:\n"
                "        python download_weights.py"
            )

        with open(names) as f:
            self.labels = f.read().strip().split("\n")

        # one fixed color per class so the same class always looks the same
        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, size=(len(self.labels), 3), dtype="uint8")

        self.confidence = confidence
        self.threshold = threshold
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        # YOLOv3 has three output layers (one per detection scale)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, frame):
        """Return the list of Detection objects found in a BGR frame."""
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                # each row: center x, center y, width, height, objectness, 80 class scores
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence <= self.confidence:
                    continue
                # YOLO returns coordinates relative to the image size
                center_x, center_y, width, height = detection[0:4] * np.array([W, H, W, H])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                class_ids.append(class_id)

        # non-maxima suppression removes overlapping boxes for the same object
        keep = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        return [
            Detection(self.labels[class_ids[i]], confidences[i], tuple(boxes[i]))
            for i in np.array(keep).flatten()
        ]

    def draw(self, frame, detections):
        """Draw boxes and labels on the frame in place."""
        for det in detections:
            x, y, w, h = det.box
            color = [int(c) for c in self.colors[self.labels.index(det.label)]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{det.label}: {det.confidence:.2f}"
            cv2.putText(frame, text, (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
