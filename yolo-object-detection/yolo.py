"""Detect objects in a single image with YOLOv3 and OpenCV.

Usage (from the yolo-object-detection folder):
    python yolo.py --image images/baggage_claim.jpg
    python yolo.py --image images/soccer.jpg --output output/soccer.jpg --no-display
"""

import argparse
import sys
import time

import cv2

from yolo_detector import DEFAULT_YOLO_DIR, YoloDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", default=DEFAULT_YOLO_DIR, help="folder with yolov3.cfg, yolov3.weights, coco.names")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to keep a detection")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="overlap threshold for non-maxima suppression")
    ap.add_argument("-o", "--output", help="optional path to save the annotated image")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (useful on servers)")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"[ERROR] could not read image: {args.image}")

    print("[INFO] loading YOLO from disk...")
    detector = YoloDetector(args.yolo, args.confidence, args.threshold)

    start = time.time()
    detections = detector.detect(image)
    print(f"[INFO] YOLO took {time.time() - start:.3f} seconds")

    for det in detections:
        print(f"  {det.label:<15} {det.confidence:.2f}  box={det.box}")
    print(f"[INFO] {len(detections)} object(s) detected")

    detector.draw(image, detections)

    if args.output:
        cv2.imwrite(args.output, image)
        print(f"[INFO] saved {args.output}")

    if not args.no_display:
        cv2.imshow("YOLO detections (press any key to close)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
