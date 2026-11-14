"""Real-time object detection with MobileNet SSD (Caffe) and OpenCV.

Reads frames from a webcam (default) or a video file, detects the 20 PASCAL VOC
classes, and shows the result live. Press q to quit.

Usage (from the SSDMobile-opencv-real-time-object-detection folder):
    python real_time_object_detection.py                       # webcam 0
    python real_time_object_detection.py --input ../yolo-object-detection/videos/airport.mp4
    python real_time_object_detection.py --input clip.mp4 --output out.avi --no-display
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# classes MobileNet SSD was trained on (PASCAL VOC + background)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prototxt", default=os.path.join(HERE, "MobileNetSSD_deploy.prototxt.txt"),
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default=os.path.join(HERE, "MobileNetSSD_deploy.caffemodel"),
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", default="0", help="webcam index (0, 1, ...) or path to a video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to keep a detection")
    ap.add_argument("-w", "--width", type=int, default=960, help="resize frames to this width before display")
    ap.add_argument("-o", "--output", help="optional path to save the annotated video (.avi)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window (useful on servers)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = no limit)")
    args = ap.parse_args()

    if not hasattr(cv2.dnn, "readNetFromCaffe"):
        sys.exit(f"[ERROR] OpenCV {cv2.__version__} cannot load Caffe models "
                 '(removed in OpenCV 5). Install OpenCV 4: pip install "opencv-python>=4.8,<5"')

    rng = np.random.default_rng(7)
    colors = rng.uniform(0, 255, size=(len(CLASSES), 3))

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    source = int(args.input) if args.input.isdigit() else args.input
    print(f"[INFO] starting video stream from {'webcam ' if isinstance(source, int) else ''}{source}...")
    vs = cv2.VideoCapture(source)
    if not vs.isOpened():
        sys.exit(f"[ERROR] could not open video source: {args.input}")
    if isinstance(source, int):
        time.sleep(2.0)  # let the camera sensor warm up

    writer = None
    frames = 0
    started = time.time()
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break  # end of file or camera disconnected
        frame = resize_to_width(frame, args.width)
        (h, w) = frame.shape[:2]

        # MobileNet SSD expects 300x300 inputs, mean 127.5 and scale 1/127.5
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()  # shape: (1, 1, N, 7)

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence <= args.confidence:
                continue
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors[idx], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        if args.output:
            if writer is None:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter.fourcc(*"MJPG"), 30, (w, h), True)
            writer.write(frame)

        frames += 1
        if not args.no_display:
            cv2.imshow("MobileNet SSD (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    elapsed = time.time() - started
    print(f"[INFO] elapsed time: {elapsed:.2f} s")
    print(f"[INFO] approx. FPS: {frames / elapsed if elapsed else 0:.2f}")
    if args.output:
        print(f"[INFO] saved {args.output}")

    if writer is not None:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
