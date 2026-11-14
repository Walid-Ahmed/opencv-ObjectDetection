"""Detect objects in every frame of a video with YOLOv3 and save the result.

Usage (from the yolo-object-detection folder):
    python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi
    python yolo_video.py --input videos/overpass.mp4 --output output/overpass.avi --display
"""

import argparse
import sys
import time

import cv2

from yolo_detector import DEFAULT_YOLO_DIR, YoloDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="path to input video")
    ap.add_argument("-o", "--output", required=True, help="path to output video (.avi)")
    ap.add_argument("-y", "--yolo", default=DEFAULT_YOLO_DIR, help="folder with yolov3.cfg, yolov3.weights, coco.names")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to keep a detection")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="overlap threshold for non-maxima suppression")
    ap.add_argument("--display", action="store_true", help="also show frames in a window while processing (q to stop)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = whole video)")
    args = ap.parse_args()

    vs = cv2.VideoCapture(args.input)
    if not vs.isOpened():
        sys.exit(f"[ERROR] could not open video: {args.input}")

    total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vs.get(cv2.CAP_PROP_FPS) or 30  # keep the original playback speed
    if args.max_frames:
        total = min(total, args.max_frames) if total > 0 else args.max_frames
    print(f"[INFO] {total if total > 0 else 'unknown number of'} frames to process")

    print("[INFO] loading YOLO from disk...")
    detector = YoloDetector(args.yolo, args.confidence, args.threshold)

    writer = None
    frames = 0
    started = time.time()
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break  # end of the video

        detector.draw(frame, detector.detect(frame))

        if writer is None:
            fourcc = cv2.VideoWriter.fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        frames += 1

        if frames == 1 and total > 0:
            elapsed = time.time() - started
            print(f"[INFO] single frame took {elapsed:.3f} s, estimated total {elapsed * total:.0f} s")
        elif frames % 50 == 0:
            print(f"[INFO] processed {frames}/{total if total > 0 else '?'} frames")

        if args.display:
            cv2.imshow("YOLO video (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    print(f"[INFO] wrote {frames} frames to {args.output} in {time.time() - started:.1f} s")
    if writer is not None:
        writer.release()
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
