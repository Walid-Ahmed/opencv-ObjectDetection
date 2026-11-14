"""Download the pre-trained YOLOv3 COCO weights (about 248 MB) into yolo-coco/.

The weights file is too large for GitHub, so it is not stored in this repo.

Usage (from the yolo-object-detection folder):
    python download_weights.py
"""

import hashlib
import os
import sys
import urllib.request

URL = "https://pjreddie.com/media/files/yolov3.weights"
SHA256 = "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-coco", "yolov3.weights")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def progress(blocks, block_size, total):
    done = min(blocks * block_size, total)
    sys.stdout.write(f"\r[INFO] {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
    sys.stdout.flush()


def main():
    if os.path.isfile(DEST) and sha256(DEST) == SHA256:
        print(f"[INFO] {DEST} already present and verified")
        return
    print(f"[INFO] downloading {URL}")
    tmp = DEST + ".part"
    urllib.request.urlretrieve(URL, tmp, progress)
    print()
    if sha256(tmp) != SHA256:
        os.remove(tmp)
        sys.exit("[ERROR] checksum mismatch, download corrupted; please run again")
    os.replace(tmp, DEST)
    print(f"[INFO] saved and verified {DEST}")


if __name__ == "__main__":
    main()
