"""Download the pre-trained YOLOv3 COCO weights (about 248 MB) into yolo-coco/.

The weights file is too large for GitHub, so it is not stored in this repo.
The script tries the official Darknet site first, then a mirror, and checks
the SHA-256 checksum so a broken or wrong download is never used.

Usage (from the yolo-object-detection folder):
    python download_weights.py
"""

import hashlib
import os
import sys
import urllib.request

URLS = [
    "https://pjreddie.com/media/files/yolov3.weights",
    "https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights",
]
SHA256 = "523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-coco", "yolov3.weights")
# some hosts reject Python's default "Python-urllib" client name with 403 Forbidden
HEADERS = {"User-Agent": "Mozilla/5.0 (opencv-ObjectDetection weights downloader)"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            size = f" / {total / 1e6:.1f}" if total else ""
            sys.stdout.write(f"\r[INFO] {done / 1e6:6.1f}{size} MB")
            sys.stdout.flush()
    print()


def main():
    if os.path.isfile(DEST) and sha256(DEST) == SHA256:
        print(f"[INFO] {DEST} already present and verified")
        return

    tmp = DEST + ".part"
    for url in URLS:
        print(f"[INFO] downloading {url}")
        try:
            download(url, tmp)
        except Exception as exc:  # network error, 403, 404 ... try the next source
            print(f"\n[WARN] download failed: {exc}")
            continue
        if sha256(tmp) == SHA256:
            os.replace(tmp, DEST)
            print(f"[INFO] saved and verified {DEST}")
            return
        print("[WARN] checksum mismatch, trying the next source")

    if os.path.exists(tmp):
        os.remove(tmp)
    sys.exit("[ERROR] could not download yolov3.weights. Download it manually from\n"
             f"        {URLS[0]}\n        and save it as {DEST}")


if __name__ == "__main__":
    main()
