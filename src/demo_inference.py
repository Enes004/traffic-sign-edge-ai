"""Render a 30-second local CPU inference slideshow from GTSDB test images."""
import argparse
import subprocess

import cv2
import numpy as np

from benchmark_pi_v3 import CLASS_NAMES
from local_inference import CONFIDENCE, ROOT, TEST_IMAGES, Detector, read_image

WIDTH, HEIGHT = 1280, 720
FPS, SECONDS_PER_IMAGE = 24, 3
COLORS = ((255, 232, 85), (80, 197, 255), (255, 155, 240))


def caption(frame, text, pos, scale, color=(245, 245, 245), thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def make_frame(path, detector, index, count, variant, seed):
    image = read_image(path)
    detections = detector.predict(image)
    frame = np.full((HEIGHT, WIDTH, 3), (26, 23, 20), np.uint8)
    height, width = image.shape[:2]
    scale = min(WIDTH / width, 512 / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    left = (WIDTH - resized.shape[1]) // 2
    top = 104 + (512 - resized.shape[0]) // 2
    for x1, y1, x2, y2, confidence, cls in reversed(detections):
        cls = int(cls)
        x1, y1, x2, y2 = (round(value * scale) for value in (x1, y1, x2, y2))
        color = COLORS[cls]
        cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{CLASS_NAMES[cls]} {confidence:.2f}"
        (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = min(x1, resized.shape[1] - label_width - 8)
        y = max(24, y1)
        cv2.rectangle(resized, (x, y - 23), (x + label_width + 8, y), (20, 20, 20), -1)
        caption(resized, label, (x + 4, y - 6), 0.55, color)
    frame[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    caption(frame, "YOLO26n  |  Traffic sign detection", (28, 43), 1.05, thickness=2)
    caption(frame, f"{detector.resolution} x {detector.resolution}  |  {variant.upper()} ONNX  |  seed {seed}",
            (28, 80), 0.70)
    caption(frame, f"GTSDB test image {path.stem}  |  {index}/{count}  |  {len(detections)} detections",
            (28, 652), 0.74)
    caption(frame, f"Local CPU inference  |  still images  |  confidence > {CONFIDENCE:g}",
            (28, 688), 0.62, (208, 214, 219))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, choices=(320, 640), default=640)
    parser.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    paths = sorted(TEST_IMAGES.glob("*.jpg"))
    if len(paths) != 227:
        parser.error(f"Expected 227 images in {TEST_IMAGES}, found {len(paths)}")
    paths = paths[:50:5]  # Fixed IDs by order, independent of predictions.
    detector = Detector(args.imgsz, args.variant, args.seed)
    directory = ROOT / "results/demo"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"yolo26n_img{args.imgsz}_seed{args.seed}_{args.variant}.mp4"
    frames = [make_frame(path, detector, idx, len(paths), args.variant, args.seed)
              for idx, path in enumerate(paths, 1)]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        with encoder.stdin:
            for frame in frames:
                pixels = frame.tobytes()
                for _ in range(FPS * SECONDS_PER_IMAGE):
                    encoder.stdin.write(pixels)
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed to encode {output}")
    print(f"{output} ({len(paths) * SECONDS_PER_IMAGE}s): " + ", ".join(p.stem for p in paths))


if __name__ == "__main__":
    main()
