#!/usr/bin/env python3
"""Static INT8 (QDQ) quantization for YOLO26n FP32 ONNX using train-split calibration."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def letterbox(img: np.ndarray, new_shape: int = 640) -> np.ndarray:
    """Resize and pad image to square (new_shape x new_shape) matching Ultralytics."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


class GTSDBCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int = 640):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self._iter = iter(self.image_paths)

    def get_next(self):
        img_path = next(self._iter, None)
        if img_path is None:
            return None
        img = cv2.imread(str(img_path))
        img = letterbox(img, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = img.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        return {self.input_name: tensor}

    def rewind(self):
        self._iter = iter(self.image_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Static INT8 quantization for YOLO26n ONNX.")
    parser.add_argument("--seed", type=int, default=0, help="Model seed (0, 1, or 2)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--calib-count", type=int, default=100, help="Number of train images for calibration")
    parser.add_argument("--calib-seed", type=int, default=42, help="Seed for deterministic calibration sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = REPO_ROOT / "models"
    fp32_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}_fp32.onnx"
    prep_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}_prep.onnx"
    int8_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}_int8.onnx"
    train_img_dir = REPO_ROOT / "data" / "gtsdb" / "images" / "train"

    if not fp32_path.is_file():
        raise FileNotFoundError(f"Missing FP32 ONNX: {fp32_path}")

    all_train_imgs = sorted(train_img_dir.glob("*.jpg"))
    rng = random.Random(args.calib_seed)
    calib_imgs = rng.sample(all_train_imgs, min(args.calib_count, len(all_train_imgs)))
    print(f"Selected {len(calib_imgs)} calibration images strictly from train split.")

    print("Running quant_pre_process...")
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)

    sess = ort.InferenceSession(str(prep_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = GTSDBCalibrationReader(calib_imgs, input_name=input_name, imgsz=args.imgsz)

    print(f"Quantizing to INT8 (QDQ, Conv layers) -> {int8_path.name}...")
    quantize_static(
        model_input=str(prep_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        per_channel=True,
    )

    prep_path.unlink(missing_ok=True)

    # Verify quantized graph nodes
    model_proto = onnx.load(str(int8_path))
    q_nodes = sum(1 for n in model_proto.graph.node if n.op_type == "QuantizeLinear")
    dq_nodes = sum(1 for n in model_proto.graph.node if n.op_type == "DequantizeLinear")

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"\n--- Quantization Summary (Seed {args.seed}) ---")
    print(f"FP32 Size : {fp32_mb:.2f} MB")
    print(f"INT8 Size : {int8_mb:.2f} MB ({(1 - int8_mb / fp32_mb) * 100:.1f}% reduction)")
    print(f"QDQ Nodes : QuantizeLinear={q_nodes}, DequantizeLinear={dq_nodes}")


if __name__ == "__main__":
    main()
