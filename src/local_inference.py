"""Shared local CPU inference for the qualitative analysis and demo."""
from pathlib import Path

import cv2
import onnxruntime as ort

from benchmark_pi_v3 import (
    decode_prediction, model_filename, preprocess, validate_onnx_shapes,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "data/gtsdb/images/test"
CONFIDENCE = 0.25
NMS_IOU = 0.70
MAX_DET = 300


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot decode {path}")
    return image


class Detector:
    def __init__(self, resolution, variant, seed):
        self.resolution = resolution
        self.path = ROOT / "models" / model_filename(resolution, seed, variant)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        validate_onnx_shapes(
            self.input.shape, self.session.get_outputs()[0].shape, resolution
        )

    def predict(self, image):
        tensor, geometry = preprocess(image, self.resolution)
        output = self.session.run(None, {self.input.name: tensor})[0]
        return decode_prediction(output, geometry, CONFIDENCE, MAX_DET, NMS_IOU)
