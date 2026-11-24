import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("psutil")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark_pi_v3 as bp


@pytest.mark.parametrize(("imgsz", "anchors"), [(320, 2100), (640, 8400)])
def test_expected_anchors_match_yolo_grids(imgsz, anchors):
    assert bp.expected_anchors(imgsz) == anchors


def test_expected_anchors_rejects_invalid_input_size():
    with pytest.raises(ValueError):
        bp.expected_anchors(321)


@pytest.mark.parametrize(("imgsz", "shape"), [
    (320, (1, 7, 2100)),
    (640, (1, 7, 8400)),
])
def test_expected_output_shape_per_resolution(imgsz, shape):
    assert bp.expected_output_shape(imgsz) == shape


@pytest.mark.parametrize("imgsz", [320, 640])
def test_validate_onnx_shapes_accepts_supported_resolution(imgsz):
    output_shape = bp.expected_output_shape(imgsz)
    bp.validate_onnx_shapes([1, 3, imgsz, imgsz], output_shape, imgsz)


def test_validate_onnx_shapes_rejects_640_input_for_320():
    with pytest.raises(ValueError, match="input shape"):
        bp.validate_onnx_shapes([1, 3, 640, 640], [1, 7, 8400], 320)


def test_validate_output_shape_rejects_640_output_for_320():
    with pytest.raises(ValueError, match="output shape"):
        bp.validate_output_shape([1, 7, 8400], 320)


def test_validate_output_shape_accepts_320_runtime_output():
    bp.validate_output_shape((1, 7, 2100), 320)


@pytest.mark.parametrize(("imgsz", "filename"), [
    (320, "yolo26n_img320_seed2_int8.onnx"),
    (640, "yolo26n_img640_seed2_int8.onnx"),
])
def test_model_filename_uses_selected_resolution(imgsz, filename):
    assert bp.model_filename(imgsz, 2, "int8") == filename


def test_preprocess_320_produces_letterboxed_tensor_and_geometry():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor, geometry = bp.preprocess(image, input_size=320)
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert geometry == (0.25, 0, 70, 1280, 720)


def test_preprocess_640_remains_the_unchanged_default():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    default_tensor, default_geometry = bp.preprocess(image)
    explicit_tensor, explicit_geometry = bp.preprocess(image, input_size=640)
    assert default_tensor.shape == (1, 3, 640, 640)
    np.testing.assert_array_equal(default_tensor, explicit_tensor)
    assert default_geometry == explicit_geometry == (0.5, 0, 140, 1280, 720)


def test_decode_320_maps_boxes_back_to_original_pixels():
    output = np.zeros((1, 7, 2100), dtype=np.float32)
    output[0, :, 0] = [160, 160, 40, 40, 0.9, 0.01, 0.01]
    predictions = bp.decode_prediction(
        output, (0.25, 0, 70, 1280, 720), conf_threshold=0.5, nms_iou=0.5
    )
    assert predictions.shape == (1, 6)
    np.testing.assert_allclose(predictions[0, :4], [560, 280, 720, 440], atol=1e-6)
    assert predictions[0, 4] == pytest.approx(0.9)
    assert predictions[0, 5] == 0


def test_decode_640_geometry_is_unchanged():
    output = np.zeros((1, 7, 8400), dtype=np.float32)
    output[0, :, 0] = [320, 320, 80, 80, 0.9, 0.01, 0.01]
    predictions = bp.decode_prediction(
        output, (0.5, 0, 140, 1280, 720), conf_threshold=0.5, nms_iou=0.5
    )
    np.testing.assert_allclose(predictions[0, :4], [560, 280, 720, 440], atol=1e-6)
