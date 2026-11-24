import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verify_onnx_suite as suite


def test_expected_anchors_for_640_and_320():
    assert suite.expected_anchors(640) == 8400
    assert suite.expected_anchors(320) == 2100


@pytest.mark.parametrize("bad_size", [0, -320, 300, 321, True, False])
def test_expected_anchors_rejects_invalid_sizes(bad_size):
    with pytest.raises(ValueError):
        suite.expected_anchors(bad_size)


def test_validate_io_metadata_accepts_expected_320_and_640():
    suite.validate_io_metadata(
        [1, 3, 320, 320], "tensor(float)", [1, 7, 2100], "tensor(float)", imgsz=320
    )
    suite.validate_io_metadata(
        [1, 3, 640, 640], "tensor(float)", [1, 7, 8400], "tensor(float)", imgsz=640
    )


@pytest.mark.parametrize(
    "in_shape,in_type,out_shape,out_type",
    [
        ([1, 3, 640, 640], "tensor(float)", [1, 7, 2100], "tensor(float)"),
        ([1, 3, 320, 320], "tensor(float16)", [1, 7, 2100], "tensor(float)"),
        ([1, 3, 320, 320], "tensor(float)", [1, 7, 8400], "tensor(float)"),
        ([1, 3, 320, 320], "tensor(float)", [1, 8, 2100], "tensor(float)"),
        ([1, 3, 320, 320], "tensor(float)", [1, 7, 2100], "tensor(uint8)"),
    ],
)
def test_validate_io_metadata_rejects_mismatches(in_shape, in_type, out_shape, out_type):
    with pytest.raises(ValueError):
        suite.validate_io_metadata(in_shape, in_type, out_shape, out_type, imgsz=320)


def test_validate_quantized_graph_rules():
    suite.validate_quantized_graph("fp32", 0, 0)
    suite.validate_quantized_graph("int8", 194, 398)
    with pytest.raises(ValueError):
        suite.validate_quantized_graph("int8", 0, 398)
    with pytest.raises(ValueError):
        suite.validate_quantized_graph("int8", 194, 0)
    with pytest.raises(ValueError):
        suite.validate_quantized_graph("fp16", 0, 0)


def test_compute_parity_status_pass_and_check():
    pt = {"P": 0.90, "R": 0.85, "mAP50": 0.8800, "mAP50_95": 0.7000}
    onnx_same = {"P": 0.90, "R": 0.85, "mAP50": 0.8800, "mAP50_95": 0.7000}
    res_pass = suite.compute_parity_status(pt, onnx_same)
    assert res_pass["status"] == "PASS"
    assert res_pass["max_map_delta"] == pytest.approx(0.0)

    onnx_drift = {"P": 0.90, "R": 0.85, "mAP50": 0.8860, "mAP50_95": 0.7000}
    res_check = suite.compute_parity_status(pt, onnx_drift)
    assert res_check["status"] == "CHECK"
    assert res_check["max_map_delta"] == pytest.approx(0.0060)


def test_summarize_metrics_mean_and_population_std():
    rows = [
        {"P": 0.8, "R": 0.7, "mAP50": 0.80, "mAP50_95": 0.60},
        {"P": 0.9, "R": 0.8, "mAP50": 0.85, "mAP50_95": 0.65},
        {"P": 1.0, "R": 0.9, "mAP50": 0.90, "mAP50_95": 0.70},
    ]
    summary = suite.summarize_metrics(rows)
    assert summary["mAP50"]["mean"] == pytest.approx(0.85)
    assert summary["mAP50"]["std"] == pytest.approx(math.sqrt(((0.05**2) * 2) / 3))
    assert summary["P"]["mean"] == pytest.approx(0.9)


def test_summarize_metrics_rejects_empty():
    with pytest.raises(ValueError):
        suite.summarize_metrics([])
