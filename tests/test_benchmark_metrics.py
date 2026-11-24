import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("psutil")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import benchmark_pi_v3 as bp


def detection(box, score, cls):
    return np.array([*box, score, cls], dtype=np.float32)


def raw_output(rows):
    output = np.zeros((1, 7, len(rows)), dtype=np.float32)
    for index, (cx, cy, width, height, scores) in enumerate(rows):
        output[0, :4, index] = [cx, cy, width, height]
        output[0, 4:, index] = scores
    return output


def test_iou_known_pairwise_values():
    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    boxes_b = np.array(
        [[0, 0, 10, 5], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32
    )
    expected = np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(bp.iou_matrix(boxes_a, boxes_b), expected, atol=1e-6)


def test_iou_contained_and_touching_boxes():
    boxes_a = np.array([[0, 0, 4, 4]], dtype=np.float32)
    boxes_b = np.array([[1, 1, 3, 3], [4, 0, 8, 4]], dtype=np.float32)
    expected = np.array([[0.25, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(bp.iou_matrix(boxes_a, boxes_b), expected, atol=1e-6)


def test_iou_empty_sets_have_expected_shapes():
    empty = np.empty((0, 4), dtype=np.float32)
    one = np.array([[0, 0, 1, 1]], dtype=np.float32)
    assert bp.iou_matrix(empty, one).shape == (0, 1)
    assert bp.iou_matrix(one, empty).shape == (1, 0)
    assert bp.iou_matrix(empty, empty).shape == (0, 0)


def test_iou_is_symmetric_and_bounded():
    rng = np.random.default_rng(123)
    top_left = rng.uniform(-20, 80, size=(25, 2))
    size = rng.uniform(1, 40, size=(25, 2))
    boxes_a = np.column_stack((top_left, top_left + size)).astype(np.float32)
    top_left = rng.uniform(-20, 80, size=(25, 2))
    size = rng.uniform(1, 40, size=(25, 2))
    boxes_b = np.column_stack((top_left, top_left + size)).astype(np.float32)
    iou = bp.iou_matrix(boxes_a, boxes_b)
    np.testing.assert_allclose(iou, bp.iou_matrix(boxes_b, boxes_a).T, atol=1e-6)
    assert iou.min() >= 0.0
    assert iou.max() <= 1.0


def test_nms_suppresses_lower_score_same_class_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int32)
    kept = bp.class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    assert kept.tolist() == [0]


def test_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int32)
    kept = bp.class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    assert kept.tolist() == [0, 1]


def test_nms_suppresses_only_above_the_iou_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int32)
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.5
    ).tolist() == [0, 1]
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.499
    ).tolist() == [0]


def test_nms_applies_max_det_to_the_global_score_order():
    boxes = np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 1, 2], dtype=np.int32)
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.5, max_det=2
    ).tolist() == [0, 1]
    classes = np.array([0, 0, 0], dtype=np.int32)
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.5, max_det=2
    ).tolist() == [0, 1]


def test_nms_is_greedy_and_suppresses_only_selected_neighbours():
    boxes = np.array(
        [[0, 0, 10, 10], [5, 0, 15, 10], [10, 0, 20, 10]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 0], dtype=np.int32)
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.3
    ).tolist() == [0, 2]
    assert bp.class_aware_nms(
        boxes, scores, classes, iou_threshold=0.4
    ).tolist() == [0, 1, 2]


def test_nms_empty_input_returns_empty_indices():
    kept = bp.class_aware_nms(
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int32),
    )
    assert kept.size == 0


def test_decode_expands_multilabel_candidates_per_anchor():
    output = raw_output(
        [
            (50, 50, 20, 20, (0.9, 0.1, 0.8)),
            (150, 150, 20, 20, (0.01, 0.01, 0.7)),
        ]
    )
    predictions = bp.decode_prediction(
        output, (1.0, 0.0, 0.0, 200, 200), conf_threshold=0.5, nms_iou=0.5
    )
    np.testing.assert_allclose(
        predictions[:, :4],
        [[40, 40, 60, 60], [40, 40, 60, 60], [140, 140, 160, 160]],
        atol=1e-6,
    )
    np.testing.assert_allclose(predictions[:, 4], [0.9, 0.8, 0.7], atol=1e-6)
    np.testing.assert_array_equal(predictions[:, 5], [0, 2, 2])


def test_decode_applies_class_aware_nms_to_raw_boxes():
    output = raw_output(
        [
            (50, 50, 20, 20, (0.9, 0.01, 0.01)),
            (51, 51, 20, 20, (0.8, 0.01, 0.01)),
        ]
    )
    predictions = bp.decode_prediction(
        output, (1.0, 0.0, 0.0, 200, 200), conf_threshold=0.5, nms_iou=0.5
    )
    assert predictions.shape == (1, 6)
    np.testing.assert_allclose(predictions[0, :4], [40, 40, 60, 60], atol=1e-6)
    assert predictions[0, 4] == pytest.approx(0.9)
    assert predictions[0, 5] == 0


def test_decode_maps_letterboxed_boxes_to_original_pixels():
    output = raw_output([(30, 40, 20, 20, (0.9, 0.01, 0.01))])
    predictions = bp.decode_prediction(
        output, (0.5, 10, 20, 200, 100), conf_threshold=0.5
    )
    np.testing.assert_allclose(predictions[0, :4], [20, 20, 60, 60], atol=1e-6)


def test_decode_clips_boxes_to_original_image():
    output = raw_output([(0, 0, 20, 20, (0.9, 0.01, 0.01))])
    predictions = bp.decode_prediction(
        output, (1.0, 0.0, 0.0, 100, 100), conf_threshold=0.5
    )
    np.testing.assert_allclose(predictions[0, :4], [0, 0, 10, 10], atol=1e-6)


def test_decode_confidence_threshold_is_strict():
    at_threshold = raw_output([(50, 50, 20, 20, (0.5, 0.01, 0.01))])
    above_threshold = raw_output([(50, 50, 20, 20, (0.5001, 0.01, 0.01))])
    assert bp.decode_prediction(
        at_threshold, (1.0, 0.0, 0.0, 100, 100), conf_threshold=0.5
    ).shape == (0, 6)
    assert bp.decode_prediction(
        above_threshold, (1.0, 0.0, 0.0, 100, 100), conf_threshold=0.5
    ).shape == (1, 6)


def test_decode_rejects_unexpected_output_shape():
    with pytest.raises(ValueError):
        bp.decode_prediction(
            np.zeros((1, 6, 10), dtype=np.float32), (1.0, 0.0, 0.0, 100, 100)
        )


def test_matching_perfect_same_class_detection():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([detection([0, 0, 10, 10], 0.9, 0)])
    correct = bp.true_positive_matrix(detections, labels)
    assert correct.shape == (1, 10)
    assert correct.all()


def test_matching_rejects_wrong_class():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([detection([0, 0, 10, 10], 0.9, 1)])
    assert not bp.true_positive_matrix(detections, labels).any()


def test_matching_requires_at_least_half_iou():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([detection([0, 0, 10, 4], 0.9, 0)])
    assert not bp.true_positive_matrix(detections, labels).any()


def test_matching_accepts_iou_exactly_at_threshold():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([detection([0, 0, 5, 10], 0.9, 0)])
    correct = bp.true_positive_matrix(detections, labels)
    assert correct[0, 0]
    assert not correct[0, 1:].any()


def test_matching_is_confidence_first_per_threshold():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    detections = np.array(
        [
            detection([0, 0, 10, 6], 0.9, 0),
            detection([0, 0, 10, 10], 0.8, 0),
        ]
    )
    correct = bp.true_positive_matrix(detections, labels)
    expected = np.zeros((2, 10), dtype=bool)
    expected[0, :3] = True
    expected[1, 3:] = True
    np.testing.assert_array_equal(correct, expected)


def test_matching_assigns_distinct_targets():
    labels = np.array(
        [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]], dtype=np.float32
    )
    detections = np.array(
        [
            detection([0, 0, 10, 10], 0.9, 0),
            detection([20, 20, 30, 30], 0.8, 0),
        ]
    )
    assert bp.true_positive_matrix(detections, labels).all()


def test_matching_empty_inputs_have_expected_shapes():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    empty_detections = np.empty((0, 6), dtype=np.float32)
    assert bp.true_positive_matrix(empty_detections, labels).shape == (0, 10)
    empty_labels = np.empty((0, 5), dtype=np.float32)
    detections = np.array([detection([0, 0, 10, 10], 0.9, 0)])
    assert bp.true_positive_matrix(detections, empty_labels).shape == (1, 10)


def test_compute_ap_perfect_detector_is_one():
    assert bp.compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)


def test_compute_ap_known_curves():
    assert bp.compute_ap(
        np.array([1.0]), np.array([0.5])
    ) == pytest.approx(0.5)
    assert bp.compute_ap(
        np.array([0.5]), np.array([1.0])
    ) == pytest.approx(51 / 101)
    assert bp.compute_ap(
        np.array([0.5, 1.0]), np.array([1.0, 0.5])
    ) == pytest.approx(76 / 101)
    assert bp.compute_ap(
        np.array([0.25, 0.5, 0.75, 1.0]),
        np.array([1.0, 0.8, 0.6, 0.4]),
    ) == pytest.approx(71 / 101)


def test_compute_ap_empty_and_zero_curves():
    assert bp.compute_ap(np.array([]), np.array([])) == 0.0
    assert bp.compute_ap(np.array([0.5]), np.array([0.0])) == 0.0


def test_evaluate_perfect_single_class():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    predictions = np.array([detection([0, 0, 10, 10], 0.9, 0)])
    result = bp.evaluate([(predictions, labels)])
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["mAP50_95"] == pytest.approx(1.0)
    assert result["gt_boxes"] == 1
    assert result["per_class"]["prohibitory"]["mAP50"] == pytest.approx(1.0)
    assert result["per_class"]["danger"]["mAP50"] is None


def test_evaluate_known_iou_06_case():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    predictions = np.array([detection([0, 0, 10, 6], 0.9, 0)])
    result = bp.evaluate([(predictions, labels)])
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["mAP50_95"] == pytest.approx(0.3)


def test_evaluate_false_positive_before_true_positive():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    predictions = np.array(
        [
            detection([20, 20, 30, 30], 0.9, 0),
            detection([0, 0, 10, 10], 0.8, 0),
        ]
    )
    result = bp.evaluate([(predictions, labels)])
    assert result["mAP50"] == pytest.approx(0.5)
    assert result["mAP50_95"] == pytest.approx(0.5)


def test_evaluate_wrong_class_and_missing_predictions_are_zero():
    labels = np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    wrong_class = np.array([detection([0, 0, 10, 10], 0.9, 1)])
    assert bp.evaluate([(wrong_class, labels)])["mAP50"] == 0.0
    no_predictions = np.empty((0, 6), dtype=np.float32)
    assert bp.evaluate([(no_predictions, labels)])["mAP50"] == 0.0


def test_evaluate_multiclass_and_absent_class_is_excluded():
    labels = np.array(
        [[0, 0, 0, 10, 10], [1, 20, 20, 30, 30]], dtype=np.float32
    )
    predictions = np.array(
        [
            detection([0, 0, 10, 10], 0.9, 0),
            detection([20, 20, 30, 30], 0.8, 1),
        ]
    )
    result = bp.evaluate([(predictions, labels)])
    assert result["mAP50"] == pytest.approx(1.0)
    assert result["mAP50_95"] == pytest.approx(1.0)
    assert result["gt_boxes"] == 2
    assert result["per_class"]["mandatory"]["mAP50"] is None


def test_evaluate_is_invariant_to_prediction_array_order():
    labels = np.array(
        [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]], dtype=np.float32
    )
    predictions = np.array(
        [
            detection([0, 0, 10, 6], 0.9, 0),
            detection([0, 0, 10, 10], 0.8, 0),
            detection([20, 20, 30, 30], 0.7, 0),
        ]
    )
    sorted_result = bp.evaluate([(predictions, labels)])
    reversed_result = bp.evaluate([(predictions[::-1], labels)])
    assert sorted_result["mAP50"] == pytest.approx(reversed_result["mAP50"])
    assert sorted_result["mAP50_95"] == pytest.approx(
        reversed_result["mAP50_95"]
    )


def test_evaluate_without_any_ground_truth_raises():
    predictions = np.array([detection([0, 0, 10, 10], 0.9, 0)])
    labels = np.empty((0, 5), dtype=np.float32)
    with pytest.raises(RuntimeError):
        bp.evaluate([(predictions, labels)])
