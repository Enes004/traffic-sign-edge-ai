import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("psutil")
pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analyze_object_size as aos


def label(box, cls=0):
    return np.array([[cls, *box]], dtype=np.float32)


def prediction(box, score, cls=0):
    return np.array([[*box, score, cls]], dtype=np.float32)


def test_assign_size_groups_tertiles_have_balanced_counts():
    areas = np.arange(1, 10, dtype=np.float64)
    group_ids, thresholds = aos.assign_size_groups(areas)
    assert np.bincount(group_ids, minlength=3).tolist() == [3, 3, 3]
    assert group_ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert thresholds["small_max_area_px2"] == pytest.approx(11 / 3)
    assert thresholds["medium_max_area_px2"] == pytest.approx(19 / 3)
    assert thresholds["quantiles"] == [1 / 3, 2 / 3]


def test_assign_size_groups_keeps_equal_areas_together():
    areas = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.float64)
    group_ids, _ = aos.assign_size_groups(areas)
    assert group_ids.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize(
    "areas",
    [
        np.array([], dtype=np.float64),
        np.array([-1.0]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_assign_size_groups_rejects_invalid_areas(areas):
    with pytest.raises(ValueError):
        aos.assign_size_groups(areas)


def test_collect_gt_objects_uses_original_pixel_area():
    labels = {
        "img": np.array([[0, 10, 20, 30, 60]], dtype=np.float32),
    }
    objects = aos.collect_gt_objects([Path("img.jpg")], labels)
    assert len(objects) == 1
    assert objects[0]["image"] == "img"
    assert objects[0]["object_index"] == 0
    assert objects[0]["class_id"] == 0
    assert objects[0]["class_name"] == "prohibitory"
    assert objects[0]["area_px2"] == pytest.approx(800.0)


def test_load_ground_truth_for_images_reads_original_dimensions(tmp_path):
    image_path = tmp_path / "00001.jpg"
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    assert aos.cv2.imwrite(str(image_path), image)
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "00001.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    labels = aos.load_ground_truth_for_images([image_path], label_dir)

    np.testing.assert_allclose(
        labels["00001"][0], [0, 40, 15, 60, 35], atol=1e-6
    )


def test_load_ground_truth_for_images_requires_label(tmp_path):
    image_path = tmp_path / "missing.jpg"
    assert aos.cv2.imwrite(
        str(image_path), np.zeros((10, 10, 3), dtype=np.uint8)
    )
    with pytest.raises(FileNotFoundError):
        aos.load_ground_truth_for_images([image_path], tmp_path / "labels")


def test_match_requires_same_class():
    labels = label([0, 0, 10, 10], cls=0)
    assert aos.match_gt_unique(labels, prediction([0, 0, 10, 10], 0.9, 0))[0]
    assert not aos.match_gt_unique(
        labels, prediction([0, 0, 10, 10], 0.9, 1)
    )[0]


def test_match_confidence_threshold_is_inclusive():
    labels = label([0, 0, 10, 10])
    assert aos.match_gt_unique(
        labels, prediction([0, 0, 10, 10], 0.25), conf_threshold=0.25
    )[0]
    assert not aos.match_gt_unique(
        labels, prediction([0, 0, 10, 10], 0.249999), conf_threshold=0.25
    )[0]


def test_match_iou_threshold_is_inclusive():
    labels = label([0, 0, 10, 10])
    assert aos.match_gt_unique(
        labels, prediction([0, 0, 5, 10], 0.9), iou_threshold=0.50
    )[0]
    assert not aos.match_gt_unique(
        labels, prediction([0, 0, 4.9, 10], 0.9), iou_threshold=0.50
    )[0]


def test_match_assigns_one_prediction_to_one_gt_object():
    labels = np.array(
        [[0, 0, 0, 10, 10], [0, 10, 0, 20, 10]], dtype=np.float32
    )
    matched = aos.match_gt_unique(
        labels, prediction([0, 0, 10, 10], 0.9)
    )
    assert matched.tolist() == [True, False]


def test_match_does_not_let_lower_confidence_duplicate_claim_gt():
    labels = label([0, 0, 10, 10])
    predictions = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [0, 0, 10, 10, 0.8, 0],
        ],
        dtype=np.float32,
    )
    assert aos.match_gt_unique(labels, predictions).tolist() == [True]


def test_match_two_distinct_gt_objects():
    labels = np.array(
        [[0, 0, 0, 10, 10], [0, 20, 0, 30, 10]], dtype=np.float32
    )
    predictions = np.array(
        [
            [0, 0, 10, 10, 0.8, 0],
            [20, 0, 30, 10, 0.9, 0],
        ],
        dtype=np.float32,
    )
    assert aos.match_gt_unique(labels, predictions).all()


def test_match_chooses_best_available_gt_object():
    labels = np.array(
        [[0, 0, 0, 10, 10], [0, 2, 0, 12, 10]], dtype=np.float32
    )
    matched = aos.match_gt_unique(
        labels, prediction([0, 0, 10, 10], 0.9)
    )
    assert matched.tolist() == [True, False]


def test_match_empty_inputs_have_expected_shapes():
    empty_labels = np.empty((0, 5), dtype=np.float32)
    empty_predictions = np.empty((0, 6), dtype=np.float32)
    assert aos.match_gt_unique(empty_labels, empty_predictions).shape == (0,)
    assert aos.match_gt_unique(
        label([0, 0, 10, 10]), empty_predictions
    ).tolist() == [False]
    assert aos.match_gt_unique(
        empty_labels, prediction([0, 0, 10, 10], 0.9)
    ).shape == (0,)


def test_decode_prediction_feeds_unique_matcher():
    output = np.zeros((1, 7, 1), dtype=np.float32)
    output[0, :4, 0] = [50, 50, 20, 20]
    output[0, 4:, 0] = [0.9, 0.01, 0.01]
    predictions = aos.bench.decode_prediction(
        output, (1.0, 0.0, 0.0, 100, 100), conf_threshold=0.25
    )
    labels = np.array([[0, 40, 40, 60, 60]], dtype=np.float32)
    assert aos.match_gt_unique(labels, predictions)[0]


def test_build_object_slices_aligns_images_and_objects():
    labels = {
        "a": np.zeros((2, 5), dtype=np.float32),
        "b": np.zeros((1, 5), dtype=np.float32),
    }
    slices, total = aos.build_object_slices(
        [Path("a.jpg"), Path("b.jpg")], labels
    )
    assert slices == {"a": (0, 2), "b": (2, 3)}
    assert total == 3


def test_run_model_fills_per_gt_matches_without_inference(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "img.jpg"
    assert aos.cv2.imwrite(
        str(image_path), np.zeros((100, 100, 3), dtype=np.uint8)
    )
    labels_by_stem = {
        "img": np.array([[0, 0, 0, 10, 10]], dtype=np.float32)
    }
    captured = {}

    class FakeSession:
        def run(self, output_names, feed):
            return [np.zeros((1, 7, 1), dtype=np.float32)]

    def fake_preprocess(image):
        return np.zeros((1, 3, 640, 640), dtype=np.float32), (
            1.0,
            0.0,
            0.0,
            100,
            100,
        )

    def fake_decode(
        output, geometry, conf_threshold, max_det, nms_iou
    ):
        captured["conf_threshold"] = conf_threshold
        return np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)

    monkeypatch.setattr(
        aos, "load_session", lambda model_path, threads: (FakeSession(), "images")
    )
    monkeypatch.setattr(aos.bench, "preprocess", fake_preprocess)
    monkeypatch.setattr(aos.bench, "decode_prediction", fake_decode)

    matched = aos.run_model(
        Path("model.onnx"),
        [image_path],
        labels_by_stem,
        {"img": (0, 1)},
        1,
        conf=0.25,
        match_iou=0.50,
        nms_iou=0.70,
        max_det=300,
        warmup=0,
        threads=1,
        model_label="synthetic",
    )

    assert matched.tolist() == [True]
    assert captured["conf_threshold"] < 0.25


def test_wilson_interval_bounds():
    low, high = aos.wilson_interval(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0
    assert aos.wilson_interval(0, 0) == (None, None)
    assert aos.wilson_interval(0, 10)[0] == pytest.approx(0.0)
    assert aos.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        aos.wilson_interval(11, 10)


def test_compute_recall_rows_and_signed_delta():
    objects = [
        {"area_px2": 1.0, "class_id": 0},
        {"area_px2": 2.0, "class_id": 0},
        {"area_px2": 10.0, "class_id": 1},
        {"area_px2": 11.0, "class_id": 1},
        {"area_px2": 100.0, "class_id": 2},
        {"area_px2": 110.0, "class_id": 2},
    ]
    group_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    thresholds = {
        "small_max_area_px2": 5.0,
        "medium_max_area_px2": 50.0,
    }
    matched_by_model = {
        (0, "fp32"): np.array(
            [True, True, True, True, False, False], dtype=bool
        ),
        (0, "int8"): np.array(
            [True, False, True, False, False, True], dtype=bool
        ),
    }

    rows = aos.compute_recall_rows(
        objects, group_ids, matched_by_model, [0], thresholds
    )

    assert len(rows) == 3
    small, medium, large = rows
    assert small["gt_count"] == 2
    assert small["fp32_recall"] == pytest.approx(1.0)
    assert small["int8_recall"] == pytest.approx(0.5)
    assert small["delta_int8_minus_fp32"] == pytest.approx(-0.5)
    assert medium["fp32_recall"] == pytest.approx(1.0)
    assert medium["int8_recall"] == pytest.approx(0.5)
    assert large["fp32_recall"] == pytest.approx(0.0)
    assert large["int8_recall"] == pytest.approx(0.5)
    assert large["delta_int8_minus_fp32"] == pytest.approx(0.5)
    assert small["small_max_area_px2"] == 5.0
    assert small["medium_max_area_px2"] == 50.0


def test_compute_recall_rows_rejects_misaligned_arrays():
    objects = [{"area_px2": 1.0, "class_id": 0}]
    group_ids = np.array([0], dtype=np.int8)
    thresholds = {
        "small_max_area_px2": 5.0,
        "medium_max_area_px2": 50.0,
    }
    matched_by_model = {
        (0, "fp32"): np.array([True], dtype=bool),
        (0, "int8"): np.array([], dtype=bool),
    }
    with pytest.raises(ValueError):
        aos.compute_recall_rows(
            objects, group_ids, matched_by_model, [0], thresholds
        )


def test_compute_class_recall_rows_includes_absent_class():
    objects = [
        {"class_id": 0},
        {"class_id": 0},
        {"class_id": 2},
    ]
    matched_by_model = {
        (0, "fp32"): np.array([True, True, False], dtype=bool),
        (0, "int8"): np.array([True, False, False], dtype=bool),
    }
    rows = aos.compute_class_recall_rows(objects, matched_by_model, [0])
    by_key = {(row["format"], row["class_name"]): row for row in rows}
    assert by_key[("fp32", "prohibitory")]["recall"] == pytest.approx(1.0)
    assert by_key[("int8", "prohibitory")]["recall"] == pytest.approx(0.5)
    assert by_key[("fp32", "mandatory")]["recall"] == pytest.approx(0.0)
    assert by_key[("fp32", "danger")]["gt_count"] == 0
    assert by_key[("fp32", "danger")]["recall"] is None


def test_summarize_size_groups_records_counts_ranges_and_classes():
    objects = [
        {"area_px2": 1.0, "class_id": 0},
        {"area_px2": 2.0, "class_id": 1},
        {"area_px2": 10.0, "class_id": 2},
    ]
    group_ids = np.array([0, 0, 1], dtype=np.int8)
    thresholds = {
        "method": "tertiles_of_gt_bbox_area",
        "quantiles": [1 / 3, 2 / 3],
        "small_max_area_px2": 5.0,
        "medium_max_area_px2": 50.0,
        "classification_rule": "small/medium/large",
    }
    summary = aos.summarize_size_groups(objects, group_ids, thresholds)
    assert summary["thresholds"]["small_max_area_px2"] == 5.0
    assert summary["groups"][0]["gt_count"] == 2
    assert summary["groups"][0]["class_counts"] == {
        "prohibitory": 1,
        "danger": 1,
        "mandatory": 0,
    }
    assert summary["groups"][0]["area_min_px2"] == 1.0
    assert summary["groups"][0]["area_max_px2"] == 2.0
    assert summary["groups"][1]["gt_count"] == 1
    assert summary["groups"][2]["gt_count"] == 0
    assert summary["groups"][2]["area_min_px2"] is None


def test_summarize_results_across_seeds():
    rows = [
        {
            "seed": 0,
            "size_group": "small",
            "fp32_recall": 1.0,
            "int8_recall": 0.9,
            "delta_int8_minus_fp32": -0.1,
        },
        {
            "seed": 1,
            "size_group": "small",
            "fp32_recall": 0.8,
            "int8_recall": 0.8,
            "delta_int8_minus_fp32": 0.0,
        },
        {
            "seed": 0,
            "size_group": "medium",
            "fp32_recall": 0.5,
            "int8_recall": 0.4,
            "delta_int8_minus_fp32": -0.1,
        },
    ]
    summary = aos.summarize_results(rows)
    small = summary["by_group"][0]
    assert small["fp32_recall_mean"] == pytest.approx(0.9)
    assert small["int8_recall_mean"] == pytest.approx(0.85)
    assert small["delta_mean"] == pytest.approx(-0.05)
    assert small["delta_min"] == pytest.approx(-0.1)
    assert small["delta_max"] == pytest.approx(0.0)
    assert summary["by_group"][2]["delta_mean"] is None


def test_build_limitations_mentions_class_imbalance_and_small_cells():
    objects = [
        {"class_id": 0, "class_name": "prohibitory"} for _ in range(10)
    ] + [{"class_id": 2, "class_name": "mandatory"} for _ in range(2)]
    group_summary = {
        "groups": [
            {
                "name": "small",
                "class_counts": {
                    "prohibitory": 5,
                    "danger": 0,
                    "mandatory": 1,
                },
            },
            {
                "name": "medium",
                "class_counts": {
                    "prohibitory": 3,
                    "danger": 0,
                    "mandatory": 1,
                },
            },
            {
                "name": "large",
                "class_counts": {
                    "prohibitory": 2,
                    "danger": 0,
                    "mandatory": 0,
                },
            },
        ]
    }
    limitations = aos.build_limitations(objects, group_summary)
    text = " ".join(limitations)
    assert "class" in text
    assert "Small-sample" in text
    assert "mandatory" in text
    assert "227" in text


def test_write_outputs_creates_csv_json_and_png(tmp_path):
    objects = [
        {"area_px2": 1.0, "class_id": 0},
        {"area_px2": 2.0, "class_id": 0},
        {"area_px2": 10.0, "class_id": 1},
        {"area_px2": 11.0, "class_id": 1},
        {"area_px2": 100.0, "class_id": 2},
        {"area_px2": 110.0, "class_id": 2},
    ]
    group_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
    thresholds = {
        "method": "tertiles_of_gt_bbox_area",
        "quantiles": [1 / 3, 2 / 3],
        "small_max_area_px2": 5.0,
        "medium_max_area_px2": 50.0,
        "classification_rule": "small/medium/large",
    }
    group_summary = aos.summarize_size_groups(
        objects, group_ids, thresholds
    )
    matched_by_model = {
        (0, "fp32"): np.array(
            [True, True, True, True, False, False], dtype=bool
        ),
        (0, "int8"): np.array(
            [True, False, True, False, False, True], dtype=bool
        ),
    }
    rows = aos.compute_recall_rows(
        objects, group_ids, matched_by_model, [0], thresholds
    )

    csv_path = tmp_path / "summary.csv"
    group_csv_path = tmp_path / "groups.csv"
    json_path = tmp_path / "summary.json"
    png_path = tmp_path / "summary.png"
    aos.write_summary_csv(rows, csv_path)
    aos.write_group_csv(group_summary, group_csv_path)
    aos.write_json({"results": rows, "size_groups": group_summary}, json_path)
    aos.plot_results(rows, group_summary, png_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "delta_int8_minus_fp32" in header
    with group_csv_path.open(newline="", encoding="utf-8") as handle:
        group_rows = list(csv.DictReader(handle))
    assert group_rows[0]["size_group"] == "small"
    assert group_rows[0]["gt_count"] == "2"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 3
    assert payload["size_groups"]["thresholds"]["small_max_area_px2"] == 5.0
    assert png_path.stat().st_size > 0
