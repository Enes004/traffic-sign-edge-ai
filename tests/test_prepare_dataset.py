import io
import math
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prepare_dataset as gtsdb


OFFICIAL_GROUPS = {
    "prohibitory": {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16},
    "danger": {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    "mandatory": {33, 34, 35, 36, 37, 38, 39, 40},
    "other": {6, 12, 13, 14, 17, 32, 41, 42},
}


def annotation(class_id, box=(0, 0, 10, 10), image="00000.ppm"):
    xmin, ymin, xmax, ymax = box
    return gtsdb.Annotation(
        image=image,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        class_id=class_id,
    )


def test_official_mapping_is_complete_and_disjoint():
    all_ids = set().union(*OFFICIAL_GROUPS.values())
    assert all_ids == set(range(43))
    assert gtsdb.PROHIBITORY_SOURCE_CLASSES == OFFICIAL_GROUPS["prohibitory"]
    assert gtsdb.DANGER_SOURCE_CLASSES == OFFICIAL_GROUPS["danger"]
    assert gtsdb.MANDATORY_SOURCE_CLASSES == OFFICIAL_GROUPS["mandatory"]
    assert gtsdb.EXCLUDED_SOURCE_CLASSES == OFFICIAL_GROUPS["other"]
    gtsdb.validate_class_mapping()


def test_source_classes_map_to_expected_target_ids():
    expected_targets = {
        "prohibitory": 0,
        "danger": 1,
        "mandatory": 2,
    }
    for category, class_ids in OFFICIAL_GROUPS.items():
        for class_id in class_ids:
            if category == "other":
                assert gtsdb.source_class_to_target(class_id) is None
            else:
                assert (
                    gtsdb.source_class_to_target(class_id)
                    == expected_targets[category]
                )


def test_unknown_source_class_is_rejected():
    with pytest.raises(ValueError):
        gtsdb.source_class_to_target(43)


def test_bbox_to_yolo_known_gtsdb_box():
    x_center, y_center, width, height = gtsdb.bbox_to_yolo(
        774, 411, 815, 446, 1360, 800
    )
    assert x_center == pytest.approx(((774 + 815) / 2.0) / 1360)
    assert y_center == pytest.approx(((411 + 446) / 2.0) / 800)
    assert width == pytest.approx((815 - 774) / 1360)
    assert height == pytest.approx((446 - 411) / 800)


def test_bbox_to_yolo_full_image():
    assert gtsdb.bbox_to_yolo(0, 0, 1360, 800, 1360, 800) == (
        0.5,
        0.5,
        1.0,
        1.0,
    )


@pytest.mark.parametrize(
    "box,width,height",
    [
        ((0, 0, 0, 10), 1360, 800),
        ((10, 10, 10, 20), 1360, 800),
        ((-1, 0, 10, 10), 1360, 800),
        ((0, 0, 1361, 10), 1360, 800),
        ((0, 0, 10, 801), 1360, 800),
        ((0, -1, 10, 10), 1360, 800),
        ((0, 0, 10, 10), 0, 800),
    ],
)
def test_bbox_to_yolo_rejects_invalid_bounds(box, width, height):
    with pytest.raises(ValueError):
        gtsdb.bbox_to_yolo(*box, width, height)


def test_format_yolo_label_is_normalized_and_deterministic():
    annotations = [
        annotation(18, (0, 0, 1360, 800)),
        annotation(0, (774, 411, 815, 446)),
    ]
    text = gtsdb.format_yolo_label(annotations, 1360, 800)
    assert text.splitlines() == [
        "0 0.584191 0.535625 0.030147 0.043750",
        "1 0.500000 0.500000 1.000000 1.000000",
    ]
    assert text.endswith("\n")
    assert gtsdb.format_yolo_label([], 1360, 800) == ""


def test_format_yolo_label_rejects_excluded_class():
    with pytest.raises(ValueError):
        gtsdb.format_yolo_label([annotation(12)], 1360, 800)


def test_parse_gt_reads_valid_rows(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "00000.ppm;774;411;815;446;11\n"
        "\n"
        "00001.ppm;0;0;10;10;33\n",
        encoding="utf-8",
    )
    annotations = gtsdb.parse_gt(gt_path)
    assert annotations == [
        gtsdb.Annotation("00000.ppm", 774, 411, 815, 446, 11),
        gtsdb.Annotation("00001.ppm", 0, 0, 10, 10, 33),
    ]


@pytest.mark.parametrize(
    "content",
    [
        "00000.ppm;1;2;3;4\n",
        "bad.ppm;1;2;3;4;5\n",
        "00000.ppm;1;2;3;4;43\n",
        "00000.ppm;1;2;3;4;five\n",
    ],
)
def test_parse_gt_rejects_invalid_rows(tmp_path, content):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(content, encoding="utf-8")
    with pytest.raises(gtsdb.PreparationError):
        gtsdb.parse_gt(gt_path)


def test_parse_gt_deduplicates_exact_repeated_rows(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "00000.ppm;1;2;3;4;14\n00000.ppm;1;2;3;4;14\n",
        encoding="utf-8",
    )
    duplicates = []
    annotations = gtsdb.parse_gt(gt_path, duplicate_sink=duplicates)
    assert len(annotations) == 1
    assert len(duplicates) == 1
    assert duplicates[0] == gtsdb.Annotation("00000.ppm", 1, 2, 3, 4, 14)


def test_select_images_drops_whole_image_with_excluded_annotation():
    names = ["00000.ppm", "00001.ppm", "00002.ppm"]
    by_image = {
        "00000.ppm": [
            annotation(1, image="00000.ppm"),
            annotation(12, image="00000.ppm"),
        ],
        "00001.ppm": [annotation(2, image="00001.ppm")],
        "00002.ppm": [],
    }
    result = gtsdb.select_images(names, by_image, include_unannotated=True)
    assert set(result.kept) == {"00001.ppm", "00002.ppm"}
    assert result.kept["00002.ppm"] == []
    assert result.dropped_images == ["00000.ppm"]
    assert result.stats["dropped_image_count"] == 1
    assert result.stats["dropped_annotation_count"] == 2
    assert result.stats["dropped_target_annotation_count"] == 1
    assert result.stats["dropped_other_annotation_count"] == 1
    assert result.stats["dropped_mixed_image_count"] == 1
    assert result.stats["kept_annotation_count"] == 1
    assert result.stats["kept_negative_image_count"] == 1


def test_select_images_can_exclude_unannotated_images():
    names = ["00000.ppm", "00001.ppm"]
    by_image = {"00000.ppm": [annotation(1, image="00000.ppm")]}
    result = gtsdb.select_images(names, by_image, include_unannotated=False)
    assert set(result.kept) == {"00000.ppm"}
    assert result.excluded_unannotated == ["00001.ppm"]
    assert result.stats["excluded_unannotated_count"] == 1


def test_compute_split_counts_sums_to_total():
    for total in range(3, 100):
        counts = gtsdb.compute_split_counts(total, (0.70, 0.15, 0.15))
        assert sum(counts) == total
        assert all(count > 0 for count in counts)


def test_split_is_deterministic_and_disjoint():
    names = [f"{index:05d}.ppm" for index in range(100)]
    first = gtsdb.split_images(names, seed=42, ratios=(0.70, 0.15, 0.15))
    second = gtsdb.split_images(names, seed=42, ratios=(0.70, 0.15, 0.15))
    assert first == second
    assigned = first["train"] + first["val"] + first["test"]
    assert len(assigned) == len(set(assigned)) == len(names)
    assert set(assigned) == set(names)
    assert all(first[split] for split in gtsdb.SPLIT_NAMES)


def test_split_seed_changes_assignment():
    names = [f"{index:05d}.ppm" for index in range(100)]
    first = gtsdb.split_images(names, seed=42, ratios=(0.70, 0.15, 0.15))
    second = gtsdb.split_images(names, seed=43, ratios=(0.70, 0.15, 0.15))
    assert first != second


def test_official_split_keeps_test_images_separate():
    names = [f"{index:05d}.ppm" for index in range(900)]
    splits = gtsdb.split_images(
        names, seed=42, ratios=(0.70, 0.15, 0.15), mode="official"
    )
    assert all(
        gtsdb.OFFICIAL_TEST_START
        <= int(Path(name).stem)
        <= gtsdb.OFFICIAL_TEST_END
        for name in splits["test"]
    )
    assert all(
        int(Path(name).stem) < gtsdb.OFFICIAL_TEST_START
        for name in splits["train"] + splits["val"]
    )
    assert len(splits["test"]) == gtsdb.OFFICIAL_TEST_IMAGE_COUNT
    assert all(splits[split] for split in gtsdb.SPLIT_NAMES)


def test_zip_source_is_extracted_and_reused(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(gtsdb, "SOURCE_IMAGE_SIZE", (8, 8))
    archive = tmp_path / "FullIJCNN2013.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("FullIJCNN2013/gt.txt", "00000.ppm;1;1;4;4;1\n")
        for index in range(3):
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buffer, format="PPM")
            zip_file.writestr(
                f"FullIJCNN2013/{index:05d}.ppm", buffer.getvalue()
            )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    root = gtsdb._extract_archive(archive, data_dir, expected_image_count=3)
    assert root == (data_dir / "FullIJCNN2013").resolve()
    assert (root / "gt.txt").is_file()
    assert len(list(root.glob("*.ppm"))) == 3
    assert gtsdb._extract_archive(archive, data_dir, expected_image_count=3) == root


def test_manifest_entry_requires_the_same_source_file():
    entry = {
        "source": "00001.ppm",
        "source_size": 3264016,
        "source_mtime_ns": 1353974400000000000,
        "jpeg_quality": 95,
    }
    assert gtsdb._manifest_entry_matches(
        entry, "00001.ppm", 3264016, 1353974400000000000, 95
    )
    assert not gtsdb._manifest_entry_matches(
        entry, "00002.ppm", 3264016, 1353974400000000000, 95
    )
    assert not gtsdb._manifest_entry_matches(
        entry, "00001.ppm", 3264016, 1353974400000000000, 90
    )


def test_official_archive_size_constant_is_reasonable():
    assert gtsdb.OFFICIAL_ARCHIVE_SIZE > 1_000_000_000
    assert math.isfinite(gtsdb.OFFICIAL_ARCHIVE_SIZE)
