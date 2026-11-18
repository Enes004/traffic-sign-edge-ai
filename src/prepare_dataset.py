#!/usr/bin/env python3
"""Build a reproducible YOLO detection dataset from GTSDB FullIJCNN2013.

Official class-to-category mapping is transcribed from the ReadMe.txt bundled
with FullIJCNN2013 and from the official benchmark code TSD_readGTData.m:

prohibitory: 0,1,2,3,4,5,7,8,9,10,15,16
danger:      11,18,19,20,21,22,23,24,25,26,27,28,29,30,31
mandatory:   33,34,35,36,37,38,39,40
other:       6,12,13,14,17,32,41,42

The official category name is "prohibitory" (also called "prohibitive").
Objects in the "other" category are excluded from the three benchmark
categories. To avoid silently treating those annotated objects as background,
the default policy removes an entire image when it contains at least one
"other" annotation. No YOLO label is ever derived from an "other" class.

Coordinates follow the continuous convention used by the official GTSDB
Jaccard evaluation: width = xmax - xmin and height = ymax - ymin. They are
converted to normalized YOLO center/size values without clamping.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


PIPELINE_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGE_SIZE = (1360, 800)
OFFICIAL_IMAGE_COUNT = 900
OFFICIAL_GT_ROWS = 1213
OFFICIAL_UNIQUE_GT_ROWS = 1212
OFFICIAL_ANNOTATED_IMAGE_COUNT = 741
OFFICIAL_UNANNOTATED_IMAGE_COUNT = 159
OFFICIAL_TRAIN_IMAGE_COUNT = 600
OFFICIAL_TEST_IMAGE_COUNT = 300
OFFICIAL_TEST_START = 600
OFFICIAL_TEST_END = 899
OFFICIAL_ARCHIVE_NAME = "FullIJCNN2013.zip"
OFFICIAL_ARCHIVE_URL = (
    "https://sid.erda.dk/public/archives/"
    "ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip"
)
OFFICIAL_ARCHIVE_SIZE = 1661866983
SPLIT_NAMES = ("train", "val", "test")
TARGET_CLASS_NAMES = ("prohibitory", "danger", "mandatory")
TARGET_CLASS_IDS = {name: index for index, name in enumerate(TARGET_CLASS_NAMES)}


SOURCE_CLASS_INFO: Dict[int, Tuple[str, str]] = {
    0: ("speed limit 20", "prohibitory"),
    1: ("speed limit 30", "prohibitory"),
    2: ("speed limit 50", "prohibitory"),
    3: ("speed limit 60", "prohibitory"),
    4: ("speed limit 70", "prohibitory"),
    5: ("speed limit 80", "prohibitory"),
    6: ("restriction ends 80", "other"),
    7: ("speed limit 100", "prohibitory"),
    8: ("speed limit 120", "prohibitory"),
    9: ("no overtaking", "prohibitory"),
    10: ("no overtaking (trucks)", "prohibitory"),
    11: ("priority at next intersection", "danger"),
    12: ("priority road", "other"),
    13: ("give way", "other"),
    14: ("stop", "other"),
    15: ("no traffic both ways", "prohibitory"),
    16: ("no trucks", "prohibitory"),
    17: ("no entry", "other"),
    18: ("danger", "danger"),
    19: ("bend left", "danger"),
    20: ("bend right", "danger"),
    21: ("bend", "danger"),
    22: ("uneven road", "danger"),
    23: ("slippery road", "danger"),
    24: ("road narrows", "danger"),
    25: ("construction", "danger"),
    26: ("traffic signal", "danger"),
    27: ("pedestrian crossing", "danger"),
    28: ("school crossing", "danger"),
    29: ("cycles crossing", "danger"),
    30: ("snow", "danger"),
    31: ("animals", "danger"),
    32: ("restriction ends", "other"),
    33: ("go right", "mandatory"),
    34: ("go left", "mandatory"),
    35: ("go straight", "mandatory"),
    36: ("go right or straight", "mandatory"),
    37: ("go left or straight", "mandatory"),
    38: ("keep right", "mandatory"),
    39: ("keep left", "mandatory"),
    40: ("roundabout", "mandatory"),
    41: ("restriction ends (overtaking)", "other"),
    42: ("restriction ends (overtaking (trucks))", "other"),
}
SOURCE_CLASS_NAMES = {class_id: info[0] for class_id, info in SOURCE_CLASS_INFO.items()}
SOURCE_CLASS_CATEGORIES = {
    class_id: info[1] for class_id, info in SOURCE_CLASS_INFO.items()
}
PROHIBITORY_SOURCE_CLASSES = frozenset(
    class_id
    for class_id, category in SOURCE_CLASS_CATEGORIES.items()
    if category == "prohibitory"
)
DANGER_SOURCE_CLASSES = frozenset(
    class_id
    for class_id, category in SOURCE_CLASS_CATEGORIES.items()
    if category == "danger"
)
MANDATORY_SOURCE_CLASSES = frozenset(
    class_id
    for class_id, category in SOURCE_CLASS_CATEGORIES.items()
    if category == "mandatory"
)
EXCLUDED_SOURCE_CLASSES = frozenset(
    class_id
    for class_id, category in SOURCE_CLASS_CATEGORIES.items()
    if category == "other"
)


class PreparationError(RuntimeError):
    """Raised when the source data or output cannot be prepared safely."""


class DependencyError(PreparationError):
    """Raised when a required third-party package is unavailable."""


@dataclass(frozen=True)
class Annotation:
    image: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_id: int


@dataclass
class SourceDataset:
    root: Path
    image_names: List[str]
    dimensions: Dict[str, Tuple[int, int]]
    annotations: List[Annotation]
    annotation_row_count: int
    duplicate_annotations: List[Annotation]
    by_image: Dict[str, List[Annotation]]
    unannotated: List[str]
    crop_counts: Optional[Dict[int, int]]
    gt_sha256: str
    warnings: List[str]


@dataclass
class SelectionResult:
    kept: Dict[str, List[Annotation]]
    dropped_images: List[str]
    excluded_unannotated: List[str]
    stats: Dict[str, int]


def _require_pillow() -> None:
    if Image is None:
        raise DependencyError(
            "Pillow is required to inspect PPM images and write JPG files. "
            "Install it with: python3 -m pip install Pillow"
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _format_limited(items: Sequence[str], limit: int = 10) -> str:
    visible = list(items[:limit])
    if len(items) > limit:
        visible.append(f"... and {len(items) - limit} more")
    return "\n".join(visible)


def validate_class_mapping() -> None:
    expected_ids = set(range(43))
    if set(SOURCE_CLASS_INFO) != expected_ids:
        raise PreparationError("Source class mapping must cover exactly IDs 0-42.")
    if set(SOURCE_CLASS_CATEGORIES) != expected_ids:
        raise PreparationError("Category mapping must cover exactly IDs 0-42.")
    category_sets = {
        "prohibitory": PROHIBITORY_SOURCE_CLASSES,
        "danger": DANGER_SOURCE_CLASSES,
        "mandatory": MANDATORY_SOURCE_CLASSES,
        "other": EXCLUDED_SOURCE_CLASSES,
    }
    for category, class_ids in category_sets.items():
        for class_id in class_ids:
            if SOURCE_CLASS_CATEGORIES[class_id] != category:
                raise PreparationError(
                    f"Class {class_id} appears in {category} and "
                    f"{SOURCE_CLASS_CATEGORIES[class_id]}."
                )
    union = set().union(*category_sets.values())
    if union != expected_ids:
        raise PreparationError("Category sets do not partition IDs 0-42.")


def source_class_to_target(class_id: int) -> Optional[int]:
    if class_id not in SOURCE_CLASS_CATEGORIES:
        raise ValueError(f"Unknown GTSDB source class ID: {class_id}")
    category = SOURCE_CLASS_CATEGORIES[class_id]
    if category == "other":
        return None
    return TARGET_CLASS_IDS[category]


def bbox_to_yolo(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {image_width}x{image_height}."
        )
    if not 0 <= xmin < xmax <= image_width:
        raise ValueError(
            f"Invalid horizontal bbox bounds ({xmin}, {xmax}) for width {image_width}."
        )
    if not 0 <= ymin < ymax <= image_height:
        raise ValueError(
            f"Invalid vertical bbox bounds ({ymin}, {ymax}) for height {image_height}."
        )
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise ValueError("Normalized bbox center is outside [0, 1].")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise ValueError("Normalized bbox size is outside (0, 1].")
    return (x_center, y_center, width, height)


def format_yolo_label(
    annotations: Sequence[Annotation], image_width: int, image_height: int
) -> str:
    rows: List[Tuple[int, float, float, float, float]] = []
    for annotation in annotations:
        target_id = source_class_to_target(annotation.class_id)
        if target_id is None:
            raise ValueError(
                f"Annotation for {annotation.image} uses excluded class "
                f"{annotation.class_id}."
            )
        x_center, y_center, width, height = bbox_to_yolo(
            annotation.xmin,
            annotation.ymin,
            annotation.xmax,
            annotation.ymax,
            image_width,
            image_height,
        )
        rows.append((target_id, x_center, y_center, width, height))
    rows.sort()
    if not rows:
        return ""
    return "\n".join(
        f"{target_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for target_id, x_center, y_center, width, height in rows
    ) + "\n"


def parse_gt(
    gt_path: Path,
    duplicate_sink: Optional[List[Annotation]] = None,
) -> List[Annotation]:
    path = Path(gt_path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise PreparationError(f"Cannot read ground truth file {path}: {exc}") from exc
    annotations: List[Annotation] = []
    errors: List[str] = []
    seen: Set[Tuple[str, int, int, int, int, int]] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 6:
            errors.append(
                f"line {line_number}: expected 6 semicolon-separated fields, "
                f"found {len(parts)}"
            )
            continue
        image = parts[0]
        if not re.fullmatch(r"\d{5}\.ppm", image):
            errors.append(
                f"line {line_number}: unexpected image name {image!r}; "
                "expected NNNNN.ppm"
            )
            continue
        try:
            xmin, ymin, xmax, ymax, class_id = (int(value) for value in parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: bbox and class fields must be integers")
            continue
        if not 0 <= class_id <= 42:
            errors.append(
                f"line {line_number}: class ID {class_id} is outside the official "
                "range 0-42"
            )
            continue
        key = (image, xmin, ymin, xmax, ymax, class_id)
        annotation = Annotation(
            image=image,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            class_id=class_id,
        )
        if key in seen:
            if duplicate_sink is not None:
                duplicate_sink.append(annotation)
            continue
        seen.add(key)
        annotations.append(annotation)
    if errors:
        raise PreparationError(
            f"Invalid ground truth file {path}:\n{_format_limited(errors)}"
        )
    if not annotations:
        raise PreparationError(f"Ground truth file {path} contains no annotations.")
    return annotations


def _check_class_crops(
    source_root: Path,
    annotations: Sequence[Annotation],
    duplicate_annotations: Sequence[Annotation],
    errors: List[str],
) -> Optional[Dict[int, int]]:
    class_dirs = [
        path
        for path in source_root.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{2}", path.name)
    ]
    if not class_dirs:
        return None
    expected_counts = Counter(annotation.class_id for annotation in annotations)
    expected_counts.update(
        annotation.class_id for annotation in duplicate_annotations
    )
    actual_counts: Dict[int, int] = {}
    for class_id in range(43):
        class_dir = source_root / f"{class_id:02d}"
        if not class_dir.is_dir():
            errors.append(f"Missing class crop directory {class_dir.name}/")
            continue
        actual_counts[class_id] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".ppm"
        )
    for class_id in range(43):
        expected = expected_counts.get(class_id, 0)
        actual = actual_counts.get(class_id)
        if actual is not None and actual != expected:
            errors.append(
                f"Class {class_id:02d} crop count is {actual}, but gt.txt has "
                f"{expected} annotations"
            )
    return actual_counts


def load_source(
    source_dir: Path,
    expected_image_count: int = OFFICIAL_IMAGE_COUNT,
) -> SourceDataset:
    _require_pillow()
    root = Path(source_dir).resolve()
    if not root.is_dir():
        raise PreparationError(f"Source dataset directory does not exist: {root}")
    gt_path = root / "gt.txt"
    if not gt_path.is_file():
        raise PreparationError(f"Missing ground truth file: {gt_path}")

    ppm_paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".ppm"
    )
    if len(ppm_paths) != expected_image_count:
        raise PreparationError(
            f"Expected {expected_image_count} top-level .ppm images in {root}, "
            f"found {len(ppm_paths)}."
        )
    image_names = [path.name for path in ppm_paths]
    if len(set(image_names)) != len(image_names):
        raise PreparationError("Duplicate image names found in source directory.")
    if expected_image_count == OFFICIAL_IMAGE_COUNT:
        expected_names = {
            f"{index:05d}.ppm" for index in range(OFFICIAL_IMAGE_COUNT)
        }
        if set(image_names) != expected_names:
            missing = sorted(expected_names - set(image_names))
            extra = sorted(set(image_names) - expected_names)
            details: List[str] = []
            if missing:
                details.append("missing:\n" + _format_limited(missing))
            if extra:
                details.append("unexpected:\n" + _format_limited(extra))
            raise PreparationError(
                "Top-level PPM names are not the official 00000.ppm-00899.ppm "
                f"sequence.\n{' '.join(details)}"
            )

    errors: List[str] = []
    dimensions: Dict[str, Tuple[int, int]] = {}
    for path in ppm_paths:
        try:
            with Image.open(path) as image:
                dimensions[path.name] = image.size
                if image.size != SOURCE_IMAGE_SIZE:
                    errors.append(
                        f"{path.name}: image size is {image.size[0]}x{image.size[1]}, "
                        f"expected {SOURCE_IMAGE_SIZE[0]}x{SOURCE_IMAGE_SIZE[1]}"
                    )
        except Exception as exc:
            errors.append(f"{path.name}: cannot read image ({exc})")

    duplicate_annotations: List[Annotation] = []
    annotations = parse_gt(gt_path, duplicate_sink=duplicate_annotations)
    annotation_row_count = len(annotations) + len(duplicate_annotations)
    by_image: Dict[str, List[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.image not in dimensions:
            errors.append(
                f"{annotation.image}: annotation references a missing source image"
            )
            continue
        width, height = dimensions[annotation.image]
        if not (
            0 <= annotation.xmin < annotation.xmax <= width
            and 0 <= annotation.ymin < annotation.ymax <= height
        ):
            errors.append(
                f"{annotation.image}: bbox "
                f"({annotation.xmin},{annotation.ymin},"
                f"{annotation.xmax},{annotation.ymax}) is outside "
                f"{width}x{height} or has zero area"
            )
            continue
        by_image[annotation.image].append(annotation)

    crop_counts = _check_class_crops(
        root, annotations, duplicate_annotations, errors
    )
    if errors:
        raise PreparationError(
            f"Source dataset validation failed for {root}:\n"
            f"{_format_limited(errors, 20)}"
        )

    unannotated = [name for name in image_names if name not in by_image]
    warnings: List[str] = []
    if duplicate_annotations:
        details = ", ".join(
            f"{annotation.image} class {annotation.class_id} "
            f"({annotation.xmin},{annotation.ymin},"
            f"{annotation.xmax},{annotation.ymax})"
            for annotation in duplicate_annotations[:5]
        )
        warnings.append(
            f"Skipped {len(duplicate_annotations)} exact duplicate annotation(s): "
            f"{details}"
        )
    if expected_image_count == OFFICIAL_IMAGE_COUNT:
        if annotation_row_count != OFFICIAL_GT_ROWS:
            warnings.append(
                f"gt.txt has {annotation_row_count} rows; the official release "
                f"has {OFFICIAL_GT_ROWS}."
            )
        if len(annotations) != OFFICIAL_UNIQUE_GT_ROWS:
            warnings.append(
                f"gt.txt has {len(annotations)} unique annotations; the official "
                f"release has {OFFICIAL_UNIQUE_GT_ROWS}."
            )
        if len(by_image) != OFFICIAL_ANNOTATED_IMAGE_COUNT:
            warnings.append(
                f"{len(by_image)} images are annotated; the official release has "
                f"{OFFICIAL_ANNOTATED_IMAGE_COUNT}."
            )
        if len(unannotated) != OFFICIAL_UNANNOTATED_IMAGE_COUNT:
            warnings.append(
                f"{len(unannotated)} images have no rows in gt.txt; the official "
                f"release has {OFFICIAL_UNANNOTATED_IMAGE_COUNT}."
            )
    if crop_counts is None:
        warnings.append(
            "Class crop directories were not found; the gt.txt cross-check was "
            "skipped."
        )

    gt_sha256 = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    return SourceDataset(
        root=root,
        image_names=image_names,
        dimensions=dimensions,
        annotations=annotations,
        annotation_row_count=annotation_row_count,
        duplicate_annotations=duplicate_annotations,
        by_image=dict(by_image),
        unannotated=unannotated,
        crop_counts=crop_counts,
        gt_sha256=gt_sha256,
        warnings=warnings,
    )


def select_images(
    image_names: Sequence[str],
    by_image: Mapping[str, Sequence[Annotation]],
    include_unannotated: bool,
) -> SelectionResult:
    kept: Dict[str, List[Annotation]] = {}
    dropped_images: List[str] = []
    excluded_unannotated: List[str] = []
    dropped_annotation_count = 0
    dropped_target_annotation_count = 0
    dropped_other_annotation_count = 0
    dropped_other_only_image_count = 0
    dropped_mixed_image_count = 0

    for image_name in sorted(image_names):
        annotations = list(by_image.get(image_name, ()))
        other_annotations = [
            annotation
            for annotation in annotations
            if SOURCE_CLASS_CATEGORIES[annotation.class_id] == "other"
        ]
        target_annotations = [
            annotation
            for annotation in annotations
            if SOURCE_CLASS_CATEGORIES[annotation.class_id] != "other"
        ]
        if other_annotations:
            dropped_images.append(image_name)
            dropped_annotation_count += len(annotations)
            dropped_target_annotation_count += len(target_annotations)
            dropped_other_annotation_count += len(other_annotations)
            if target_annotations:
                dropped_mixed_image_count += 1
            else:
                dropped_other_only_image_count += 1
            continue
        if not annotations and not include_unannotated:
            excluded_unannotated.append(image_name)
            continue
        kept[image_name] = sorted(
            target_annotations,
            key=lambda annotation: (
                source_class_to_target(annotation.class_id),
                annotation.xmin,
                annotation.ymin,
                annotation.xmax,
                annotation.ymax,
            ),
        )

    kept_annotation_count = sum(len(values) for values in kept.values())
    kept_negative_image_count = sum(1 for values in kept.values() if not values)
    stats = {
        "source_image_count": len(image_names),
        "kept_image_count": len(kept),
        "kept_annotation_count": kept_annotation_count,
        "kept_negative_image_count": kept_negative_image_count,
        "dropped_image_count": len(dropped_images),
        "dropped_annotation_count": dropped_annotation_count,
        "dropped_target_annotation_count": dropped_target_annotation_count,
        "dropped_other_annotation_count": dropped_other_annotation_count,
        "dropped_other_only_image_count": dropped_other_only_image_count,
        "dropped_mixed_image_count": dropped_mixed_image_count,
        "excluded_unannotated_count": len(excluded_unannotated),
    }
    return SelectionResult(
        kept=kept,
        dropped_images=dropped_images,
        excluded_unannotated=excluded_unannotated,
        stats=stats,
    )


def compute_split_counts(total: int, ratios: Sequence[float]) -> List[int]:
    if len(ratios) < 2:
        raise ValueError("At least two split ratios are required.")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be greater than zero.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")
    if total < len(ratios):
        raise ValueError(
            f"At least {len(ratios)} images are required for three non-empty splits."
        )
    raw = [total * ratio for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            donor = max(range(len(counts)), key=lambda candidate: counts[candidate])
            if counts[donor] <= 1:
                raise ValueError("Cannot create three non-empty splits.")
            counts[donor] -= 1
            counts[index] += 1
    if sum(counts) != total:
        raise ValueError("Internal error while computing split counts.")
    return counts


def split_images(
    image_names: Sequence[str],
    seed: int,
    ratios: Sequence[float],
    mode: str = "random",
) -> Dict[str, List[str]]:
    names = sorted(image_names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate image names cannot be split.")
    if mode == "random":
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        counts = compute_split_counts(len(shuffled), ratios)
        train_end = counts[0]
        val_end = train_end + counts[1]
        splits = {
            "train": sorted(shuffled[:train_end]),
            "val": sorted(shuffled[train_end:val_end]),
            "test": sorted(shuffled[val_end:]),
        }
    elif mode == "official":
        test_names: List[str] = []
        remaining: List[str] = []
        for name in names:
            number = int(Path(name).stem)
            if OFFICIAL_TEST_START <= number <= OFFICIAL_TEST_END:
                test_names.append(name)
            else:
                remaining.append(name)
        if not test_names:
            raise ValueError(
                "Official split mode requires images 00600.ppm-00899.ppm."
            )
        random.Random(seed).shuffle(remaining)
        relative_ratios = [
            ratios[0] / (ratios[0] + ratios[1]),
            ratios[1] / (ratios[0] + ratios[1]),
        ]
        counts = compute_split_counts(len(remaining), relative_ratios)
        splits = {
            "train": sorted(remaining[: counts[0]]),
            "val": sorted(remaining[counts[0] :]),
            "test": sorted(test_names),
        }
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    assigned = splits["train"] + splits["val"] + splits["test"]
    if len(assigned) != len(names) or set(assigned) != set(names):
        raise ValueError("Split assignment did not partition the input images.")
    if len(assigned) != len(set(assigned)):
        raise ValueError("Image leakage detected: an image was assigned twice.")
    return splits


def _zip_root_and_images(
    archive: Path, expected_image_count: int
) -> Tuple[str, List[str]]:
    try:
        with zipfile.ZipFile(archive) as zip_file:
            names = zip_file.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f"Cannot read ZIP archive {archive}: {exc}") from exc
    if len(names) != len(set(names)):
        raise PreparationError(f"ZIP archive {archive} contains duplicate entries.")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise PreparationError(
                f"Unsafe path in ZIP archive {archive}: {name!r}"
            )
    gt_members = [
        name
        for name in names
        if PurePosixPath(name).name == "gt.txt" and not name.endswith("/")
    ]
    if len(gt_members) != 1:
        raise PreparationError(
            f"Expected exactly one gt.txt in {archive}, found {len(gt_members)}."
        )
    gt_member = gt_members[0]
    root = PurePosixPath(gt_member).parent
    ppm_members = [
        name
        for name in names
        if PurePosixPath(name).parent == root
        and PurePosixPath(name).suffix.lower() == ".ppm"
        and not name.endswith("/")
    ]
    if len(ppm_members) != expected_image_count:
        raise PreparationError(
            f"Expected {expected_image_count} top-level PPM entries in {archive}, "
            f"found {len(ppm_members)}."
        )
    expected_names = {
        f"{index:05d}.ppm" for index in range(expected_image_count)
    }
    actual_names = {PurePosixPath(name).name for name in ppm_members}
    if actual_names != expected_names:
        raise PreparationError(
            f"ZIP archive {archive} does not contain the official "
            f"00000.ppm-{expected_image_count - 1:05d}.ppm image sequence."
        )
    root_name = "" if str(root) == "." else str(root)
    return root_name, ppm_members


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename:
                raise PreparationError(
                    f"Unsafe ZIP member path: {info.filename!r}"
                )
            target = destination.joinpath(*relative.parts)
            if info.is_dir() or info.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            try:
                with zip_file.open(info) as source, open(temporary, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()


def _manual_download_instructions(data_dir: Path) -> str:
    return (
        "Raw GTSDB data is unavailable.\n"
        f"Download {OFFICIAL_ARCHIVE_NAME} from the verified official archive:\n"
        f"  {OFFICIAL_ARCHIVE_URL}\n"
        f"Expected size: {OFFICIAL_ARCHIVE_SIZE} bytes.\n"
        f"Place the ZIP in {data_dir} and rerun, or extract it so that "
        f"{data_dir / 'FullIJCNN2013' / 'gt.txt'} exists.\n"
        "Alternatively rerun with --download to fetch the archive automatically."
    )


def _extract_archive(
    archive: Path,
    data_dir: Path,
    expected_image_count: int,
) -> Path:
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise PreparationError(f"ZIP archive does not exist: {archive}")
    root_name, _ = _zip_root_and_images(archive, expected_image_count)
    target = data_dir / "FullIJCNN2013"
    if target.is_dir():
        try:
            load_source(target, expected_image_count)
            return target.resolve()
        except PreparationError as exc:
            raise PreparationError(
                f"Extracted directory exists but is invalid and will not be "
                f"overwritten: {target}\n{exc}"
            ) from exc
    stage = Path(tempfile.mkdtemp(prefix=".gtsdb-extract-", dir=str(data_dir)))
    try:
        _safe_extract(archive, stage)
        extracted = stage / root_name if root_name else stage
        load_source(extracted, expected_image_count)
        if target.exists():
            raise PreparationError(
                f"Refusing to overwrite existing path: {target}"
            )
        os.replace(str(extracted), str(target))
        return target.resolve()
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def download_official_archive(destination: Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        size = destination.stat().st_size
        if size == OFFICIAL_ARCHIVE_SIZE:
            return destination
        raise PreparationError(
            f"Existing archive {destination} has size {size}, expected "
            f"{OFFICIAL_ARCHIVE_SIZE}. It will not be overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        OFFICIAL_ARCHIVE_URL,
        headers={"User-Agent": "traffic-sign-edge-ai-gtsdb-preparer/1"},
    )
    print(f"Downloading {OFFICIAL_ARCHIVE_URL}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if (
                content_length is not None
                and int(content_length) != OFFICIAL_ARCHIVE_SIZE
            ):
                raise PreparationError(
                    f"Server reported {content_length} bytes, expected "
                    f"{OFFICIAL_ARCHIVE_SIZE}."
                )
            with open(partial, "wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        size = partial.stat().st_size
        if size != OFFICIAL_ARCHIVE_SIZE:
            raise PreparationError(
                f"Downloaded archive has size {size}, expected "
                f"{OFFICIAL_ARCHIVE_SIZE}."
            )
        os.replace(partial, destination)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise PreparationError(
            f"Download failed: {exc}\n"
            f"{_manual_download_instructions(destination.parent)}"
        ) from exc
    finally:
        if partial.exists():
            partial.unlink()
    print(f"Saved {destination} ({OFFICIAL_ARCHIVE_SIZE} bytes)")
    return destination


def acquire_source(
    data_dir: Path,
    source_arg: Optional[str] = None,
    zip_arg: Optional[str] = None,
    download: bool = False,
    expected_image_count: int = OFFICIAL_IMAGE_COUNT,
) -> Path:
    _require_pillow()
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if source_arg:
        candidate = Path(source_arg)
        if candidate.is_dir():
            load_source(candidate, expected_image_count)
            return candidate.resolve()
        if candidate.is_file() and zipfile.is_zipfile(candidate):
            return _extract_archive(candidate, data_dir, expected_image_count)
        raise PreparationError(
            f"--source must be an extracted dataset directory or a ZIP archive: "
            f"{candidate}"
        )

    if zip_arg:
        candidate = Path(zip_arg)
        if not candidate.is_file():
            raise PreparationError(f"ZIP archive does not exist: {candidate}")
        return _extract_archive(candidate, data_dir, expected_image_count)

    extracted = data_dir / "FullIJCNN2013"
    extracted_error: Optional[PreparationError] = None
    if extracted.is_dir():
        try:
            load_source(extracted, expected_image_count)
            return extracted.resolve()
        except PreparationError as exc:
            extracted_error = exc

    candidate_zips = [
        data_dir / "FullIJCNN2013.zip",
        data_dir / "GTSDB_FullIJCNN2013.zip",
    ]
    for candidate in candidate_zips:
        if candidate.is_file():
            return _extract_archive(candidate, data_dir, expected_image_count)

    if download:
        archive = data_dir / OFFICIAL_ARCHIVE_NAME
        download_official_archive(archive)
        return _extract_archive(archive, data_dir, expected_image_count)

    message = _manual_download_instructions(data_dir)
    if extracted_error is not None:
        message += f"\nExisting extracted directory is invalid:\n{extracted_error}"
    raise PreparationError(message)


def _load_manifest(output_dir: Path) -> Dict[str, object]:
    path = output_dir / ".prepare_manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _manifest_entry_matches(
    entry: object,
    source_name: str,
    source_size: int,
    source_mtime_ns: int,
    jpeg_quality: int,
) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        entry.get("source") == source_name
        and entry.get("source_size") == source_size
        and entry.get("source_mtime_ns") == source_mtime_ns
        and entry.get("jpeg_quality") == jpeg_quality
    )


def _convert_ppm_to_jpg(
    source_path: Path,
    output_path: Path,
    jpeg_quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}-",
        suffix=".jpg.tmp",
        dir=str(output_path.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with Image.open(source_path) as image:
            if image.size != SOURCE_IMAGE_SIZE:
                raise PreparationError(
                    f"{source_path.name}: size {image.size} does not match "
                    f"{SOURCE_IMAGE_SIZE}"
                )
            image.convert("RGB").save(
                temporary,
                format="JPEG",
                quality=jpeg_quality,
                subsampling=0,
                optimize=False,
            )
        with Image.open(temporary) as check:
            if check.size != SOURCE_IMAGE_SIZE:
                raise PreparationError(
                    f"Converted {source_path.name} has unexpected size "
                    f"{check.size}"
                )
        os.replace(temporary, output_path)
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(
            f"Failed to convert {source_path} to {output_path}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_if_changed(path: Path, content: str) -> bool:
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _write_json_if_changed(path: Path, payload: Mapping[str, object]) -> bool:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text_if_changed(path, content)


def _remove_stale_outputs(
    output_dir: Path,
    previous_manifest: Mapping[str, object],
    expected_outputs: Set[str],
) -> int:
    previous_outputs: Set[str] = set()
    raw_outputs = previous_manifest.get("outputs", [])
    if isinstance(raw_outputs, list):
        previous_outputs.update(
            value for value in raw_outputs if isinstance(value, str)
        )
    raw_images = previous_manifest.get("images", {})
    if isinstance(raw_images, dict):
        previous_outputs.update(
            value for value in raw_images.keys() if isinstance(value, str)
        )
    removed = 0
    for relative_name in sorted(previous_outputs - expected_outputs):
        candidate = (output_dir / relative_name).resolve()
        if not _is_relative_to(candidate, output_dir):
            continue
        if candidate.is_file():
            candidate.unlink()
            removed += 1
    return removed


def _write_dataset_yaml(output_dir: Path) -> Path:
    path = output_dir / "dataset.yaml"
    lines = [
        f"path: {json.dumps(str(output_dir))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(TARGET_CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {name}"
        for class_id, name in enumerate(TARGET_CLASS_NAMES)
    )
    _write_text_if_changed(path, "\n".join(lines) + "\n")
    return path


def _write_dataset_outputs(
    source: SourceDataset,
    selection: SelectionResult,
    splits: Mapping[str, Sequence[str]],
    output_dir: Path,
    jpeg_quality: int,
    previous_manifest: Mapping[str, object],
    force_reconvert: bool,
) -> Tuple[Dict[str, int], Set[str], Dict[str, Dict[str, object]]]:
    converted = 0
    reused = 0
    labels_written = 0
    labels_unchanged = 0
    expected_outputs: Set[str] = set()
    image_entries: Dict[str, Dict[str, object]] = {}
    previous_images = previous_manifest.get("images", {})
    if not isinstance(previous_images, dict):
        previous_images = {}

    for split in SPLIT_NAMES:
        for image_name in splits[split]:
            source_path = source.root / image_name
            stem = Path(image_name).stem
            relative_image = Path("images") / split / f"{stem}.jpg"
            relative_label = Path("labels") / split / f"{stem}.txt"
            output_image = output_dir / relative_image
            output_label = output_dir / relative_label
            source_stat = source_path.stat()
            previous_entry = previous_images.get(relative_image.as_posix())
            can_reuse = (
                not force_reconvert
                and output_image.is_file()
                and output_image.stat().st_size > 0
                and _manifest_entry_matches(
                    previous_entry,
                    image_name,
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                    jpeg_quality,
                )
            )
            if can_reuse:
                reused += 1
            else:
                _convert_ppm_to_jpg(source_path, output_image, jpeg_quality)
                converted += 1
            label_content = format_yolo_label(
                selection.kept[image_name], *source.dimensions[image_name]
            )
            if _write_text_if_changed(output_label, label_content):
                labels_written += 1
            else:
                labels_unchanged += 1
            expected_outputs.add(relative_image.as_posix())
            expected_outputs.add(relative_label.as_posix())
            image_entries[relative_image.as_posix()] = {
                "source": image_name,
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "jpeg_quality": jpeg_quality,
            }

    removed_stale = _remove_stale_outputs(
        output_dir, previous_manifest, expected_outputs
    )
    stats = {
        "converted_images": converted,
        "reused_images": reused,
        "labels_written": labels_written,
        "labels_unchanged": labels_unchanged,
        "removed_stale_files": removed_stale,
    }
    return stats, expected_outputs, image_entries


def _update_range(
    ranges: Dict[str, List[Optional[float]]], values: Sequence[float]
) -> None:
    for key, value in zip(("x_center", "y_center", "width", "height"), values):
        current = ranges[key]
        if current[0] is None or value < current[0]:
            current[0] = value
        if current[1] is None or value > current[1]:
            current[1] = value


def validate_outputs(
    source: SourceDataset,
    selection: SelectionResult,
    splits: Mapping[str, Sequence[str]],
    output_dir: Path,
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = list(source.warnings)
    split_reports: Dict[str, Dict[str, object]] = {}
    class_totals = Counter()
    coordinate_ranges: Dict[str, List[Optional[float]]] = {
        "x_center": [None, None],
        "y_center": [None, None],
        "width": [None, None],
        "height": [None, None],
    }
    total_boxes = 0
    missing_images = 0
    extra_images = 0
    missing_labels = 0
    extra_labels = 0
    invalid_boxes = 0
    split_overlap = 0
    image_to_split: Dict[str, str] = {}

    for split in SPLIT_NAMES:
        expected_stems = {Path(name).stem for name in splits[split]}
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        actual_image_stems = (
            {path.stem for path in image_dir.glob("*.jpg")}
            if image_dir.is_dir()
            else set()
        )
        actual_label_stems = (
            {path.stem for path in label_dir.glob("*.txt")}
            if label_dir.is_dir()
            else set()
        )
        missing_image_stems = expected_stems - actual_image_stems
        extra_image_stems = actual_image_stems - expected_stems
        missing_label_stems = expected_stems - actual_label_stems
        extra_label_stems = actual_label_stems - expected_stems
        missing_images += len(missing_image_stems)
        extra_images += len(extra_image_stems)
        missing_labels += len(missing_label_stems)
        extra_labels += len(extra_label_stems)
        if missing_image_stems:
            errors.append(
                f"{split}: missing images "
                f"{_format_limited(sorted(missing_image_stems))}"
            )
        if extra_image_stems:
            errors.append(
                f"{split}: unexpected images "
                f"{_format_limited(sorted(extra_image_stems))}"
            )
        if missing_label_stems:
            errors.append(
                f"{split}: missing labels "
                f"{_format_limited(sorted(missing_label_stems))}"
            )
        if extra_label_stems:
            errors.append(
                f"{split}: unexpected labels "
                f"{_format_limited(sorted(extra_label_stems))}"
            )

        class_counts = Counter()
        split_boxes = 0
        split_negatives = 0
        for image_name in splits[split]:
            stem = Path(image_name).stem
            if stem in image_to_split:
                split_overlap += 1
                errors.append(
                    f"{stem}: present in both {image_to_split[stem]} and {split}"
                )
            image_to_split[stem] = split
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            if image_path.is_file():
                try:
                    with Image.open(image_path) as image:
                        if image.size != source.dimensions[image_name]:
                            errors.append(
                                f"{image_path}: size {image.size} does not match "
                                f"source {source.dimensions[image_name]}"
                            )
                except Exception as exc:
                    errors.append(f"{image_path}: cannot read ({exc})")
            if not label_path.is_file():
                continue
            try:
                content = label_path.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"{label_path}: cannot read ({exc})")
                continue
            expected_content = format_yolo_label(
                selection.kept[image_name], *source.dimensions[image_name]
            )
            if content != expected_content:
                errors.append(f"{label_path}: content differs from expected conversion")
            lines = [line for line in content.splitlines() if line.strip()]
            if not lines:
                split_negatives += 1
                if selection.kept[image_name]:
                    errors.append(
                        f"{label_path}: empty label for an annotated image"
                    )
            elif not selection.kept[image_name]:
                errors.append(
                    f"{label_path}: non-empty label for an unannotated image"
                )
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_boxes += 1
                    errors.append(
                        f"{label_path}:{line_number}: expected 5 fields, "
                        f"found {len(parts)}"
                    )
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_boxes += 1
                    errors.append(
                        f"{label_path}:{line_number}: non-numeric YOLO fields"
                    )
                    continue
                if class_id not in range(len(TARGET_CLASS_NAMES)):
                    invalid_boxes += 1
                    errors.append(
                        f"{label_path}:{line_number}: class ID {class_id} is not "
                        f"in 0-{len(TARGET_CLASS_NAMES) - 1}"
                    )
                    continue
                x_center, y_center, width, height = values
                left = x_center - width / 2.0
                right = x_center + width / 2.0
                top = y_center - height / 2.0
                bottom = y_center + height / 2.0
                values_valid = (
                    0.0 <= x_center <= 1.0
                    and 0.0 <= y_center <= 1.0
                    and 0.0 < width <= 1.0
                    and 0.0 < height <= 1.0
                    and left >= -1e-6
                    and right <= 1.0 + 1e-6
                    and top >= -1e-6
                    and bottom <= 1.0 + 1e-6
                )
                if not values_valid:
                    invalid_boxes += 1
                    errors.append(
                        f"{label_path}:{line_number}: normalized bbox is outside "
                        f"[0,1]: {line}"
                    )
                    continue
                class_counts[class_id] += 1
                split_boxes += 1
                _update_range(
                    coordinate_ranges,
                    (x_center, y_center, width, height),
                )
        total_boxes += split_boxes
        for class_id, name in enumerate(TARGET_CLASS_NAMES):
            class_totals[name] += class_counts.get(class_id, 0)
        class_report = {
            name: class_counts.get(class_id, 0)
            for class_id, name in enumerate(TARGET_CLASS_NAMES)
        }
        for name, count in class_report.items():
            if count == 0:
                warnings.append(f"{split}: target class {name!r} has no boxes")
        split_reports[split] = {
            "images": len(expected_stems),
            "labels": len(actual_label_stems),
            "boxes": split_boxes,
            "negatives": split_negatives,
            "classes": class_report,
        }

    if total_boxes != selection.stats["kept_annotation_count"]:
        errors.append(
            f"Output contains {total_boxes} boxes, expected "
            f"{selection.stats['kept_annotation_count']}."
        )
    for image_name in selection.dropped_images:
        stem = Path(image_name).stem
        if stem in image_to_split:
            errors.append(
                f"{stem}: image with excluded-class annotations was written to "
                f"{image_to_split[stem]}"
            )
    accounted = (
        set(selection.kept)
        | set(selection.dropped_images)
        | set(selection.excluded_unannotated)
    )
    if accounted != set(source.image_names):
        errors.append(
            "Selected, dropped, and excluded images do not partition the source "
            "images."
        )
    if len(accounted) != (
        len(selection.kept)
        + len(selection.dropped_images)
        + len(selection.excluded_unannotated)
    ):
        errors.append("An image was classified into more than one selection group.")

    checks = {
        "missing_images": missing_images,
        "extra_images": extra_images,
        "missing_labels": missing_labels,
        "extra_labels": extra_labels,
        "invalid_boxes": invalid_boxes,
        "split_overlap": split_overlap,
        "total_boxes": total_boxes,
        "coordinate_ranges": coordinate_ranges,
        "class_totals": dict(class_totals),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "splits": split_reports,
        "checks": checks,
    }


def prepare_dataset(
    source_root: Path,
    output_dir: Path,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    jpeg_quality: int = 95,
    include_unannotated: bool = True,
    split_mode: str = "random",
    force_reconvert: bool = False,
) -> Dict[str, object]:
    validate_class_mapping()
    if not 1 <= jpeg_quality <= 100:
        raise PreparationError("JPEG quality must be between 1 and 100.")
    if len(ratios) != 3:
        raise PreparationError("Exactly three split ratios are required.")
    if any(ratio <= 0 for ratio in ratios):
        raise PreparationError("All split ratios must be greater than zero.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise PreparationError(f"Split ratios must sum to 1.0, got {sum(ratios)}.")

    source = load_source(source_root)
    output_dir = Path(output_dir).resolve()
    if output_dir == source.root or _is_relative_to(output_dir, source.root):
        raise PreparationError(
            f"Output directory must not be inside the raw source directory: "
            f"{output_dir}"
        )

    selection = select_images(
        source.image_names,
        source.by_image,
        include_unannotated=include_unannotated,
    )
    if not selection.kept:
        raise PreparationError("No images remain after applying the filter policy.")

    try:
        splits = split_images(
            sorted(selection.kept), seed, ratios, mode=split_mode
        )
    except ValueError as exc:
        raise PreparationError(f"Cannot create train/val/test split: {exc}") from exc
    for split in SPLIT_NAMES:
        if not splits[split]:
            raise PreparationError(
                f"Split {split!r} is empty after excluding filtered images. "
                "Adjust the seed/ratios or use random split mode."
            )
    assigned = splits["train"] + splits["val"] + splits["test"]
    if set(assigned) != set(selection.kept) or len(assigned) != len(selection.kept):
        raise PreparationError("Split assignment does not match selected images.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    previous_manifest = _load_manifest(output_dir)
    write_stats, expected_outputs, image_entries = _write_dataset_outputs(
        source,
        selection,
        splits,
        output_dir,
        jpeg_quality,
        previous_manifest,
        force_reconvert,
    )
    dataset_yaml = _write_dataset_yaml(output_dir)
    validation = validate_outputs(source, selection, splits, output_dir)

    class_distribution: Dict[str, Dict[str, int]] = {}
    class_totals = Counter()
    for split in SPLIT_NAMES:
        counts = Counter()
        for image_name in splits[split]:
            for annotation in selection.kept[image_name]:
                target_id = source_class_to_target(annotation.class_id)
                if target_id is not None:
                    counts[target_id] += 1
        named_counts = {
            name: counts.get(class_id, 0)
            for class_id, name in enumerate(TARGET_CLASS_NAMES)
        }
        class_distribution[split] = named_counts
        for name, count in named_counts.items():
            class_totals[name] += count

    report_path = output_dir / "preparation_report.json"
    report: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "ok": bool(validation["ok"]),
        "source": {
            "root": str(source.root),
            "image_count": len(source.image_names),
            "image_width": SOURCE_IMAGE_SIZE[0],
            "image_height": SOURCE_IMAGE_SIZE[1],
            "annotation_count": len(source.annotations),
            "annotation_row_count": source.annotation_row_count,
            "duplicate_annotation_count": len(source.duplicate_annotations),
            "annotated_image_count": len(source.by_image),
            "unannotated_image_count": len(source.unannotated),
            "unannotated_images": list(source.unannotated),
            "gt_sha256": source.gt_sha256,
            "crop_counts": (
                {str(key): value for key, value in source.crop_counts.items()}
                if source.crop_counts is not None
                else None
            ),
            "warnings": list(source.warnings),
        },
        "mapping": {
            "target_classes": {
                str(class_id): name
                for class_id, name in enumerate(TARGET_CLASS_NAMES)
            },
            "prohibitory_source_classes": sorted(PROHIBITORY_SOURCE_CLASSES),
            "danger_source_classes": sorted(DANGER_SOURCE_CLASSES),
            "mandatory_source_classes": sorted(MANDATORY_SOURCE_CLASSES),
            "excluded_source_classes": sorted(EXCLUDED_SOURCE_CLASSES),
            "excluded_policy": "drop entire image if any excluded-class annotation is present",
        },
        "selection": {
            **selection.stats,
            "dropped_images": list(selection.dropped_images),
            "excluded_unannotated_images": list(selection.excluded_unannotated),
        },
        "split": {
            "mode": split_mode,
            "seed": seed,
            "ratios": {
                "train": ratios[0],
                "val": ratios[1],
                "test": ratios[2],
            },
            "image_counts": {
                split: len(splits[split]) for split in SPLIT_NAMES
            },
            "class_distribution": class_distribution,
            "class_totals": dict(class_totals),
            "official_reference": {
                "train_images": OFFICIAL_TRAIN_IMAGE_COUNT,
                "test_images": OFFICIAL_TEST_IMAGE_COUNT,
                "test_source_numbers": [
                    OFFICIAL_TEST_START,
                    OFFICIAL_TEST_END,
                ],
                "note": (
                    "Official test membership was verified by SHA-256 checks at "
                    "sampled positions 0, 1, 150, and 299 against "
                    "TestIJCNN2013.zip. Random mode is a custom split and does "
                    "not claim official benchmark comparability."
                ),
            },
        },
        "outputs": {
            **write_stats,
            "image_count": sum(len(splits[split]) for split in SPLIT_NAMES),
            "label_count": sum(len(splits[split]) for split in SPLIT_NAMES),
            "dataset_yaml": str(dataset_yaml),
            "report": str(report_path),
        },
        "validation": validation,
    }

    if validation["ok"]:
        manifest: Dict[str, object] = {
            "pipeline_version": PIPELINE_VERSION,
            "source": str(source.root),
            "source_gt_sha256": source.gt_sha256,
            "jpeg_quality": jpeg_quality,
            "seed": seed,
            "split_mode": split_mode,
            "ratios": {
                "train": ratios[0],
                "val": ratios[1],
                "test": ratios[2],
            },
            "include_unannotated": include_unannotated,
            "splits": {split: list(splits[split]) for split in SPLIT_NAMES},
            "images": image_entries,
            "outputs": sorted(expected_outputs),
        }
        _write_json_if_changed(output_dir / ".prepare_manifest.json", manifest)

    _write_json_if_changed(report_path, report)
    return report


def format_report(report: Mapping[str, object]) -> str:
    source = report["source"]
    selection = report["selection"]
    split = report["split"]
    outputs = report["outputs"]
    validation = report["validation"]
    split_counts = split["image_counts"]
    class_totals = split["class_totals"]
    dataset_yaml = outputs["dataset_yaml"]
    report_path = outputs["report"]

    lines = ["GTSDB -> YOLO dataset preparation"]
    lines.append(f"source: {source['root']}")
    lines.append(
        f"source images: {source['image_count']} "
        f"({source['image_width']}x{source['image_height']}), "
        f"annotations: {source['annotation_count']} unique "
        f"({source['annotation_row_count']} rows), "
        f"unannotated: {source['unannotated_image_count']}"
    )
    lines.append(
        "target classes: "
        + ", ".join(
            f"{class_id}={name} ({class_totals.get(name, 0)})"
            for class_id, name in enumerate(TARGET_CLASS_NAMES)
        )
    )
    lines.append(
        "excluded source classes: "
        + ", ".join(str(value) for value in report["mapping"]["excluded_source_classes"])
        + " (policy: drop entire image)"
    )
    lines.append(
        f"dropped images: {selection['dropped_image_count']} "
        f"({selection['dropped_annotation_count']} annotations, "
        f"{selection['dropped_target_annotation_count']} target annotations)"
    )
    if selection["kept_negative_image_count"]:
        lines.append(
            f"unannotated images included as negatives: "
            f"{selection['kept_negative_image_count']} (empty label files)"
        )
    if selection["excluded_unannotated_count"]:
        lines.append(
            f"unannotated images excluded: {selection['excluded_unannotated_count']}"
        )
    lines.append(
        f"split: mode={split['mode']} seed={split['seed']} "
        + " ".join(
            f"{name}={split_counts[name]}"
            for name in SPLIT_NAMES
        )
    )
    lines.append(
        "split classes: "
        + " | ".join(
            f"{name}: "
            + ", ".join(
                f"{class_name}={count}"
                for class_name, count in split["class_distribution"][name].items()
            )
            for name in SPLIT_NAMES
        )
    )
    lines.append(
        f"outputs: {outputs['image_count']} images, {outputs['label_count']} labels; "
        f"converted={outputs['converted_images']}, "
        f"reused={outputs['reused_images']}, "
        f"stale_removed={outputs['removed_stale_files']}"
    )
    checks = validation["checks"]
    lines.append(
        f"validation: {'OK' if validation['ok'] else 'FAILED'} "
        f"(missing_images={checks['missing_images']}, "
        f"missing_labels={checks['missing_labels']}, "
        f"invalid_boxes={checks['invalid_boxes']}, "
        f"split_overlap={checks['split_overlap']}, "
        f"boxes={checks['total_boxes']})"
    )
    for warning in validation["warnings"]:
        lines.append(f"warning: {warning}")
    for error in validation["errors"]:
        lines.append(f"error: {error}")
    lines.append(f"dataset.yaml: {dataset_yaml}")
    lines.append(f"report: {report_path}")
    return "\n".join(lines)


def _quality_argument(value: str) -> int:
    try:
        quality = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("JPEG quality must be an integer") from exc
    if not 1 <= quality <= 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100")
    return quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official GTSDB FullIJCNN2013 dataset for YOLO training. "
            "Images containing officially excluded ('other') classes are dropped "
            "entirely so they are never silently treated as background."
        )
    )
    parser.add_argument(
        "--source",
        help="Extracted FullIJCNN2013 directory or ZIP archive.",
    )
    parser.add_argument(
        "--zip",
        help="Local FullIJCNN2013 ZIP archive; extracted into the data directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the verified official FullIJCNN2013.zip if no local source "
            "or archive is available."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "gtsdb"),
        help="Output YOLO dataset directory (default: data/gtsdb).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Training ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation ratio."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test ratio."
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "official"),
        default="random",
        help=(
            "random: seeded split over all selected images. official: keep the "
            "verified official test images 00600-00899 and split the rest into "
            "train/val with the seed."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=_quality_argument,
        default=95,
        help="JPEG quality, 1-100 (default: 95).",
    )
    parser.add_argument(
        "--exclude-unannotated",
        action="store_true",
        help="Exclude images that have no rows in gt.txt instead of using them as negatives.",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Reconvert all JPGs even when the manifest says they are current.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    try:
        source_root = acquire_source(
            REPO_ROOT / "data",
            source_arg=args.source,
            zip_arg=args.zip,
            download=args.download,
        )
        report = prepare_dataset(
            source_root=source_root,
            output_dir=Path(args.output),
            seed=args.seed,
            ratios=ratios,
            jpeg_quality=args.jpeg_quality,
            include_unannotated=not args.exclude_unannotated,
            split_mode=args.split_mode,
            force_reconvert=args.force_reconvert,
        )
    except DependencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
