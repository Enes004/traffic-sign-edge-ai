#!/usr/bin/env python3
"""Stratified FP32 vs INT8 recall analysis by ground-truth object size.

The script is read-only for the existing benchmark, model, and dataset files.
It reuses benchmark_pi_v3.preprocess and benchmark_pi_v3.decode_prediction so
that letterbox preprocessing and class-aware NMS match the Raspberry Pi
benchmark exactly.

Ground-truth objects are split into small/medium/large groups by tertiles of
their original-pixel bounding-box area. A prediction matches a ground-truth
object only when it has the same class, confidence >= 0.25, IoU >= 0.50, and
has not already been assigned to another ground-truth object. Matching is
confidence-first and greedy, so each prediction is used at most once.

Outputs: summary CSV, JSON metadata/results, and a PNG chart.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import benchmark_pi_v3 as bench

CLASS_NAMES = bench.CLASS_NAMES
SIZE_GROUPS = ("small", "medium", "large")
GROUP_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_CONF = 0.25
DEFAULT_MATCH_IOU = 0.50
DEFAULT_NMS_IOU = 0.70
DEFAULT_MAX_DET = 300
SMALL_SAMPLE_THRESHOLD = 30

RECALL_COLUMNS = (
    "seed",
    "size_group",
    "gt_count",
    "area_min_px2",
    "area_max_px2",
    "small_max_area_px2",
    "medium_max_area_px2",
    "fp32_matched",
    "fp32_recall",
    "fp32_ci95_low",
    "fp32_ci95_high",
    "int8_matched",
    "int8_recall",
    "int8_ci95_low",
    "int8_ci95_high",
    "delta_int8_minus_fp32",
)

GROUP_COLUMNS = (
    "size_group",
    "gt_count",
    "prohibitory",
    "danger",
    "mandatory",
    "area_min_px2",
    "area_max_px2",
    "area_mean_px2",
    "area_median_px2",
)


def load_ground_truth_for_images(
    image_paths: list[Path], label_dir: Path
) -> dict[str, np.ndarray]:
    """Load YOLO labels for every image using the original image dimensions."""
    labels_by_stem: dict[str, np.ndarray] = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        height, width = image.shape[:2]
        labels_by_stem[image_path.stem] = bench.load_ground_truth(
            label_dir / f"{image_path.stem}.txt", width, height
        )
    return labels_by_stem


def collect_gt_objects(
    image_paths: list[Path], labels_by_stem: dict[str, np.ndarray]
) -> list[dict]:
    """Flatten GT labels into objects with original-pixel bbox areas."""
    objects: list[dict] = []
    for image_path in image_paths:
        labels = labels_by_stem[image_path.stem]
        for object_index, row in enumerate(labels):
            width = max(0.0, float(row[3] - row[1]))
            height = max(0.0, float(row[4] - row[2]))
            objects.append(
                {
                    "image": image_path.stem,
                    "object_index": int(object_index),
                    "class_id": int(row[0]),
                    "class_name": CLASS_NAMES[int(row[0])],
                    "area_px2": width * height,
                }
            )
    if not objects:
        raise RuntimeError("No ground-truth objects found")
    return objects


def assign_size_groups(areas: np.ndarray) -> tuple[np.ndarray, dict]:
    """Assign small/medium/large groups using GT-area tertiles.

    Equal areas always receive the same group. Quantile thresholds are saved
    so the grouping can be audited and reproduced.
    """
    values = np.asarray(areas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("areas must be a non-empty 1-D array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("areas must be finite and non-negative")
    low, high = np.quantile(values, GROUP_QUANTILES)
    group_ids = np.where(values <= low, 0, np.where(values <= high, 1, 2))
    thresholds = {
        "method": "tertiles_of_gt_bbox_area",
        "quantiles": list(GROUP_QUANTILES),
        "small_max_area_px2": float(low),
        "medium_max_area_px2": float(high),
        "classification_rule": (
            "small: area <= small_max; "
            "medium: small_max < area <= medium_max; "
            "large: area > medium_max"
        ),
    }
    return group_ids.astype(np.int8), thresholds


def summarize_size_groups(
    objects: list[dict], group_ids: np.ndarray, thresholds: dict
) -> dict:
    """Record group sizes, area ranges, and class composition."""
    groups = []
    for group_id, name in enumerate(SIZE_GROUPS):
        indices = np.flatnonzero(group_ids == group_id)
        areas = np.asarray(
            [objects[index]["area_px2"] for index in indices], dtype=np.float64
        )
        class_counts = {class_name: 0 for class_name in CLASS_NAMES}
        for index in indices:
            class_counts[CLASS_NAMES[objects[index]["class_id"]]] += 1
        groups.append(
            {
                "name": name,
                "gt_count": int(indices.size),
                "class_counts": class_counts,
                "area_min_px2": float(areas.min()) if areas.size else None,
                "area_max_px2": float(areas.max()) if areas.size else None,
                "area_mean_px2": float(areas.mean()) if areas.size else None,
                "area_median_px2": float(np.median(areas)) if areas.size else None,
            }
        )
    return {
        "method": thresholds["method"],
        "quantiles": thresholds["quantiles"],
        "thresholds": {
            "small_max_area_px2": thresholds["small_max_area_px2"],
            "medium_max_area_px2": thresholds["medium_max_area_px2"],
        },
        "classification_rule": thresholds["classification_rule"],
        "small_sample_warning_threshold": SMALL_SAMPLE_THRESHOLD,
        "groups": groups,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (None, None)
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def match_gt_unique(
    labels: np.ndarray,
    predictions: np.ndarray,
    iou_threshold: float = DEFAULT_MATCH_IOU,
    conf_threshold: float = DEFAULT_CONF,
) -> np.ndarray:
    """Confidence-first one-to-one matching, restricted to the same class.

    Each ground-truth object can be claimed by at most one prediction, and each
    prediction can claim at most one ground-truth object. A prediction is only
    eligible when its confidence is >= conf_threshold and the IoU with the
    chosen object is >= iou_threshold.
    """
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 5)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 6)
    matched = np.zeros(len(labels), dtype=bool)
    if len(labels) == 0 or len(predictions) == 0:
        return matched
    candidates = predictions[predictions[:, 4] >= conf_threshold]
    if len(candidates) == 0:
        return matched
    candidates = candidates[np.argsort(-candidates[:, 4], kind="stable")]
    ious = bench.iou_matrix(labels[:, 1:5], candidates[:, :4])
    ious *= labels[:, 0:1] == candidates[None, :, 5]
    claimed = np.zeros(len(labels), dtype=bool)
    for prediction_index in range(len(candidates)):
        available = np.where(claimed, -1.0, ious[:, prediction_index])
        best_gt = int(np.argmax(available))
        if available[best_gt] >= iou_threshold:
            claimed[best_gt] = True
            matched[best_gt] = True
    return matched


def compute_recall_rows(
    objects: list[dict],
    group_ids: np.ndarray,
    matched_by_model: dict[tuple[int, str], np.ndarray],
    seeds: list[int],
    thresholds: dict,
) -> list[dict]:
    """Compute FP32/INT8 recall and signed deltas per seed and size group."""
    rows = []
    for seed in seeds:
        fp32 = matched_by_model[(seed, "fp32")]
        int8 = matched_by_model[(seed, "int8")]
        if len(fp32) != len(objects) or len(int8) != len(objects):
            raise ValueError("matched arrays must align with GT objects")
        for group_id, name in enumerate(SIZE_GROUPS):
            mask = group_ids == group_id
            indices = np.flatnonzero(mask)
            total = int(indices.size)
            areas = np.asarray(
                [objects[index]["area_px2"] for index in indices], dtype=np.float64
            )
            fp32_matched = int(fp32[mask].sum())
            int8_matched = int(int8[mask].sum())
            fp32_recall = fp32_matched / total if total else None
            int8_recall = int8_matched / total if total else None
            fp32_low, fp32_high = wilson_interval(fp32_matched, total)
            int8_low, int8_high = wilson_interval(int8_matched, total)
            delta = (
                int8_recall - fp32_recall
                if fp32_recall is not None and int8_recall is not None
                else None
            )
            rows.append(
                {
                    "seed": seed,
                    "size_group": name,
                    "gt_count": total,
                    "area_min_px2": float(areas.min()) if areas.size else None,
                    "area_max_px2": float(areas.max()) if areas.size else None,
                    "small_max_area_px2": thresholds["small_max_area_px2"],
                    "medium_max_area_px2": thresholds["medium_max_area_px2"],
                    "fp32_matched": fp32_matched,
                    "fp32_recall": fp32_recall,
                    "fp32_ci95_low": fp32_low,
                    "fp32_ci95_high": fp32_high,
                    "int8_matched": int8_matched,
                    "int8_recall": int8_recall,
                    "int8_ci95_low": int8_low,
                    "int8_ci95_high": int8_high,
                    "delta_int8_minus_fp32": delta,
                }
            )
    return rows


def compute_class_recall_rows(
    objects: list[dict],
    matched_by_model: dict[tuple[int, str], np.ndarray],
    seeds: list[int],
) -> list[dict]:
    """Compute unstratified per-class recall to expose class effects."""
    class_ids = np.asarray([obj["class_id"] for obj in objects], dtype=int)
    rows = []
    for seed in seeds:
        for variant in ("fp32", "int8"):
            matched = matched_by_model[(seed, variant)]
            for class_id, class_name in enumerate(CLASS_NAMES):
                mask = class_ids == class_id
                total = int(mask.sum())
                hits = int(matched[mask].sum())
                low, high = wilson_interval(hits, total)
                rows.append(
                    {
                        "seed": seed,
                        "format": variant,
                        "class_name": class_name,
                        "gt_count": total,
                        "matched": hits,
                        "recall": hits / total if total else None,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return rows


def summarize_results(rows: list[dict]) -> dict:
    """Summarize recall and deltas across seeds for each size group."""
    by_group = []
    for name in SIZE_GROUPS:
        group_rows = [row for row in rows if row["size_group"] == name]
        fp32 = [
            row["fp32_recall"]
            for row in group_rows
            if row["fp32_recall"] is not None
        ]
        int8 = [
            row["int8_recall"]
            for row in group_rows
            if row["int8_recall"] is not None
        ]
        deltas = [
            row["delta_int8_minus_fp32"]
            for row in group_rows
            if row["delta_int8_minus_fp32"] is not None
        ]
        by_group.append(
            {
                "size_group": name,
                "seeds": len(group_rows),
                "fp32_recall_mean": float(np.mean(fp32)) if fp32 else None,
                "int8_recall_mean": float(np.mean(int8)) if int8 else None,
                "delta_mean": float(np.mean(deltas)) if deltas else None,
                "delta_min": float(np.min(deltas)) if deltas else None,
                "delta_max": float(np.max(deltas)) if deltas else None,
            }
        )
    return {"by_group": by_group}


def build_limitations(
    objects: list[dict], group_summary: dict, small_sample_threshold: int = SMALL_SAMPLE_THRESHOLD
) -> list[str]:
    """Generate interpretation limits from class balance and cell counts."""
    class_totals = {class_name: 0 for class_name in CLASS_NAMES}
    for obj in objects:
        class_totals[obj["class_name"]] += 1
    totals_text = ", ".join(
        f"{class_name}={count}" for class_name, count in class_totals.items()
    )
    rarest_class = min(class_totals, key=class_totals.get)
    small_cells = []
    for group in group_summary["groups"]:
        for class_name in CLASS_NAMES:
            count = group["class_counts"][class_name]
            if count < small_sample_threshold:
                small_cells.append(f"{group['name']}/{class_name}={count}")
    cells_text = ", ".join(small_cells) if small_cells else "none"
    return [
        (
            "Size groups are tertiles of GT bbox area in original image pixels; "
            "the thresholds are dataset-specific and do not transfer directly "
            "to other camera resolutions or viewpoints."
        ),
        (
            "Groups are not class-stratified. Overall GT counts are "
            f"{totals_text}; class composition differs by group, so size-group "
            "recall differences may be confounded by class effects."
        ),
        (
            f"Small-sample cells (<{small_sample_threshold} GT boxes): "
            f"{cells_text}. The rarest class is {rarest_class} "
            f"({class_totals[rarest_class]} GT boxes); confidence intervals are "
            "wide and no significance test is performed."
        ),
        (
            "The 227-image test split comes from video-derived GTSDB data; "
            "near-duplicate frames reduce the effective number of independent "
            "samples."
        ),
        (
            "Matching uses one operating point (confidence >= 0.25, IoU >= 0.50); "
            "recall rankings can change at other thresholds."
        ),
        (
            "Each model is evaluated in one deterministic pass; quantization "
            "calibration uncertainty and repeated-run variance are not estimated."
        ),
        (
            "Only three training seeds are compared, so seed-level variation "
            "limits strong conclusions about INT8 degradation."
        ),
    ]


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    _write_csv(path, rows, RECALL_COLUMNS)


def write_group_csv(group_summary: dict, path: Path) -> None:
    rows = []
    for group in group_summary["groups"]:
        rows.append(
            {
                "size_group": group["name"],
                "gt_count": group["gt_count"],
                **group["class_counts"],
                "area_min_px2": group["area_min_px2"],
                "area_max_px2": group["area_max_px2"],
                "area_mean_px2": group["area_mean_px2"],
                "area_median_px2": group["area_median_px2"],
            }
        )
    _write_csv(path, rows, GROUP_COLUMNS)


def plot_results(rows: list[dict], group_summary: dict, path: Path) -> None:
    """Create a three-panel PNG: recall, per-seed delta, class composition."""
    if plt is None:
        raise RuntimeError("matplotlib is required to write the PNG chart")
    groups = list(SIZE_GROUPS)
    seeds = sorted({row["seed"] for row in rows})
    group_counts = {
        group["name"]: group["gt_count"] for group in group_summary["groups"]
    }

    def mean_or_zero(values: list) -> float:
        present = [value for value in values if value is not None]
        return float(np.mean(present)) if present else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    positions = np.arange(len(groups))
    width = 0.36

    fp32_means = []
    int8_means = []
    for name in groups:
        group_rows = [row for row in rows if row["size_group"] == name]
        fp32_means.append(
            mean_or_zero([row["fp32_recall"] for row in group_rows])
        )
        int8_means.append(
            mean_or_zero([row["int8_recall"] for row in group_rows])
        )
    axes[0].bar(positions - width / 2, fp32_means, width, label="FP32")
    axes[0].bar(positions + width / 2, int8_means, width, label="INT8")
    for seed in seeds:
        seed_rows = {row["size_group"]: row for row in rows if row["seed"] == seed}
        for index, name in enumerate(groups):
            row = seed_rows.get(name)
            if row is None:
                continue
            if row["fp32_recall"] is not None:
                axes[0].scatter(
                    index - width / 2,
                    row["fp32_recall"],
                    color="black",
                    marker="o",
                    s=18,
                    zorder=3,
                )
            if row["int8_recall"] is not None:
                axes[0].scatter(
                    index + width / 2,
                    row["int8_recall"],
                    color="black",
                    marker="D",
                    s=18,
                    zorder=3,
                )
    axes[0].set_xticks(
        positions,
        [f"{name}\n(n={group_counts.get(name, 0)})" for name in groups],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Recall (conf >= 0.25, IoU >= 0.50)")
    axes[0].set_title("FP32 vs INT8 recall by GT size")
    axes[0].legend()

    delta_width = 0.8 / max(len(seeds), 1)
    for seed_index, seed in enumerate(seeds):
        deltas = []
        for name in groups:
            row = next(
                (
                    item
                    for item in rows
                    if item["seed"] == seed and item["size_group"] == name
                ),
                None,
            )
            deltas.append(
                row["delta_int8_minus_fp32"]
                if row is not None and row["delta_int8_minus_fp32"] is not None
                else 0.0
            )
        offset = (seed_index - (len(seeds) - 1) / 2.0) * delta_width
        axes[1].bar(
            positions + offset, deltas, delta_width, label=f"seed {seed}"
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(positions, groups)
    axes[1].set_ylabel("INT8 recall - FP32 recall")
    axes[1].set_title("Per-seed recall delta")
    axes[1].legend()

    bottom = np.zeros(len(groups))
    for class_name in CLASS_NAMES:
        counts = np.asarray(
            [
                next(
                    group["class_counts"][class_name]
                    for group in group_summary["groups"]
                    if group["name"] == name
                )
                for name in groups
            ]
        )
        axes[2].bar(groups, counts, bottom=bottom, label=class_name)
        bottom += counts
    axes[2].set_ylabel("GT objects")
    axes[2].set_title("Class composition by size group")
    axes[2].legend()

    thresholds = group_summary["thresholds"]
    fig.suptitle(
        "Object-size recall analysis "
        f"(small <= {thresholds['small_max_area_px2']:.1f} px^2, "
        f"medium <= {thresholds['medium_max_area_px2']:.1f} px^2)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_session(model_path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPUExecutionProvider is not active")
    input_meta = session.get_inputs()[0]
    if list(input_meta.shape) != [1, 3, 640, 640]:
        raise ValueError(f"Unexpected model input shape: {input_meta.shape}")
    return session, input_meta.name


def run_model(
    model_path: Path,
    image_paths: list[Path],
    labels_by_stem: dict[str, np.ndarray],
    object_slices: dict[str, tuple[int, int]],
    n_objects: int,
    conf: float,
    match_iou: float,
    nms_iou: float,
    max_det: int,
    warmup: int,
    threads: int,
    model_label: str,
) -> np.ndarray:
    """Run one model and return a per-GT boolean matched array."""
    session, input_name = load_session(model_path, threads)
    matched = np.zeros(n_objects, dtype=bool)
    if warmup > 0:
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise FileNotFoundError(f"Cannot read {image_paths[0]}")
        tensor, _ = bench.preprocess(first)
        for _ in range(warmup):
            session.run(None, {input_name: tensor})
    decode_conf = float(np.nextafter(conf, -np.inf)) if conf > 0 else 0.0
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        tensor, geometry = bench.preprocess(image)
        output = session.run(None, {input_name: tensor})[0]
        predictions = bench.decode_prediction(
            output,
            geometry,
            conf_threshold=decode_conf,
            max_det=max_det,
            nms_iou=nms_iou,
        )
        labels = labels_by_stem[image_path.stem]
        matched_image = match_gt_unique(
            labels,
            predictions,
            iou_threshold=match_iou,
            conf_threshold=conf,
        )
        start, end = object_slices[image_path.stem]
        if end - start != len(matched_image):
            raise RuntimeError("GT object slice does not match label count")
        matched[start:end] = matched_image
        if index % 50 == 0:
            print(f"    {model_label}: {index}/{len(image_paths)} images", flush=True)
    return matched


def build_object_slices(
    image_paths: list[Path], labels_by_stem: dict[str, np.ndarray]
) -> tuple[dict[str, tuple[int, int]], int]:
    object_slices = {}
    offset = 0
    for image_path in image_paths:
        count = len(labels_by_stem[image_path.stem])
        object_slices[image_path.stem] = (offset, offset + count)
        offset += count
    return object_slices, offset


def metadata_common(
    args: argparse.Namespace, image_paths: list[Path], objects: list[dict], stamp: str
) -> dict:
    return {
        "run_at": stamp,
        "test_images": len(image_paths),
        "gt_objects": len(objects),
        "max_images": args.max_images,
        "confidence_threshold": args.conf,
        "match_iou_threshold": args.match_iou,
        "nms_iou_threshold": args.nms_iou,
        "max_det": args.max_det,
        "matching": (
            "confidence-first greedy one-to-one; same class; "
            "confidence >= threshold; IoU >= threshold"
        ),
        "size_grouping": "tertiles of GT bbox area in original image pixels",
        "area_units": "original image pixels squared",
        "preprocessing": (
            "benchmark_pi_v3.preprocess: 640x640 letterbox, BGR->RGB, "
            "NCHW FP32 /255"
        ),
        "postprocessing": (
            "benchmark_pi_v3.decode_prediction: raw [1,7,8400] -> "
            "multi-label candidates -> class-aware NMS"
        ),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified FP32 vs INT8 recall by GT object size"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing data/gtsdb and models",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seeds", type=int, nargs="+", choices=(0, 1, 2), default=(0, 1, 2)
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="SMOKE TEST only; subset results are not final",
    )
    parser.add_argument(
        "--groups-only",
        action="store_true",
        help="Only compute and save GT size groups; do not run inference",
    )
    parser.add_argument("--prefix", default="object_size_recall")
    args = parser.parse_args()

    if not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1]")
    if not 0 < args.match_iou <= 1:
        parser.error("--match-iou must be in (0, 1]")
    if not 0 < args.nms_iou < 1:
        parser.error("--nms-iou must be in (0, 1)")
    if args.warmup < 0 or args.threads < 1 or args.max_det < 1:
        parser.error("warmup >= 0, threads >= 1 and max-det >= 1 are required")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >= 1")
    args.seeds = list(dict.fromkeys(args.seeds))

    root = args.root.expanduser().resolve()
    image_dir = root / "data/gtsdb/images/test"
    label_dir = root / "data/gtsdb/labels/test"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No test images found under {image_dir}")
    if args.max_images is None and len(image_paths) != 227:
        raise RuntimeError(f"Expected 227 test images, found {len(image_paths)}")
    missing = [
        path.stem
        for path in image_paths
        if not (label_dir / f"{path.stem}.txt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing corresponding YOLO label files, e.g. {missing[:5]}"
        )

    cv2.setNumThreads(1)
    labels_by_stem = load_ground_truth_for_images(image_paths, label_dir)
    objects = collect_gt_objects(image_paths, labels_by_stem)
    areas = np.asarray([obj["area_px2"] for obj in objects], dtype=np.float64)
    group_ids, thresholds = assign_size_groups(areas)
    for obj, group_id in zip(objects, group_ids):
        obj["size_group"] = SIZE_GROUPS[int(group_id)]
    group_summary = summarize_size_groups(objects, group_ids, thresholds)
    object_slices, n_objects = build_object_slices(image_paths, labels_by_stem)
    if n_objects != len(objects):
        raise RuntimeError("GT object indexing is inconsistent")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(
        f"GT objects={len(objects)} images={len(image_paths)} "
        f"small_max={thresholds['small_max_area_px2']:.3f} px^2 "
        f"medium_max={thresholds['medium_max_area_px2']:.3f} px^2"
    )
    for group in group_summary["groups"]:
        if group["area_min_px2"] is None:
            area_text = "empty"
        else:
            area_text = (
                f"[{group['area_min_px2']:.1f}, "
                f"{group['area_max_px2']:.1f}]"
            )
        print(
            f"  {group['name']:<6} n={group['gt_count']:>3} "
            f"classes={group['class_counts']} area={area_text}"
        )

    limitations = build_limitations(objects, group_summary)

    if args.groups_only:
        payload = {
            "metadata": {
                **metadata_common(args, image_paths, objects, stamp),
                "mode": "groups_only",
                "inference_run": False,
            },
            "size_groups": group_summary,
            "limitations": limitations,
        }
        json_path = output_dir / f"{args.prefix}_groups_{stamp}.json"
        csv_path = output_dir / f"{args.prefix}_groups_{stamp}.csv"
        write_json(payload, json_path)
        write_group_csv(group_summary, csv_path)
        print(f"Saved: {json_path}\nSaved: {csv_path}")
        return

    model_specs = [
        (
            seed,
            variant,
            root / "models" / f"yolo26n_img640_seed{seed}_{variant}.onnx",
        )
        for seed in args.seeds
        for variant in ("fp32", "int8")
    ]
    missing_models = [
        str(model_path)
        for _, _, model_path in model_specs
        if not model_path.is_file()
    ]
    if missing_models:
        raise FileNotFoundError(f"Missing models: {missing_models}")

    matched_by_model: dict[tuple[int, str], np.ndarray] = {}
    for seed, variant, model_path in model_specs:
        label = f"seed{seed} {variant}"
        print(f"\nSTART {label} model={model_path.name}", flush=True)
        matched = run_model(
            model_path,
            image_paths,
            labels_by_stem,
            object_slices,
            n_objects,
            args.conf,
            args.match_iou,
            args.nms_iou,
            args.max_det,
            args.warmup,
            args.threads,
            label,
        )
        matched_by_model[(seed, variant)] = matched
        print(
            f"  matched {int(matched.sum())}/{len(matched)} GT objects",
            flush=True,
        )

    rows = compute_recall_rows(
        objects, group_ids, matched_by_model, args.seeds, thresholds
    )
    class_rows = compute_class_recall_rows(objects, matched_by_model, args.seeds)
    summary = summarize_results(rows)
    payload = {
        "metadata": {
            **metadata_common(args, image_paths, objects, stamp),
            "mode": "full",
            "inference_run": True,
            "seeds": args.seeds,
            "models": [
                {
                    "seed": seed,
                    "format": variant,
                    "file": model_path.name,
                    "bytes": model_path.stat().st_size,
                }
                for seed, variant, model_path in model_specs
            ],
            "onnxruntime_version": bench.ort.__version__,
        },
        "size_groups": group_summary,
        "results": rows,
        "summary": summary,
        "per_class_overall": class_rows,
        "limitations": limitations,
    }

    run_kind = "smoke" if args.max_images is not None else "full"
    base = f"{args.prefix}_{run_kind}_{stamp}"
    json_path = output_dir / f"{base}.json"
    csv_path = output_dir / f"{base}.csv"
    png_path = output_dir / f"{base}.png"
    write_json(payload, json_path)
    write_summary_csv(rows, csv_path)
    plot_results(rows, group_summary, png_path)

    print("\nseed group   GT  FP32   INT8   delta")
    for row in rows:
        print(
            f"{row['seed']:>4} {row['size_group']:<6} {row['gt_count']:>3} "
            f"{row['fp32_recall']:.4f} {row['int8_recall']:.4f} "
            f"{row['delta_int8_minus_fp32']:+.4f}"
        )
    print("\nLimitations:")
    for limitation in limitations:
        print(f"  - {limitation}")
    print(f"\nSaved: {json_path}\nSaved: {csv_path}\nSaved: {png_path}")


if __name__ == "__main__":
    main()
