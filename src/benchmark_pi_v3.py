#!/usr/bin/env python3
"""Benchmark and evaluate YOLO26n ONNX models on Raspberry Pi 5.

Requires: numpy, opencv-python-headless, onnxruntime, psutil.
No PyTorch/Ultralytics installation needed on Raspberry Pi.
Model output: (1, 7, 8400) = xywh + 3 class scores (raw one-to-many head).
Raw output requires class-aware Non-Maximum Suppression (NMS).

Metrics: COCO-style mAP50 and mAP50-95 (101-point interpolated AP).
Validation NMS uses multi_label=True, matching Ultralytics DetectionValidator.postprocess
(v8.4.215), so each anchor may yield predictions for multiple classes.
Matching uses confidence-first one-to-one matching. AP uses the standard COCO
101-point mean precision and intentionally does not copy Ultralytics' trapezoid
sentinel approximation, so small numeric differences are expected. Validate
results against the laptop reference for the same ONNX model and test split
before claiming parity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

CLASS_NAMES = ("prohibitory", "danger", "mandatory")
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def preprocess(image: np.ndarray, input_size: int = 640) -> tuple[np.ndarray, tuple]:
    """Ultralytics-style fixed square letterbox + BGR->RGB + NCHW FP32."""
    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_w, pad_h = (input_size - new_w) / 2, (input_size - new_h) / 2
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, (scale, left, top, width, height)


def class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
    iou_threshold: float = 0.7, max_det: int = 300,
) -> np.ndarray:
    """Greedy NMS separately per class, then global top-score max_det.

    This is needed for our exported YOLO26 (1, 7, 8400) one-to-many output.
    The NMS-free YOLO26 export is instead (1, max_det, 6).
    """
    kept: list[int] = []
    for cls in np.unique(classes):
        indices = np.flatnonzero(classes == cls)
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        class_kept = 0
        while indices.size and class_kept < max_det:
            selected = int(indices[0])
            kept.append(selected)
            class_kept += 1
            indices = indices[1:]
            if not indices.size:
                break
            overlap = iou_matrix(boxes[selected:selected + 1], boxes[indices])[0]
            indices = indices[overlap <= iou_threshold]
    if not kept:
        return np.empty(0, dtype=np.intp)
    kept_array = np.asarray(kept, dtype=np.intp)
    return kept_array[np.argsort(-scores[kept_array], kind="stable")[:max_det]]


def decode_prediction(
    output: np.ndarray, geometry: tuple, conf_threshold: float = 0.001,
    max_det: int = 300, nms_iou: float = 0.7,
) -> np.ndarray:
    """Decode YOLO26 raw (1, 7, 8400) to N x 6: x1,y1,x2,y2,score,class.

    Use Ultralytics validation multi-label selection and per-class NMS on
    one-to-many head. The first four output channels are xywh in input pixels.
    """
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 7:
        raise ValueError(f"Unexpected output shape {output.shape}; expected [1,7,N]")
    pred = output[0]
    class_scores = pred[4:7, :]  # [class, anchor]

    # CRITICAL: Ultralytics DetectionValidator.postprocess sets multi_label=True.
    # An anchor with scores [0.80, 0.05, 0.70] must generate TWO candidate
    # detections (classes 0 and 2), not just the argmax (class 0).
    # np.nonzero returns matching (class, anchor) pairs.
    filtered_classes, chosen_anchors = np.nonzero(class_scores > conf_threshold)
    if chosen_anchors.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    # Perform class-aware NMS in 640x640 model coordinates, as Ultralytics does.
    cx, cy, bw, bh = pred[:4, chosen_anchors]
    model_boxes = np.column_stack((cx - bw / 2, cy - bh / 2,
                                   cx + bw / 2, cy + bh / 2)).astype(np.float32)
    filtered_scores = class_scores[filtered_classes, chosen_anchors].astype(np.float32)
    filtered_classes = filtered_classes.astype(np.int32)
    # Like Ultralytics, cap candidates before NMS if ever above max_nms.
    if len(filtered_scores) > 30000:
        top = np.argsort(-filtered_scores, kind="stable")[:30000]
        model_boxes = model_boxes[top]
        filtered_scores = filtered_scores[top]
        filtered_classes = filtered_classes[top]
    keep = class_aware_nms(model_boxes, filtered_scores, filtered_classes,
                           iou_threshold=nms_iou, max_det=max_det)
    boxes = model_boxes[keep].copy()
    scale, left, top, width, height = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return np.column_stack((boxes, filtered_scores[keep],
                            filtered_classes[keep])).astype(np.float32)


def load_ground_truth(path: Path, width: int, height: int) -> np.ndarray:
    """Load YOLO labels as Nx5: class,x1,y1,x2,y2 in original pixels."""
    if not path.exists():
        raise FileNotFoundError(f"Missing label: {path}")
    rows = []
    for line_num, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label {path}:{line_num}")
        cls, xc, yc, bw, bh = map(float, parts)
        if int(cls) != cls or not 0 <= cls < len(CLASS_NAMES):
            raise ValueError(f"Invalid class {cls} in {path}")
        rows.append((cls, (xc - bw / 2) * width, (yc - bh / 2) * height,
                     (xc + bw / 2) * width, (yc + bh / 2) * height))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise Intersection over Union between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])
    wh = np.maximum(bottom_right - top_left, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:4] - boxes_a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:4] - boxes_b[:, :2], 0), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def true_positive_matrix(detections: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confidence-first one-to-one GT matching following Ultralytics validator.

    Detections must be sorted by descending confidence, as returned by NMS.
    evaluate() enforces this ordering before calling the matcher.
    For every IoU threshold, a detection claims its best *available* target.
    This differs from sorting candidate pairs globally by IoU.
    """
    correct = np.zeros((len(detections), len(IOU_THRESHOLDS)), dtype=bool)
    if len(detections) == 0 or len(labels) == 0:
        return correct
    ious = iou_matrix(labels[:, 1:5], detections[:, :4])
    ious *= labels[:, 0:1] == detections[None, :, 5]
    claimed = np.zeros((len(labels), len(IOU_THRESHOLDS)), dtype=bool)
    # decode_prediction() already returns detections ordered by confidence.
    for pred_idx in np.flatnonzero((ious >= IOU_THRESHOLDS.min()).any(axis=0)):
        available = np.where(claimed, 0.0, ious[:, pred_idx, None])
        best_gt = available.argmax(axis=0)
        thresholds = np.arange(len(IOU_THRESHOLDS))
        hit = available[best_gt, thresholds] >= IOU_THRESHOLDS
        correct[pred_idx, :] = hit
        claimed[best_gt, thresholds] |= hit
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated Average Precision.

    recall must be non-decreasing. Precision is replaced by its envelope
    (maximum precision at or after each recall value), then sampled at 101
    evenly spaced recall levels and averaged. This gives AP=1.0 for a perfect
    detector and avoids the endpoint/sentinel bias of the Ultralytics
    trapezoid approximation.
    """
    if recall.size == 0:
        return 0.0
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_levels = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, recall_levels, side="left")
    valid = indices < recall.size
    sampled = np.where(
        valid, precision[np.minimum(indices, recall.size - 1)], 0.0
    )
    return float(sampled.mean())


def evaluate(entries: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Aggregate detection metrics over all test images and three classes.

    Predictions are sorted by descending confidence before matching so the
    result does not depend on the caller's array order.
    """
    targets = np.zeros(len(CLASS_NAMES), dtype=int)
    for _, labels in entries:
        for cls in labels[:, 0].astype(int):
            targets[cls] += 1
    all_pred, all_true = [], []
    for predictions, labels in entries:
        if len(predictions):
            predictions = predictions[
                np.argsort(-predictions[:, 4], kind="stable")
            ]
            all_pred.append(predictions)
            all_true.append(true_positive_matrix(predictions, labels))
    if all_pred:
        pred = np.concatenate(all_pred)
        true = np.concatenate(all_true)
    else:
        pred = np.empty((0, 6), dtype=np.float32)
        true = np.empty((0, 10), dtype=bool)
    by_class: dict[str, dict] = {}
    maps = []
    for c, name in enumerate(CLASS_NAMES):
        if targets[c] == 0:
            by_class[name] = {"gt_boxes": 0, "mAP50": None, "mAP50_95": None}
            continue
        indices = np.flatnonzero(pred[:, 5] == c)
        indices = indices[np.argsort(-pred[indices, 4], kind="stable")]
        tp = true[indices].astype(np.float64)
        cum_tp = np.cumsum(tp, axis=0)
        cum_fp = np.cumsum(1.0 - tp, axis=0)
        ap = np.zeros(10)
        for k in range(10):
            recall = cum_tp[:, k] / targets[c]
            precision = cum_tp[:, k] / np.maximum(cum_tp[:, k] + cum_fp[:, k], 1e-10)
            ap[k] = compute_ap(recall, precision)
        by_class[name] = {"gt_boxes": int(targets[c]), "mAP50": float(ap[0]),
                          "mAP50_95": float(np.mean(ap))}
        maps.append(ap)
    if not maps:
        raise RuntimeError("No ground-truth labels found; cannot calculate mAP")
    mean_aps = np.mean(maps, axis=0)
    return {"mAP50": float(mean_aps[0]), "mAP50_95": float(np.mean(mean_aps)),
            "gt_boxes": int(sum(targets)), "per_class": by_class}


def temperature_c() -> float | None:
    """Read Raspberry Pi CPU temperature where the platform provides it."""
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def benchmark_model(model_path: Path, images: list[Path], label_dir: Path,
                    repeat_count: int, warmup: int, threads: int,
                    conf: float, max_det: int, nms_iou: float) -> dict:
    """Time CPU inference; evaluate one pass; repeat passes for latency statistics."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPUExecutionProvider is not active")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != [1, 3, 640, 640]:
        raise ValueError(f"Unexpected model input shape: {input_meta.shape}")
    first = cv2.imread(str(images[0]))
    if first is None:
        raise FileNotFoundError(f"Cannot read {images[0]}")
    first_tensor, _ = preprocess(first)
    for _ in range(warmup):
        session.run(None, {input_meta.name: first_tensor})
    timings_ms: list[float] = []
    stage_timings: dict[str, list[float]] = {"preprocess": [], "postprocess": []}
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    start_temp = temperature_c()
    print(f"  Model: {model_path.name}; input={input_meta.shape}; output={output_meta.shape}", flush=True)
    for rep in range(repeat_count):
        for idx, image_path in enumerate(images):
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read {image_path}")
            start = time.perf_counter_ns()
            tensor, geometry = preprocess(image)
            pre_end = time.perf_counter_ns()
            output = session.run(None, {input_meta.name: tensor})[0]
            inf_end = time.perf_counter_ns()
            pred = decode_prediction(output, geometry, conf, max_det, nms_iou)
            end = time.perf_counter_ns()
            stage_timings["preprocess"].append((pre_end - start) / 1e6)
            timings_ms.append((inf_end - pre_end) / 1e6)
            stage_timings["postprocess"].append((end - inf_end) / 1e6)
            if rep == 0:
                labels = load_ground_truth(label_dir / f"{image_path.stem}.txt",
                                           image.shape[1], image.shape[0])
                samples.append((pred, labels))
            if (idx + 1) % 50 == 0:
                print(f"    repeat {rep+1}/{repeat_count}: {idx+1}/{len(images)} images", flush=True)
    scores = evaluate(samples)
    end_temp = temperature_c()
    mean_inf = statistics.mean(timings_ms)
    total_ms = statistics.mean(stage_timings["preprocess"]) + mean_inf + statistics.mean(stage_timings["postprocess"])
    return {
        "model": model_path.name,
        "nms_iou": nms_iou,
        "model_bytes": model_path.stat().st_size,
        "images": len(images),
        "repeats": repeat_count,
        "inference_mean_ms": mean_inf,
        "inference_median_ms": statistics.median(timings_ms),
        "inference_p95_ms": float(np.percentile(timings_ms, 95)),
        "preprocess_mean_ms": statistics.mean(stage_timings["preprocess"]),
        "postprocess_mean_ms": statistics.mean(stage_timings["postprocess"]),
        "pipeline_mean_ms_no_decode": total_ms,
        "inference_fps": 1000.0 / mean_inf,
        "pipeline_fps_no_decode": 1000.0 / total_ms,
        "temp_start_c": start_temp,
        "temp_end_c": end_temp,
        **scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 YOLO26n ONNX FP32-vs-INT8 benchmark")
    parser.add_argument("--root", type=Path, default=Path.home() / "traffic-sign-edge-ai")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=None,
                        help="Evaluate only one seed for parity checking")
    parser.add_argument("--variant", choices=("fp32", "int8"), default=None,
                        help="Evaluate only one precision variant")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.7,
                        help="Class-aware NMS IoU threshold for raw [1,7,8400] outputs")
    parser.add_argument("--max-images", type=int, default=None,
                        help="SMOKE TEST only. Results from subsets are not final mAP")
    args = parser.parse_args()
    if not 0 < args.nms_iou < 1:
        parser.error("--nms-iou must be between 0 and 1")
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("warmup >= 0, repeats >= 1 and threads >= 1 are required")
    root = args.root.expanduser().resolve()
    image_dir = root / "data/gtsdb/images/test"
    label_dir = root / "data/gtsdb/labels/test"
    model_dir = root / "models"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No test images found under {image_dir}")
    if args.max_images is None and len(image_paths) != 227:
        raise RuntimeError(f"Expected 227 test images, found {len(image_paths)}")
    missing = [p.stem for p in image_paths if not (label_dir / f"{p.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing corresponding YOLO label files, e.g. {missing[:5]}")
    cv2.setNumThreads(1)
    print(f"Raspberry Pi benchmark; images={len(image_paths)} threads={args.threads} "
          f"repeat={args.repeats} warmup={args.warmup}")
    print("NOTE: multi-label class-aware NMS, confidence-first matching, COCO 101-point AP; exact Ultralytics parity is not claimed")
    results = []
    # Alternating FP32/INT8 order across seeds partially balances device heating.
    selected_seeds = (args.seed,) if args.seed is not None else (0, 1, 2)
    for seed in selected_seeds:
        variants = (args.variant,) if args.variant else (("fp32", "int8") if seed != 1 else ("int8", "fp32"))
        for variant in variants:
            model = model_dir / f"yolo26n_img640_seed{seed}_{variant}.onnx"
            if not model.is_file():
                raise FileNotFoundError(f"Missing model: {model}")
            print(f"\nSTART seed={seed} {variant}", flush=True)
            row = benchmark_model(model, image_paths, label_dir, args.repeats,
                                  args.warmup, args.threads, args.conf, args.max_det, args.nms_iou)
            row["seed"] = seed
            row["format"] = variant
            results.append(row)
            print(f"  mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} "
                  f"inference={row['inference_mean_ms']:.2f}ms "
                  f"FPS={row['inference_fps']:.2f} "
                  f"temp={row['temp_end_c']}", flush=True)
    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "smoke" if args.max_images else "full"
    if args.seed is not None:
        prefix += f"_seed{args.seed}"
    if args.variant is not None:
        prefix += f"_{args.variant}"
    json_path = output_dir / f"pi_benchmark_{prefix}_{stamp}.json"
    csv_path = output_dir / f"pi_benchmark_{prefix}_{stamp}.csv"
    payload = {
        "metadata": {
            "run_at": stamp,
            "test_images": len(image_paths),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "threads": args.threads,
            "selected_seed": args.seed,
            "selected_variant": args.variant,
            "confidence_threshold": args.conf,
            "max_det": args.max_det,
            "host": "Raspberry Pi 5 (expected, verify externally)",
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_bytes": psutil.virtual_memory().total,
            "important": "Accuracy uses an independently implemented COCO-style AP and Ultralytics-compatible matching/NMS. Exact parity is not claimed; pipeline FPS excludes JPEG disk decode",
            "matching": "confidence-first greedy per-IoU matching; compatible with Ultralytics 8.4.215 default validator",
            "nms_multilabel": True,
            "ap_interpolation": "COCO 101-point mean precision with precision envelope; no trapezoid sentinel",
        },
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    columns = ("seed", "format", "model_bytes", "images", "repeats",
               "mAP50", "mAP50_95", "inference_mean_ms", "inference_median_ms",
               "inference_p95_ms", "inference_fps", "preprocess_mean_ms",
               "postprocess_mean_ms", "pipeline_mean_ms_no_decode",
               "pipeline_fps_no_decode", "temp_start_c", "temp_end_c")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["seed"], r["format"])))
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")
    for seed in selected_seeds:
        seed_rows = {x["format"]: x for x in results if x["seed"] == seed}
        if "fp32" not in seed_rows or "int8" not in seed_rows:
            continue
        fp, int8 = seed_rows["fp32"], seed_rows["int8"]
        print(f"Seed {seed}: INT8/FP32 mean speedup = "
              f"{fp['inference_mean_ms']/int8['inference_mean_ms']:.3f}x; "
              f"delta mAP50-95={int8['mAP50_95'] - fp['mAP50_95']:+.4f}")


if __name__ == "__main__":
    main()
