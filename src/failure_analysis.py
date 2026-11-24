"""Generate matched error records and five-panel comparisons from test images."""
import argparse
import hashlib
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.patches import Rectangle

from benchmark_pi_v3 import CLASS_NAMES, iou_matrix, load_ground_truth
from local_inference import (
    ROOT, TEST_IMAGES, CONFIDENCE, NMS_IOU, MAX_DET, Detector, read_image,
)

CONFIGS = [(640, "fp32"), (640, "int8"), (320, "fp32"), (320, "int8")]
MATCH_IOU = 0.50
LOCALIZATION_FLOOR = 0.10
TYPES = ("classification_error", "localization_error", "false_positive", "false_negative")


def match_errors(labels, predictions):
    """Assign one-to-one matches, with TP priority over diagnostic matches."""
    overlaps = iou_matrix(predictions[:, :4], labels[:, 1:])
    same_class = predictions[:, 5, None] == labels[:, 0]
    matches = []
    used_pred, used_gt = set(), set()
    for kind, eligible in (
        ("true_positive", same_class & (overlaps >= MATCH_IOU)),
        ("classification_error", ~same_class & (overlaps >= MATCH_IOU)),
        ("localization_error", same_class & (overlaps >= LOCALIZATION_FLOOR)
         & (overlaps < MATCH_IOU)),
    ):
        for pred in np.argsort(-predictions[:, 4], kind="stable"):
            available = [gt for gt in np.flatnonzero(eligible[pred]) if gt not in used_gt]
            if pred in used_pred or not available:
                continue
            gt = max(available, key=lambda idx: overlaps[pred, idx])
            used_pred.add(pred)
            used_gt.add(gt)
            matches.append(dict(kind=kind, prediction=int(pred), ground_truth=int(gt),
                                iou=float(overlaps[pred, gt])))
    matches.extend(dict(kind="false_positive", prediction=pred, ground_truth=None, iou=None)
                   for pred in range(len(predictions)) if pred not in used_pred)
    matches.extend(dict(kind="false_negative", prediction=None, ground_truth=gt, iou=None)
                   for gt in range(len(labels)) if gt not in used_gt)
    return matches


def select_examples(records):
    """Prefer different images and the clearest observed match for each category."""
    selected = []
    used = set()
    for kind in TYPES:
        candidates = []
        for stem, record in records.items():
            if stem in used:
                continue
            for config, result in record["models"].items():
                for event in result["events"]:
                    if event["kind"] != kind:
                        continue
                    pred = event["prediction"]
                    gt = event["ground_truth"]
                    box = (record["ground_truth"][gt][1:] if gt is not None
                           else result["predictions"][pred][:4])
                    score = event["iou"]
                    if score is None:
                        score = (result["predictions"][pred][4] if pred is not None
                                 else (box[2] - box[0]) * (box[3] - box[1]))
                    candidates.append((score, stem, config, event, box))
        if candidates:
            _, stem, config, event, box = max(candidates, key=lambda candidate: candidate[0])
            used.add(stem)
            selected.append(dict(image_id=stem, model=config, **event, focus_xyxy=box,
                                 figure=f"{stem}_{kind}.png"))
    return selected


def draw_comparison(example, record, output, seed):
    stem = example["image_id"]
    rgb = read_image(TEST_IMAGES / f"{stem}.jpg")[:, :, ::-1]
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = example["focus_xyxy"]
    margin = max(55, 1.1 * max(x2 - x1, y2 - y1))
    crop = (max(0, x1 - margin), min(width, x2 + margin),
            min(height, y2 + margin), max(0, y1 - margin))
    fig, axes = plt.subplots(2, 5, figsize=(19, 6), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.45]})
    colors = ("#55edff", "#ffc557", "#f3a1ff")
    panels = {"Ground truth": [(row[1:], int(row[0]), None) for row in record["ground_truth"]]}
    panels.update({name: [(row[:4], int(row[5]), row[4]) for row in data["predictions"]]
                   for name, data in record["models"].items()})
    for col, (name, boxes) in enumerate(panels.items()):
        axes[0, col].set_title(f"{name} ({len(boxes)})")
        for index, axis in enumerate(axes[:, col]):
            axis.imshow(rgb)
            axis.axis("off")
            for (left, top, right, bottom), cls, score in reversed(boxes):
                color = colors[cls]
                axis.add_patch(Rectangle((left, top), right - left, bottom - top,
                                         fill=False, edgecolor=color, linewidth=1.6))
                label = CLASS_NAMES[cls] if score is None else f"{CLASS_NAMES[cls]} {score:.2f}"
                axis.text(left, top, label, color=color, fontsize=8 if index else 6,
                          clip_on=True, bbox=dict(facecolor="#171717", alpha=0.85, pad=1))
            if index:
                axis.set_xlim(*crop[:2])
                axis.set_ylim(*crop[2:])
            else:
                axis.add_patch(Rectangle((crop[0], crop[3]), crop[1] - crop[0],
                                         crop[2] - crop[3], fill=False, edgecolor="white",
                                         linewidth=0.8, linestyle="--"))
    detail = "" if example["iou"] is None else f" | IoU {example['iou']:.3f}"
    fig.suptitle(f"GTSDB {stem} | {example['kind'].replace('_', ' ')} in {example['model']}"
                 f"{detail} | seed {seed} | confidence > {CONFIDENCE:g} | matching IoU {MATCH_IOU:g}"
                 "\nTop: full scene   •   Bottom: same detail crop in every panel")
    fig.savefig(output / example["figure"], dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    image_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    if len(image_paths) != 227:
        parser.error(f"Expected 227 test images under {TEST_IMAGES}, found {len(image_paths)}")
    detectors = {f"{resolution} {variant.upper()}": Detector(resolution, variant, args.seed)
                 for resolution, variant in CONFIGS}
    records = {}
    counts = {name: Counter() for name in detectors}
    for path in image_paths:
        image = read_image(path)
        height, width = image.shape[:2]
        labels = load_ground_truth(TEST_IMAGES.parents[1] / "labels/test" / path.with_suffix(".txt").name,
                                   width, height)
        record = dict(ground_truth=labels.tolist(), models={})
        for name, detector in detectors.items():
            predictions = detector.predict(image)
            events = match_errors(labels, predictions)
            record["models"][name] = dict(predictions=predictions.tolist(), events=events)
            counts[name].update(event["kind"] for event in events)
        records[path.stem] = record
    examples = select_examples(records)
    output = ROOT / "results/failure_analysis"
    output.mkdir(parents=True, exist_ok=True)
    for example in examples:
        draw_comparison(example, records[example["image_id"]], output, args.seed)
    models = {name: dict(file=model.path.name, sha256=hashlib.sha256(model.path.read_bytes()).hexdigest())
              for name, model in detectors.items()}
    metadata = dict(seed=args.seed, test_images=len(records), host="local CPU",
                    provider="CPUExecutionProvider", onnxruntime=ort.__version__, threads=4,
                    confidence_strict_gt=CONFIDENCE, nms_iou=NMS_IOU, max_det=MAX_DET,
                    match_iou=MATCH_IOU, localization_iou_floor=LOCALIZATION_FLOOR,
                    coordinates="xyxy original pixels; GT [class,x1,y1,x2,y2]; predictions [x1,y1,x2,y2,confidence,class]",
                    classes=CLASS_NAMES, models=models)
    (output / "examples.json").write_text(
        json.dumps(dict(metadata=metadata, diagnostic_counts=counts, examples=examples,
                        selected_records={item["image_id"]: records[item["image_id"]] for item in examples}),
                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(counts=counts, examples=examples), indent=2))


if __name__ == "__main__":
    main()
