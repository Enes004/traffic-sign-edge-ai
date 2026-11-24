#!/usr/bin/env python3
"""Verify exported FP32 and INT8 ONNX models for integrity, shapes, inference, and val metrics.

Never touches the held-out test split; sample inference and optional evaluation
strictly use the validation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import benchmark_pi_v3 as bench

CLASS_NAMES = ("prohibitory", "danger", "mandatory")
PARITY_TOLERANCE = 0.005


def expected_anchors(imgsz: int) -> int:
    """Return total P3/8 + P4/16 + P5/32 detection grid cells for a square input."""
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError(f"imgsz must be a positive multiple of 32, got {imgsz!r}")
    return (imgsz // 8) ** 2 + (imgsz // 16) ** 2 + (imgsz // 32) ** 2


def validate_io_metadata(
    input_shape: Sequence[Any],
    input_type: str,
    output_shape: Sequence[Any],
    output_type: str,
    imgsz: int,
    num_classes: int = len(CLASS_NAMES),
) -> None:
    """Validate static ONNX input/output shapes and tensor dtypes."""
    expected_in = [1, 3, imgsz, imgsz]
    expected_out = [1, 4 + num_classes, expected_anchors(imgsz)]
    if list(input_shape) != expected_in:
        raise ValueError(f"Unexpected input shape {list(input_shape)}; expected {expected_in}")
    if input_type != "tensor(float)":
        raise ValueError(f"Unexpected input type {input_type!r}; expected 'tensor(float)'")
    if list(output_shape) != expected_out:
        raise ValueError(f"Unexpected output shape {list(output_shape)}; expected {expected_out}")
    if output_type != "tensor(float)":
        raise ValueError(f"Unexpected output type {output_type!r}; expected 'tensor(float)'")


def validate_quantized_graph(format_name: str, q_nodes: int, dq_nodes: int) -> None:
    """Ensure INT8 QDQ models contain quantized graph operators."""
    if format_name == "int8":
        if q_nodes <= 0 or dq_nodes <= 0:
            raise ValueError(
                f"INT8 model missing QDQ nodes: QuantizeLinear={q_nodes}, DequantizeLinear={dq_nodes}"
            )
    elif format_name != "fp32":
        raise ValueError(f"Unknown model format: {format_name!r}")


def compute_parity_status(pt_metrics: Mapping[str, float], onnx_metrics: Mapping[str, float]) -> dict[str, Any]:
    """Compare PyTorch .pt and FP32 ONNX validation metrics."""
    deltas = {
        key: float(onnx_metrics[key] - pt_metrics[key])
        for key in ("P", "R", "mAP50", "mAP50_95")
    }
    max_map_delta = max(abs(deltas["mAP50"]), abs(deltas["mAP50_95"]))
    return {
        "deltas": deltas,
        "max_map_delta": float(max_map_delta),
        "status": "PASS" if max_map_delta < PARITY_TOLERANCE else "CHECK",
    }


def summarize_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Compute arithmetic mean and population standard deviation across seeds."""
    if not rows:
        raise ValueError("Cannot summarize an empty sequence of metric rows")
    summary: dict[str, dict[str, float]] = {}
    for key in ("P", "R", "mAP50", "mAP50_95"):
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        summary[key] = {"mean": mean, "std": math.sqrt(var)}
    return summary


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_onnx_graph(model_path: Path) -> dict[str, Any]:
    """Load, validate with onnx.checker, and inspect graph operators and Conv bias types."""
    import onnx

    proto = onnx.load(str(model_path))
    onnx.checker.check_model(proto)

    init_types = {
        init.name: onnx.TensorProto.DataType.Name(init.data_type)
        for init in proto.graph.initializer
    }
    q_nodes = 0
    dq_nodes = 0
    conv_nodes = 0
    conv_bias_types: dict[str, int] = {}

    for node in proto.graph.node:
        if node.op_type == "QuantizeLinear":
            q_nodes += 1
        elif node.op_type == "DequantizeLinear":
            dq_nodes += 1
        elif node.op_type == "Conv":
            conv_nodes += 1
            if len(node.input) >= 3 and node.input[2]:
                bias_name = node.input[2]
                dtype = init_types.get(bias_name)
                if dtype is None:
                    # For QDQ graphs, Conv bias often comes from a DequantizeLinear node.
                    for producer in proto.graph.node:
                        if bias_name in producer.output and producer.input:
                            dtype = init_types.get(producer.input[0], producer.op_type)
                            break
                dtype = dtype or "UNKNOWN"
                conv_bias_types[dtype] = conv_bias_types.get(dtype, 0) + 1

    return {
        "opsets": [{"domain": op.domain or "ai.onnx", "version": op.version} for op in proto.opset_import],
        "total_nodes": len(proto.graph.node),
        "conv_nodes": conv_nodes,
        "quantize_linear_nodes": q_nodes,
        "dequantize_linear_nodes": dq_nodes,
        "conv_bias_types": conv_bias_types,
    }


def verify_single_onnx(
    model_path: Path,
    imgsz: int,
    format_name: str,
    sample_image_path: Path,
) -> dict[str, Any]:
    """Verify ONNX file opens, shapes match, QDQ nodes exist (INT8), and sample inference succeeds."""
    import onnxruntime as ort

    if not model_path.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")

    graph_info = inspect_onnx_graph(model_path)
    validate_quantized_graph(
        format_name,
        graph_info["quantize_linear_nodes"],
        graph_info["dequantize_linear_nodes"],
    )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    validate_io_metadata(
        input_meta.shape,
        input_meta.type,
        output_meta.shape,
        output_meta.type,
        imgsz=imgsz,
    )

    image = cv2.imread(str(sample_image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read sample validation image: {sample_image_path}")
    tensor, geometry = bench.preprocess(image, input_size=imgsz)
    raw_output = session.run(None, {input_meta.name: tensor})[0]

    expected_out_tuple = (1, 4 + len(CLASS_NAMES), expected_anchors(imgsz))
    if tuple(raw_output.shape) != expected_out_tuple:
        raise RuntimeError(
            f"Runtime output shape {tuple(raw_output.shape)} != expected {expected_out_tuple}"
        )
    if not np.isfinite(raw_output).all():
        raise RuntimeError(f"Non-finite values in sample inference output for {model_path.name}")

    decoded = bench.decode_prediction(raw_output, geometry, conf_threshold=0.25)
    size_bytes = model_path.stat().st_size

    return {
        "file": model_path.name,
        "format": format_name,
        "size_bytes": size_bytes,
        "size_mib": round(size_bytes / (1024 * 1024), 4),
        "sha256": sha256_file(model_path),
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "input_type": input_meta.type,
        "output_name": output_meta.name,
        "output_shape": list(output_meta.shape),
        "output_type": output_meta.type,
        "graph": graph_info,
        "smoke_inference": {
            "sample_image": sample_image_path.name,
            "all_finite": True,
            "output_min": float(np.min(raw_output)),
            "output_max": float(np.max(raw_output)),
            "detections_conf_025": int(len(decoded)),
        },
        "verified": True,
    }


def eval_ultralytics_val(model_path: Path, data_yaml: Path, imgsz: int) -> dict[str, Any]:
    """Evaluate a .pt or .onnx model on the validation split (batch=1, rect=False, CPU)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    res = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=1,
        rect=False,
        device="cpu",
        verbose=False,
        plots=False,
    )
    metrics = res.results_dict
    per_class: dict[str, dict[str, float]] = {}
    if hasattr(res, "box") and hasattr(res.box, "ap50") and hasattr(res.box, "ap"):
        for idx, class_name in enumerate(CLASS_NAMES):
            per_class[class_name] = {
                "mAP50": float(res.box.ap50[idx]),
                "mAP50_95": float(res.box.ap[idx]),
            }
    return {
        "P": float(metrics.get("metrics/precision(B)", 0.0)),
        "R": float(metrics.get("metrics/recall(B)", 0.0)),
        "mAP50": float(metrics.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
        "speed_ms": {k: float(v) for k, v in getattr(res, "speed", {}).items()},
        "per_class": per_class,
    }


def pick_sample_val_image(val_dir: Path) -> Path:
    """Pick the first non-empty-label validation image deterministically, or first sorted val image."""
    val_images = sorted(val_dir.glob("*.jpg"))
    if not val_images:
        raise FileNotFoundError(f"No validation images found in {val_dir}")
    label_dir = val_dir.parent.parent / "labels" / "val"
    for img_path in val_images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.is_file() and label_path.read_text(encoding="utf-8").strip():
            return img_path
    return val_images[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify FP32 and INT8 ONNX models (integrity, shapes, smoke inference, optional val metrics)."
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Square model input resolution (e.g. 320 or 640)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Model seeds to verify")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "models",
        help="Directory containing trained .pt and exported .onnx models",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "gtsdb" / "dataset.yaml",
        help="Path to dataset.yaml (only val split is used)",
    )
    parser.add_argument(
        "--eval-val",
        action="store_true",
        help="Also run Ultralytics validation (split=val, batch=1, rect=False, CPU) for .pt, FP32, and INT8",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write verification summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import onnx
    import onnxruntime as ort

    args = parse_args(argv)
    val_dir = REPO_ROOT / "data" / "gtsdb" / "images" / "val"
    sample_image = pick_sample_val_image(val_dir)

    seed_results: list[dict[str, Any]] = []
    pt_rows: list[dict[str, float]] = []
    fp32_rows: list[dict[str, float]] = []
    int8_rows: list[dict[str, float]] = []

    print(
        f"Verifying YOLO26n ONNX suite (imgsz={args.imgsz}, seeds={args.seeds}, "
        f"sample={sample_image.name})..."
    )

    for seed in args.seeds:
        pt_path = args.models_dir / f"yolo26n_img{args.imgsz}_seed{seed}.pt"
        fp32_path = args.models_dir / f"yolo26n_img{args.imgsz}_seed{seed}_fp32.onnx"
        int8_path = args.models_dir / f"yolo26n_img{args.imgsz}_seed{seed}_int8.onnx"

        fp32_info = verify_single_onnx(fp32_path, args.imgsz, "fp32", sample_image)
        int8_info = verify_single_onnx(int8_path, args.imgsz, "int8", sample_image)
        reduction_pct = (1.0 - int8_info["size_bytes"] / fp32_info["size_bytes"]) * 100.0

        print(
            f"  [Seed {seed}] FP32: {fp32_info['size_mib']:.2f} MiB {fp32_info['input_shape']}->{fp32_info['output_shape']} "
            f"(det@0.25={fp32_info['smoke_inference']['detections_conf_025']}) | "
            f"INT8: {int8_info['size_mib']:.2f} MiB (-{reduction_pct:.1f}%, "
            f"Q={int8_info['graph']['quantize_linear_nodes']}, DQ={int8_info['graph']['dequantize_linear_nodes']}, "
            f"det@0.25={int8_info['smoke_inference']['detections_conf_025']}) -> OK"
        )

        entry: dict[str, Any] = {
            "seed": seed,
            "pt_checkpoint": {
                "file": pt_path.name,
                "exists": pt_path.is_file(),
                "size_bytes": pt_path.stat().st_size if pt_path.is_file() else None,
                "sha256": sha256_file(pt_path) if pt_path.is_file() else None,
            },
            "fp32_onnx": fp32_info,
            "int8_onnx": int8_info,
            "size_reduction_pct": round(reduction_pct, 2),
        }

        if args.eval_val:
            if not pt_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint for --eval-val: {pt_path}")
            pt_val = eval_ultralytics_val(pt_path, args.data, args.imgsz)
            fp32_val = eval_ultralytics_val(fp32_path, args.data, args.imgsz)
            int8_val = eval_ultralytics_val(int8_path, args.data, args.imgsz)
            parity = compute_parity_status(pt_val, fp32_val)
            int8_deltas = {
                key: float(int8_val[key] - fp32_val[key])
                for key in ("P", "R", "mAP50", "mAP50_95")
            }
            entry["val_eval"] = {
                "split": "val",
                "batch": 1,
                "rect": False,
                "device": "cpu",
                "pt": pt_val,
                "fp32_onnx": fp32_val,
                "int8_onnx": int8_val,
                "fp32_parity": parity,
                "int8_minus_fp32": int8_deltas,
            }
            pt_rows.append(pt_val)
            fp32_rows.append(fp32_val)
            int8_rows.append(int8_val)
            print(
                f"           Val (rect=False, b=1): "
                f"FP32 mAP50={fp32_val['mAP50']:.4f} (parity {parity['status']} d={parity['max_map_delta']:.4f}) | "
                f"INT8 mAP50={int8_val['mAP50']:.4f} ({int8_deltas['mAP50']:+.4f}), "
                f"mAP50-95={int8_val['mAP50_95']:.4f} ({int8_deltas['mAP50_95']:+.4f})"
            )

        seed_results.append(entry)

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "imgsz": args.imgsz,
            "expected_input_shape": [1, 3, args.imgsz, args.imgsz],
            "expected_output_shape": [1, 4 + len(CLASS_NAMES), expected_anchors(args.imgsz)],
            "seeds": list(args.seeds),
            "sample_val_image": sample_image.name,
            "test_split_touched": False,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "calibration": {
                "split": "train",
                "calib_count": 100,
                "calib_seed": 42,
                "quant_format": "QDQ",
                "calibration_method": "MinMax",
                "op_types_to_quantize": ["Conv"],
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "per_channel": True,
            },
        },
        "seeds": seed_results,
    }

    if args.eval_val:
        payload["val_summary_mean_std"] = {
            "pt": summarize_metrics(pt_rows),
            "fp32_onnx": summarize_metrics(fp32_rows),
            "int8_onnx": summarize_metrics(int8_rows),
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved verification summary: {args.output}")

    print(f"\nAll {len(seed_results) * 2} ONNX models verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
