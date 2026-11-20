#!/usr/bin/env python3
"""Compare PyTorch (.pt) and exported FP32 ONNX (.onnx) on the validation split."""

import argparse
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Verify FP32 ONNX parity against PyTorch .pt.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to verify (0, 1, or 2)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "gtsdb" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    return parser.parse_args()


def eval_model(model_path: Path, data_yaml: Path, imgsz: int) -> dict:
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
    m = res.results_dict
    return {
        "P": m.get("metrics/precision(B)", 0.0),
        "R": m.get("metrics/recall(B)", 0.0),
        "mAP50": m.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": m.get("metrics/mAP50-95(B)", 0.0),
    }


def main():
    args = parse_args()
    models_dir = REPO_ROOT / "models"
    pt_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}.pt"
    onnx_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}_fp32.onnx"

    for p in (pt_path, onnx_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing model file: {p}")

    print(f"Evaluating PyTorch (.pt)  : {pt_path.name}")
    pt_m = eval_model(pt_path, args.data, args.imgsz)

    print(f"Evaluating ONNX (FP32)    : {onnx_path.name}")
    onnx_m = eval_model(onnx_path, args.data, args.imgsz)

    print(f"\n--- Parity Check (Seed {args.seed}, Val Split, imgsz={args.imgsz}) ---")
    print(f"{'Metric':<12} {'PyTorch (.pt)':>14} {'ONNX (FP32)':>14} {'Delta':>10}")
    print("-" * 54)
    for k in ("P", "R", "mAP50", "mAP50-95"):
        diff = onnx_m[k] - pt_m[k]
        print(f"{k:<12} {pt_m[k]:>14.4f} {onnx_m[k]:>14.4f} {diff:>+10.4f}")

    max_map_diff = max(abs(onnx_m["mAP50"] - pt_m["mAP50"]), abs(onnx_m["mAP50-95"] - pt_m["mAP50-95"]))
    status = "PASS" if max_map_diff < 0.005 else "CHECK"
    print(f"\nParity Status: {status} (max mAP delta = {max_map_diff:.4f})")


if __name__ == "__main__":
    main()
