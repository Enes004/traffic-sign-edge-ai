#!/usr/bin/env python3
"""Export a trained YOLO26n checkpoint to FP32 ONNX (640px)."""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO26n .pt model to FP32 ONNX.")
    parser.add_argument("--seed", type=int, default=0, help="Seed to export (0, 1, or 2)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = REPO_ROOT / "models"
    pt_path = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}.pt"

    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {pt_path}")

    print(f"Exporting {pt_path.name} -> FP32 ONNX (imgsz={args.imgsz}, opset={args.opset})...")
    model = YOLO(str(pt_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,
            half=False,
        )
    )

    target = models_dir / f"yolo26n_img{args.imgsz}_seed{args.seed}_fp32.onnx"
    shutil.move(str(exported), str(target))
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Saved: {target} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
