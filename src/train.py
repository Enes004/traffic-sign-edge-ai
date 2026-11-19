#!/usr/bin/env python3
"""Train YOLO26n on the prepared GTSDB dataset with reproducible seeds."""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26n on GTSDB (3 classes).")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "gtsdb" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Base YOLO weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (640 or 320)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, 2)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    model_stem = Path(args.model).stem
    run_name = f"{model_stem}_img{args.imgsz}_seed{args.seed}"
    project_dir = REPO_ROOT / "runs" / "train"

    print(f"Starting training: {run_name} | data={args.data} | device={args.device}")
    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        device=args.device,
        fliplr=0.0,  # Avoid mirroring traffic sign numbers/symbols
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best_pt = project_dir / run_name / "weights" / "best.pt"
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    target_pt = models_dir / f"{model_stem}_img{args.imgsz}_seed{args.seed}.pt"

    if best_pt.is_file():
        shutil.copy2(best_pt, target_pt)
        print(f"\nSaved best checkpoint to: {target_pt}")

    metrics = results.results_dict
    print("\n--- Validation Summary (Best Checkpoint) ---")
    print(f"Precision : {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall    : {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP50     : {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95  : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Logs CSV  : {project_dir / run_name / 'results.csv'}")


if __name__ == "__main__":
    main()