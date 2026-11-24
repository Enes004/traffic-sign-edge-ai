#!/usr/bin/env python3
"""Summarize the controlled 640 px and 320 px ABBA benchmark campaigns.

The script is read-only for raw results. It reads the two ABBA logs and only
the raw JSON/CSV files referenced by their ``Saved:`` lines:

* 23 Nov 2026, 640 px: 3 seeds x 2 formats x 2 ABBA repeats = 12 runs
* 24 Nov 2026, 320 px: 3 seeds x 2 formats x 2 ABBA repeats = 12 runs

Older benchmark files, smoke tests, and unrelated artifacts are excluded.
Each seed/format cell combines its two ABBA measurement repeats. Reported
mean +/- standard deviation uses the three training seeds (sample SD, n-1);
ABBA repeats are repeated measurements of the same checkpoint, not
independent training runs. Missing analyses (for example 320 px stratified
small-object recall) are reported explicitly and are never imputed.

Outputs are written to ``results/final`` as CSV, JSON, and 300 dpi PNG/PDF
figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FINAL_DIR = DEFAULT_RESULTS_DIR / "final"

SEEDS = (0, 1, 2)
FORMATS = ("fp32", "int8")
RESOLUTIONS = (640, 320)
ABBA_REPEATS_PER_CELL = 2
EXPECTED_RUNS = len(SEEDS) * len(FORMATS) * len(RESOLUTIONS) * ABBA_REPEATS_PER_CELL
EXPECTED_RUNS_PER_LOG = EXPECTED_RUNS // len(RESOLUTIONS)
EXPECTED_TEST_IMAGES = 227
EXPECTED_BENCH_REPEATS = 5
EXPECTED_DATE_BY_RESOLUTION = {640: "20261123", 320: "20261124"}
LOG_PATTERNS = ("abba_3seed_*.log", "abba_320_*.log")
SIZE_GROUPS = ("small", "medium", "large")
NOMINAL_FREQ_MHZ = 2400.0
TEMP_WARN_C = 80.0

COMBINED_FIELDS = (
    "mAP50",
    "mAP50_95",
    "inference_mean_ms",
    "inference_median_ms",
    "inference_p95_ms",
    "inference_fps",
    "preprocess_mean_ms",
    "postprocess_mean_ms",
    "pipeline_mean_ms_no_decode",
    "pipeline_fps_no_decode",
    "model_bytes",
    "model_mib",
    "temp_start_c",
    "temp_end_c",
)

SUMMARY_METRICS = COMBINED_FIELDS

METRIC_SPECS = {
    "mAP50": ("mAP50", "", True),
    "mAP50_95": ("mAP50-95", "", True),
    "inference_mean_ms": ("Inference latency", "ms", False),
    "inference_p95_ms": ("Inference p95 latency", "ms", False),
    "inference_fps": ("Inference FPS", "FPS", True),
    "pipeline_mean_ms_no_decode": ("Pipeline latency (no decode)", "ms", False),
    "pipeline_fps_no_decode": ("Pipeline FPS (no decode)", "FPS", True),
    "model_bytes": ("Model size", "bytes", False),
    "model_mib": ("Model size", "MiB", False),
}

EFFECT_METRICS = (
    "mAP50",
    "mAP50_95",
    "inference_mean_ms",
    "inference_fps",
    "model_mib",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
START_RE = re.compile(r"^START seed=(\d+) (fp32|int8)(?: imgsz=(\d+))?$")
MODEL_META_RE = re.compile(
    r"^\s*Model: (?P<model>\S+\.onnx)(?:; imgsz=(?P<imgsz>\d+))?; "
    r"input=\[(?P<input>[0-9,\s]+)\]; output=\[(?P<output>[0-9,\s]+)\]"
)
RESULT_RE = re.compile(
    r"^\s*mAP50=([0-9.]+) mAP50-95=([0-9.]+) inference=([0-9.]+)ms "
    r"FPS=([0-9.]+) temp=([0-9.]+)"
)
SAVED_RE = re.compile(r"^Saved: (.+)$")
FREQ_RE = re.compile(r"^frequency\(0\)=(\d+)$")
TEMP_RE = re.compile(r"^temp=([0-9.]+)'C$")
THROTTLE_RE = re.compile(r"^throttled=0x([0-9a-fA-F]+)$")
MODEL_NAME_RE = re.compile(r"yolo26n_img(\d+)_seed(\d+)_(fp32|int8)\.onnx$")
RUN_DATE_RE = re.compile(r"_(\d{8})_\d{6}\.(?:json|csv)$")

SEED_COLUMNS = (
    "resolution_px",
    "format",
    "seed",
    "abba_runs",
    "run_ids",
    "mAP50",
    "mAP50_95",
    "mAP50_abba_max_abs_diff",
    "mAP50_95_abba_max_abs_diff",
    "inference_mean_ms",
    "inference_median_ms",
    "inference_p95_ms",
    "inference_fps",
    "preprocess_mean_ms",
    "postprocess_mean_ms",
    "pipeline_mean_ms_no_decode",
    "pipeline_fps_no_decode",
    "model_bytes",
    "model_mib",
    "temp_start_c",
    "temp_end_c",
)

SUMMARY_COLUMNS = ("resolution_px", "format", "n_seeds", "abba_repeats_per_seed")
for _metric in SUMMARY_METRICS:
    SUMMARY_COLUMNS += (f"{_metric}_mean", f"{_metric}_std")

EFFECT_COLUMNS = (
    "effect_type",
    "resolution_px",
    "series",
    "baseline",
    "comparison",
    "metric",
    "unit",
    "baseline_mean",
    "comparison_mean",
    "absolute_delta",
    "percent_change",
    "higher_is_better",
    "outcome",
)

SMALL_OBJECT_COLUMNS = (
    "resolution_px",
    "size_group",
    "gt_count",
    "n_seeds",
    "fp32_recall_mean",
    "fp32_recall_std",
    "int8_recall_mean",
    "int8_recall_std",
    "delta_int8_minus_fp32_mean",
    "delta_int8_minus_fp32_std",
    "fp32_recall_pct",
    "int8_recall_pct",
    "source_json",
)

HEALTH_RUN_COLUMNS = (
    "run_index",
    "resolution_px",
    "seed",
    "format",
    "abba_repeat",
    "run_at",
    "pre_freq_mhz",
    "post_freq_mhz",
    "mean_freq_mhz",
    "min_freq_mhz",
    "max_freq_mhz",
    "pre_temp_c",
    "post_temp_c",
    "max_temp_c",
    "pre_throttled",
    "post_throttled",
    "throttle_flags",
    "throttled",
    "status",
)

HEALTH_SUMMARY_COLUMNS = (
    "resolution_px",
    "n_runs",
    "mean_freq_mhz",
    "min_freq_mhz",
    "max_freq_mhz",
    "freq_range_pct",
    "pre_temp_c_mean",
    "post_temp_c_mean",
    "max_temp_c",
    "throttled_runs",
    "status",
)


def fail(message: str) -> None:
    """Raise a consistent analysis error."""
    raise RuntimeError(message)


def sample_stats(values: list[float]) -> dict:
    """Return mean and sample standard deviation (n-1) for finite values."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        fail("Cannot summarize an empty value list")
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(clean), "values": clean}


def expected_anchors(imgsz: int) -> int:
    """Return P3/8 + P4/16 + P5/32 anchors for a square input."""
    return (imgsz // 8) ** 2 + (imgsz // 16) ** 2 + (imgsz // 32) ** 2


def discover_logs(results_dir: Path) -> list[Path]:
    """Find exactly one log for each required ABBA campaign."""
    logs: list[Path] = []
    for pattern in LOG_PATTERNS:
        matches = sorted(results_dir.glob(pattern))
        if len(matches) != 1:
            fail(
                f"Expected exactly one log matching {pattern!r} in {results_dir}, "
                f"found {len(matches)}: {[path.name for path in matches]}"
            )
        logs.append(matches[0])
    return logs


def parse_health(lines: list[str], index: int) -> tuple[dict, int]:
    """Parse one three-line frequency/temperature/throttle block."""
    frequency_match = FREQ_RE.match(lines[index])
    if frequency_match is None:
        fail(f"Invalid frequency line: {lines[index]!r}")
    if index + 2 >= len(lines):
        fail("Truncated CPU health block at end of log")
    temp_match = TEMP_RE.match(lines[index + 1])
    throttle_match = THROTTLE_RE.match(lines[index + 2])
    if temp_match is None or throttle_match is None:
        fail(
            "Incomplete CPU health block after "
            f"{lines[index]!r}: {lines[index + 1:index + 3]!r}"
        )
    return (
        {
            "frequency_hz": int(frequency_match.group(1)),
            "temp_c": float(temp_match.group(1)),
            "throttled": int(throttle_match.group(1), 16),
        },
        index + 3,
    )


def parse_log(log_path: Path) -> list[dict]:
    """Parse run boundaries, metrics, saved paths, and CPU health from a log."""
    raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ANSI_RE.sub("", line).rstrip("\r") for line in raw_lines]
    events: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if FREQ_RE.match(line):
            health, index = parse_health(lines, index)
            events.append(("health", health))
            continue
        start_match = START_RE.match(line)
        if start_match:
            events.append(
                (
                    "start",
                    {
                        "seed": int(start_match.group(1)),
                        "format": start_match.group(2),
                        "imgsz": (
                            int(start_match.group(3))
                            if start_match.group(3)
                            else None
                        ),
                    },
                )
            )
            index += 1
            continue
        model_match = MODEL_META_RE.match(line)
        if model_match:
            events.append(
                (
                    "model",
                    {
                        "model_name": model_match.group("model"),
                        "imgsz": (
                            int(model_match.group("imgsz"))
                            if model_match.group("imgsz")
                            else None
                        ),
                        "input_shape": [
                            int(part)
                            for part in model_match.group("input").split(",")
                        ],
                        "output_shape": [
                            int(part)
                            for part in model_match.group("output").split(",")
                        ],
                    },
                )
            )
            index += 1
            continue
        result_match = RESULT_RE.match(line)
        if result_match:
            events.append(
                (
                    "result",
                    {
                        "log_mAP50": float(result_match.group(1)),
                        "log_mAP50_95": float(result_match.group(2)),
                        "log_inference_ms": float(result_match.group(3)),
                        "log_fps": float(result_match.group(4)),
                        "log_temp_c": float(result_match.group(5)),
                    },
                )
            )
            index += 1
            continue
        saved_match = SAVED_RE.match(line)
        if saved_match:
            events.append(("saved", Path(saved_match.group(1).strip())))
        index += 1

    runs: list[dict] = []
    current: dict | None = None
    last_health: dict | None = None
    for kind, payload in events:
        if kind == "health":
            if current is not None:
                if (
                    current.get("saved_json") is None
                    or current.get("saved_csv") is None
                    or current.get("post_health") is not None
                ):
                    fail(
                        f"Unexpected CPU health block inside incomplete run "
                        f"in {log_path.name}: {current.get('start_line')}"
                    )
                current["post_health"] = payload
                runs.append(current)
                current = None
            else:
                last_health = payload
        elif kind == "start":
            if current is not None:
                fail(f"START encountered before previous run completed in {log_path}")
            current = {
                "log_file": log_path.name,
                "start_line": payload,
                "seed": payload["seed"],
                "format": payload["format"],
                "start_imgsz": payload["imgsz"],
                "pre_health": last_health,
                "post_health": None,
                "model_name": None,
                "model_imgsz": None,
                "model_input_shape": None,
                "model_output_shape": None,
                "log_metrics": None,
                "saved_json": None,
                "saved_csv": None,
            }
            last_health = None
        elif kind == "model":
            if current is None:
                fail(f"Model line outside a run in {log_path}")
            current.update(
                {
                    "model_name": payload["model_name"],
                    "model_imgsz": payload["imgsz"],
                    "model_input_shape": payload["input_shape"],
                    "model_output_shape": payload["output_shape"],
                }
            )
        elif kind == "result":
            if current is None:
                fail(f"Result line outside a run in {log_path}")
            current["log_metrics"] = payload
        elif kind == "saved":
            if current is None:
                fail(f"Saved line outside a run in {log_path}")
            if payload.suffix == ".json":
                current["saved_json"] = payload
            elif payload.suffix == ".csv":
                current["saved_csv"] = payload
            else:
                fail(f"Unexpected saved artifact: {payload}")
    if current is not None:
        fail(f"Incomplete final run in {log_path.name}: missing post-run health block")
    return runs


def parse_shape(value: object) -> list[int]:
    """Normalize a JSON/metadata shape field to a list of ints."""
    if value is None:
        return []
    return [int(part) for part in value]


def read_json_run(run: dict, results_dir: Path, models_dir: Path) -> dict:
    """Load, validate, and cross-check the JSON result for one run."""
    saved_json = run["saved_json"]
    local_json = results_dir / saved_json.name
    if not local_json.is_file():
        fail(f"Missing raw JSON referenced by log: {local_json}")
    payload = json.loads(local_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    rows = payload.get("results", [])
    if len(rows) != 1:
        fail(f"Expected exactly one result in {local_json}, found {len(rows)}")
    result = rows[0]
    model_name = run["model_name"]
    model_match = MODEL_NAME_RE.match(str(result.get("model", "")))
    if model_match is None:
        fail(f"Unexpected model name in {local_json}: {result.get('model')!r}")
    resolution = int(model_match.group(1))
    model_seed = int(model_match.group(2))
    model_format = model_match.group(3)
    if model_name != result["model"]:
        fail(
            f"Log/JSON model mismatch for {local_json}: "
            f"{model_name} vs {result['model']}"
        )
    if model_seed != run["seed"] or model_format != run["format"]:
        fail(f"Model filename does not match run metadata for {local_json}")
    if resolution not in RESOLUTIONS:
        fail(f"Unsupported resolution {resolution} in {local_json}")
    if run["start_imgsz"] is not None and run["start_imgsz"] != resolution:
        fail(
            f"START imgsz={run['start_imgsz']} disagrees with model resolution "
            f"{resolution} in {local_json}"
        )
    if run["model_imgsz"] is not None and run["model_imgsz"] != resolution:
        fail(f"Logged model imgsz disagrees with model name for {local_json}")
    expected_input = [1, 3, resolution, resolution]
    expected_output = [1, 7, expected_anchors(resolution)]
    if run["model_input_shape"] != expected_input:
        fail(
            f"Logged model input shape {run['model_input_shape']} != "
            f"{expected_input} in {local_json}"
        )
    if run["model_output_shape"] != expected_output:
        fail(
            f"Logged model output shape {run['model_output_shape']} != "
            f"{expected_output} in {local_json}"
        )
    if metadata.get("selected_seed") != run["seed"]:
        fail(f"JSON selected_seed mismatch in {local_json}")
    if metadata.get("selected_variant") != run["format"]:
        fail(f"JSON selected_variant mismatch in {local_json}")
    if int(metadata.get("test_images", -1)) != EXPECTED_TEST_IMAGES:
        fail(f"Unexpected test_images in {local_json}: {metadata.get('test_images')}")
    if int(metadata.get("repeats", -1)) != EXPECTED_BENCH_REPEATS:
        fail(f"Unexpected repeats in {local_json}: {metadata.get('repeats')}")
    if int(result.get("images", -1)) != EXPECTED_TEST_IMAGES:
        fail(f"Unexpected result images in {local_json}: {result.get('images')}")
    if int(result.get("repeats", -1)) != EXPECTED_BENCH_REPEATS:
        fail(f"Unexpected result repeats in {local_json}: {result.get('repeats')}")
    if metadata.get("imgsz") is not None and int(metadata["imgsz"]) != resolution:
        fail(f"JSON imgsz metadata mismatch in {local_json}")
    if metadata.get("input_shape") is not None:
        if parse_shape(metadata["input_shape"]) != expected_input:
            fail(f"JSON input_shape metadata mismatch in {local_json}")
    if metadata.get("output_shape") is not None:
        if parse_shape(metadata["output_shape"]) != expected_output:
            fail(f"JSON output_shape metadata mismatch in {local_json}")

    run_at = str(metadata.get("run_at", ""))
    if not run_at:
        fail(f"Missing run_at in {local_json}")
    expected_date = EXPECTED_DATE_BY_RESOLUTION[resolution]
    if not run_at.startswith(expected_date):
        fail(
            f"Run {local_json.name} has run_at={run_at}, expected date "
            f"{expected_date} for {resolution} px"
        )
    file_date_match = RUN_DATE_RE.search(saved_json.name)
    if file_date_match is None or file_date_match.group(1) != expected_date:
        fail(f"Unexpected run date in filename {saved_json.name}")

    local_csv = results_dir / run["saved_csv"].name
    if not local_csv.is_file():
        fail(f"Missing raw CSV referenced by log: {local_csv}")
    with local_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 1:
        fail(f"Expected exactly one CSV row in {local_csv}, found {len(csv_rows)}")
    csv_row = csv_rows[0]
    if int(csv_row["seed"]) != run["seed"] or csv_row["format"] != run["format"]:
        fail(f"CSV seed/format mismatch in {local_csv}")
    for field in ("mAP50", "mAP50_95", "inference_mean_ms", "model_bytes"):
        if not math.isclose(
            float(csv_row[field]),
            float(result[field]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            fail(f"CSV/JSON mismatch for {field} in {local_csv}")

    log_metrics = run["log_metrics"]
    if log_metrics is None:
        fail(f"Missing logged result line for {local_json.name}")
    if abs(log_metrics["log_mAP50"] - float(result["mAP50"])) > 1e-3:
        fail(f"Log/JSON mAP50 mismatch for {local_json.name}")
    if abs(log_metrics["log_mAP50_95"] - float(result["mAP50_95"])) > 1e-3:
        fail(f"Log/JSON mAP50-95 mismatch for {local_json.name}")
    if (
        abs(log_metrics["log_inference_ms"] - float(result["inference_mean_ms"]))
        > 0.05
    ):
        fail(f"Log/JSON latency mismatch for {local_json.name}")
    if abs(log_metrics["log_fps"] - float(result["inference_fps"])) > 0.02:
        fail(f"Log/JSON FPS mismatch for {local_json.name}")

    model_path = models_dir / str(result["model"])
    if not model_path.is_file():
        fail(f"Missing model file for size verification: {model_path}")
    if model_path.stat().st_size != int(result["model_bytes"]):
        fail(
            f"Model size mismatch for {model_path.name}: "
            f"{model_path.stat().st_size} vs {result['model_bytes']}"
        )

    enriched = dict(run)
    enriched.update(
        {
            "resolution_px": resolution,
            "run_at": run_at,
            "result": result,
            "metadata": metadata,
            "csv_row": csv_row,
            "local_json": local_json,
            "local_csv": local_csv,
            "model_bytes_on_disk": model_path.stat().st_size,
        }
    )
    return enriched


def load_runs(results_dir: Path, logs: list[Path]) -> tuple[list[dict], list[dict]]:
    """Parse logs, load referenced raw results, and enforce run completeness."""
    runs: list[dict] = []
    log_records: list[dict] = []
    models_dir = results_dir.parent / "models"
    for log_path in logs:
        parsed = parse_log(log_path)
        if len(parsed) != EXPECTED_RUNS_PER_LOG:
            fail(
                f"{log_path.name} contains {len(parsed)} completed runs; "
                f"expected {EXPECTED_RUNS_PER_LOG}"
            )
        resolutions = set()
        for run in parsed:
            model_match = MODEL_NAME_RE.match(str(run["model_name"]))
            if model_match is None:
                fail(f"Unparseable model name in {log_path.name}: {run['model_name']}")
            resolution = int(model_match.group(1))
            resolutions.add(resolution)
        if len(resolutions) != 1:
            fail(f"{log_path.name} mixes resolutions: {sorted(resolutions)}")
        resolution = resolutions.pop()
        if resolution not in RESOLUTIONS:
            fail(f"{log_path.name} contains unsupported resolution {resolution}")
        log_records.append(
            {
                "file": log_path.name,
                "resolution_px": resolution,
                "completed_runs": len(parsed),
                "expected_runs": EXPECTED_RUNS_PER_LOG,
                "first_run_at": None,
                "last_run_at": None,
            }
        )
        for run in parsed:
            runs.append(read_json_run(run, results_dir, models_dir))
    if len(runs) != EXPECTED_RUNS:
        fail(f"Loaded {len(runs)} runs; expected {EXPECTED_RUNS}")
    if {run["resolution_px"] for run in runs} != set(RESOLUTIONS):
        fail("Both 640 px and 320 px runs are required")
    cells: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for run in runs:
        cells[(run["resolution_px"], run["format"], run["seed"])].append(run)
    expected_cells = {
        (resolution, fmt, seed)
        for resolution in RESOLUTIONS
        for fmt in FORMATS
        for seed in SEEDS
    }
    if set(cells) != expected_cells:
        fail(
            "Unexpected seed/format/resolution cells: "
            f"{sorted(set(cells) ^ expected_cells)}"
        )
    for cell, cell_runs in cells.items():
        if len(cell_runs) != ABBA_REPEATS_PER_CELL:
            fail(f"Cell {cell} has {len(cell_runs)} ABBA runs; expected 2")
        dates = {RUN_DATE_RE.search(run["saved_json"].name).group(1) for run in cell_runs}
        if dates != {EXPECTED_DATE_BY_RESOLUTION[cell[0]]}:
            fail(f"Cell {cell} mixes dates: {sorted(dates)}")
    for record in log_records:
        matching = [
            run
            for run in runs
            if run["resolution_px"] == record["resolution_px"]
        ]
        matching.sort(key=lambda run: run["run_at"])
        record["first_run_at"] = matching[0]["run_at"]
        record["last_run_at"] = matching[-1]["run_at"]
    return runs, log_records


def combine_abba(runs: list[dict]) -> list[dict]:
    """Combine the two ABBA repeats into one record per seed and format."""
    cells: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for run in runs:
        cells[(run["resolution_px"], run["format"], run["seed"])].append(run)
    combined: list[dict] = []
    for (resolution, fmt, seed), cell_runs in sorted(cells.items()):
        cell_runs = sorted(cell_runs, key=lambda run: run["run_at"])
        row = {
            "resolution_px": resolution,
            "format": fmt,
            "seed": seed,
            "abba_runs": len(cell_runs),
            "run_ids": ";".join(run["run_at"] for run in cell_runs),
            "source_json_files": [
                run["local_json"].name for run in cell_runs
            ],
        }
        for field in COMBINED_FIELDS:
            values = []
            for run in cell_runs:
                if field == "model_mib":
                    values.append(float(run["result"]["model_bytes"]) / (1024.0**2))
                else:
                    values.append(float(run["result"][field]))
            row[field] = statistics.fmean(values)
        mAP50_values = [float(run["result"]["mAP50"]) for run in cell_runs]
        mAP50_95_values = [float(run["result"]["mAP50_95"]) for run in cell_runs]
        row["mAP50_abba_max_abs_diff"] = max(mAP50_values) - min(mAP50_values)
        row["mAP50_95_abba_max_abs_diff"] = max(mAP50_95_values) - min(
            mAP50_95_values
        )
        combined.append(row)
    return combined


def build_summary(seed_rows: list[dict]) -> list[dict]:
    """Compute mean and sample SD across the three training seeds."""
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["resolution_px"], row["format"])].append(row)
    summary = []
    for (resolution, fmt), rows in sorted(groups.items()):
        if len(rows) != len(SEEDS):
            fail(
                f"Summary cell {resolution}/{fmt} has {len(rows)} seeds; "
                f"expected {len(SEEDS)}"
            )
        entry = {
            "resolution_px": resolution,
            "format": fmt,
            "n_seeds": len(rows),
            "abba_repeats_per_seed": ABBA_REPEATS_PER_CELL,
            "metrics": {},
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            entry["metrics"][metric] = sample_stats(values)
        summary.append(entry)
    return summary


def summary_lookup(summary: list[dict]) -> dict[tuple[int, str], dict]:
    """Index summary entries by resolution and format."""
    return {(entry["resolution_px"], entry["format"]): entry for entry in summary}


def metric_mean(summary: list[dict], resolution: int, fmt: str, metric: str) -> float:
    """Return a summary metric mean for one configuration."""
    entry = summary_lookup(summary).get((resolution, fmt))
    if entry is None:
        fail(f"Missing summary entry for {resolution}/{fmt}")
    return float(entry["metrics"][metric]["mean"])


def outcome_for(percent_change: float, higher_is_better: bool) -> str:
    """Classify a relative change as improved, degraded, or unchanged."""
    tolerance = 1e-9
    if abs(percent_change) <= tolerance:
        return "unchanged"
    positive = percent_change > 0
    return "improved" if positive == higher_is_better else "degraded"


def build_effects(summary: list[dict]) -> list[dict]:
    """Compute quantization and resolution effects as percentages."""
    effects: list[dict] = []
    for resolution in RESOLUTIONS:
        for metric in EFFECT_METRICS:
            label, unit, higher_is_better = METRIC_SPECS[metric]
            baseline = metric_mean(summary, resolution, "fp32", metric)
            comparison = metric_mean(summary, resolution, "int8", metric)
            delta = comparison - baseline
            percent = (delta / baseline * 100.0) if baseline else float("nan")
            effects.append(
                {
                    "effect_type": "quantization",
                    "resolution_px": resolution,
                    "series": str(resolution),
                    "baseline": "fp32",
                    "comparison": "int8",
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "baseline_mean": baseline,
                    "comparison_mean": comparison,
                    "absolute_delta": delta,
                    "percent_change": percent,
                    "higher_is_better": higher_is_better,
                    "outcome": outcome_for(percent, higher_is_better),
                }
            )
    for fmt in FORMATS:
        for metric in EFFECT_METRICS:
            label, unit, higher_is_better = METRIC_SPECS[metric]
            baseline = metric_mean(summary, 640, fmt, metric)
            comparison = metric_mean(summary, 320, fmt, metric)
            delta = comparison - baseline
            percent = (delta / baseline * 100.0) if baseline else float("nan")
            effects.append(
                {
                    "effect_type": "resolution",
                    "resolution_px": "640->320",
                    "series": fmt,
                    "baseline": "640",
                    "comparison": "320",
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "baseline_mean": baseline,
                    "comparison_mean": comparison,
                    "absolute_delta": delta,
                    "percent_change": percent,
                    "higher_is_better": higher_is_better,
                    "outcome": outcome_for(percent, higher_is_better),
                }
            )
    return effects


def object_analysis_resolution(payload: dict) -> int | None:
    """Infer the benchmark resolution of a stratified object-size analysis."""
    resolutions = set()
    for model in payload.get("metadata", {}).get("models", []):
        match = re.search(r"_img(\d+)_", str(model.get("file", "")))
        if match:
            resolutions.add(int(match.group(1)))
    if len(resolutions) == 1:
        return resolutions.pop()
    return None


def load_small_object_analysis(results_dir: Path) -> dict:
    """Load the newest full stratified recall analysis for each resolution."""
    candidates = sorted(results_dir.glob("object_size_recall_full_*.json"))
    latest: dict[int, tuple[str, Path, dict]] = {}
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata", {})
        if metadata.get("mode") != "full" or not metadata.get("inference_run"):
            continue
        if int(metadata.get("test_images", -1)) != EXPECTED_TEST_IMAGES:
            continue
        if list(metadata.get("seeds", [])) != list(SEEDS):
            continue
        resolution = object_analysis_resolution(payload)
        if resolution not in RESOLUTIONS:
            continue
        run_at = str(metadata.get("run_at", ""))
        if resolution not in latest or run_at > latest[resolution][0]:
            latest[resolution] = (run_at, path, payload)

    groups = []
    sources = []
    available_resolutions = sorted(latest.keys(), reverse=True)
    for resolution in available_resolutions:
        run_at, path, payload = latest[resolution]
        results = payload.get("results", [])
        for group_name in SIZE_GROUPS:
            group_rows = sorted(
                [row for row in results if row.get("size_group") == group_name],
                key=lambda row: int(row["seed"]),
            )
            if len(group_rows) != len(SEEDS):
                fail(
                    f"{path.name} has {len(group_rows)} rows for {group_name}; "
                    f"expected {len(SEEDS)}"
                )
            if [int(row["seed"]) for row in group_rows] != list(SEEDS):
                fail(f"{path.name} has unexpected seeds for {group_name}")
            fp32_values = [float(row["fp32_recall"]) for row in group_rows]
            int8_values = [float(row["int8_recall"]) for row in group_rows]
            delta_values = [float(row["delta_int8_minus_fp32"]) for row in group_rows]
            fp32_stats = sample_stats(fp32_values)
            int8_stats = sample_stats(int8_values)
            delta_stats = sample_stats(delta_values)
            groups.append(
                {
                    "resolution_px": resolution,
                    "size_group": group_name,
                    "gt_count": int(group_rows[0]["gt_count"]),
                    "n_seeds": len(group_rows),
                    "fp32_recall": fp32_stats,
                    "int8_recall": int8_stats,
                    "delta_int8_minus_fp32": delta_stats,
                    "source_json": path.name,
                }
            )
        sources.append(
            {
                "resolution_px": resolution,
                "run_at": run_at,
                "file": path.name,
                "limitations": payload.get("limitations", []),
            }
        )
    missing = [
        resolution
        for resolution in RESOLUTIONS
        if resolution not in available_resolutions
    ]
    return {
        "available": bool(groups),
        "groups": groups,
        "sources": sources,
        "available_resolutions_px": available_resolutions,
        "missing_resolutions_px": missing,
        "note": (
            "Stratified small-object recall is computed from existing full "
            "analysis files only; no new inference was run. Missing resolutions "
            "are reported, not imputed."
        ),
    }


def build_system_health(runs: list[dict]) -> dict:
    """Summarize logged CPU frequency, temperature, and throttle flags."""
    order_index: dict[str, int] = {}
    for index, run in enumerate(
        sorted(runs, key=lambda run: run["run_at"]), start=1
    ):
        order_index[run["saved_json"].name] = index
    per_run = []
    for run in sorted(runs, key=lambda run: order_index[run["saved_json"].name]):
        pre = run["pre_health"]
        post = run["post_health"]
        if pre is None or post is None:
            fail(f"Missing CPU health data for {run['saved_json'].name}")
        frequencies_mhz = [
            pre["frequency_hz"] / 1e6,
            post["frequency_hz"] / 1e6,
        ]
        temps = [pre["temp_c"], post["temp_c"]]
        throttle_flags = int(pre["throttled"]) | int(post["throttled"])
        mean_freq = statistics.fmean(frequencies_mhz)
        status = "ok"
        if throttle_flags:
            status = "throttled"
        elif min(frequencies_mhz) < 0.95 * max(frequencies_mhz):
            status = "frequency_drop"
        if max(temps) >= TEMP_WARN_C:
            status = "high_temperature" if status == "ok" else status + "+high_temperature"
        per_run.append(
            {
                "run_index": order_index[run["saved_json"].name],
                "resolution_px": run["resolution_px"],
                "seed": run["seed"],
                "format": run["format"],
                "abba_repeat": None,
                "run_at": run["run_at"],
                "pre_freq_mhz": frequencies_mhz[0],
                "post_freq_mhz": frequencies_mhz[1],
                "mean_freq_mhz": mean_freq,
                "min_freq_mhz": min(frequencies_mhz),
                "max_freq_mhz": max(frequencies_mhz),
                "pre_temp_c": pre["temp_c"],
                "post_temp_c": post["temp_c"],
                "max_temp_c": max(temps),
                "pre_throttled": pre["throttled"],
                "post_throttled": post["throttled"],
                "throttle_flags": throttle_flags,
                "throttled": bool(throttle_flags),
                "status": status,
            }
        )
    for resolution in RESOLUTIONS:
        cell_counts: dict[tuple[int, str], int] = defaultdict(int)
        for row in sorted(per_run, key=lambda item: item["run_at"]):
            if row["resolution_px"] != resolution:
                continue
            key = (row["seed"], row["format"])
            cell_counts[key] += 1
            row["abba_repeat"] = cell_counts[key]

    summary = []
    for resolution in RESOLUTIONS:
        rows = [row for row in per_run if row["resolution_px"] == resolution]
        if len(rows) != EXPECTED_RUNS_PER_LOG:
            fail(f"CPU health summary for {resolution} px has {len(rows)} runs")
        frequencies = [
            value
            for row in rows
            for value in (row["pre_freq_mhz"], row["post_freq_mhz"])
        ]
        pre_temps = [row["pre_temp_c"] for row in rows]
        post_temps = [row["post_temp_c"] for row in rows]
        throttled_runs = sum(1 for row in rows if row["throttled"])
        freq_range_pct = (
            (max(frequencies) - min(frequencies)) / max(frequencies) * 100.0
        )
        max_temp = max(max(row["max_temp_c"] for row in rows), 0.0)
        status = "ok"
        if throttled_runs:
            status = "throttled"
        elif freq_range_pct > 5.0:
            status = "frequency_drop"
        if max_temp >= TEMP_WARN_C:
            status = "high_temperature" if status == "ok" else status + "+high_temperature"
        summary.append(
            {
                "resolution_px": resolution,
                "n_runs": len(rows),
                "mean_freq_mhz": statistics.fmean(frequencies),
                "min_freq_mhz": min(frequencies),
                "max_freq_mhz": max(frequencies),
                "freq_range_pct": freq_range_pct,
                "pre_temp_c_mean": statistics.fmean(pre_temps),
                "post_temp_c_mean": statistics.fmean(post_temps),
                "max_temp_c": max_temp,
                "throttled_runs": throttled_runs,
                "status": status,
            }
        )

    by_configuration = []
    for resolution in RESOLUTIONS:
        for fmt in FORMATS:
            rows = [
                row
                for row in per_run
                if row["resolution_px"] == resolution and row["format"] == fmt
            ]
            frequencies = [
                value
                for row in rows
                for value in (row["pre_freq_mhz"], row["post_freq_mhz"])
            ]
            by_configuration.append(
                {
                    "resolution_px": resolution,
                    "format": fmt,
                    "n_runs": len(rows),
                    "mean_freq_mhz": statistics.fmean(frequencies),
                    "min_freq_mhz": min(frequencies),
                    "max_freq_mhz": max(frequencies),
                    "max_temp_c": max(row["max_temp_c"] for row in rows),
                    "throttled_runs": sum(1 for row in rows if row["throttled"]),
                }
            )
    return {
        "per_run": per_run,
        "summary": summary,
        "by_configuration": by_configuration,
        "checks": {
            "throttling_absent": all(not row["throttled"] for row in per_run),
            "frequency_stable_within_5pct": all(
                row["freq_range_pct"] <= 5.0 for row in summary
            ),
            "temperature_below_80c": all(
                row["max_temp_c"] < TEMP_WARN_C for row in summary
            ),
            "nominal_frequency_mhz": NOMINAL_FREQ_MHZ,
            "temperature_warning_c": TEMP_WARN_C,
            "throttle_flags_observed": sorted(
                {row["throttle_flags"] for row in per_run}
            ),
        },
    }


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    """Write dictionaries to CSV using a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    """Write an indented UTF-8 JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def setup_matplotlib() -> None:
    """Apply a compact, academic figure style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure as 300 dpi PNG and vector PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def config_style(resolution: int, fmt: str) -> tuple[str, str]:
    """Return a stable label and color for a benchmark configuration."""
    colors = {
        (640, "fp32"): "#4C72B0",
        (640, "int8"): "#DD8452",
        (320, "fp32"): "#55A868",
        (320, "int8"): "#C44E52",
    }
    return f"{resolution} {fmt.upper()}", colors[(resolution, fmt)]


def plot_accuracy(summary: list[dict], path: Path) -> None:
    """Plot mAP50 and mAP50-95 with seed-level sample SD error bars."""
    lookup = summary_lookup(summary)
    configs = [(resolution, fmt) for resolution in RESOLUTIONS for fmt in FORMATS]
    labels = [config_style(*config)[0] for config in configs]
    colors = [config_style(*config)[1] for config in configs]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for axis, metric, title in (
        (axes[0], "mAP50", "mAP50"),
        (axes[1], "mAP50_95", "mAP50-95"),
    ):
        means = [lookup[config]["metrics"][metric]["mean"] for config in configs]
        stds = [lookup[config]["metrics"][metric]["std"] for config in configs]
        axis.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.0},
        )
        for xpos, mean, std in zip(x, means, stds):
            offset = std + 0.015 if std == std else 0.015
            axis.text(
                xpos,
                min(mean + offset, 1.02),
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(x, labels, rotation=12)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title(title)
    fig.suptitle(
        "Detection accuracy: mean ± sample SD across 3 training seeds "
        "(2 ABBA repeats combined per seed)"
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_latency_fps(summary: list[dict], path: Path) -> None:
    """Plot inference latency and FPS with seed-level sample SD error bars."""
    lookup = summary_lookup(summary)
    configs = [(resolution, fmt) for resolution in RESOLUTIONS for fmt in FORMATS]
    labels = [config_style(*config)[0] for config in configs]
    colors = [config_style(*config)[1] for config in configs]
    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for axis, metric, title in (
        (axes[0], "inference_mean_ms", "Inference latency (ms, lower is better)"),
        (axes[1], "inference_fps", "Inference throughput (FPS, higher is better)"),
    ):
        means = [lookup[config]["metrics"][metric]["mean"] for config in configs]
        stds = [lookup[config]["metrics"][metric]["std"] for config in configs]
        axis.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.0},
        )
        for xpos, mean, std in zip(x, means, stds):
            axis.text(
                xpos,
                mean + std + max(means) * 0.02,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(x, labels, rotation=12)
        axis.set_ylabel("Milliseconds" if "ms" in metric else "Frames per second")
        axis.set_title(title)
        axis.set_ylim(0.0, max(means) * 1.22)
    fig.suptitle(
        "Raspberry Pi 5 CPU benchmark: mean ± sample SD across 3 training seeds"
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_effect(
    effects: list[dict], effect_type: str, title: str, subtitle: str, path: Path
) -> None:
    """Plot grouped horizontal percentage changes for one effect type."""
    rows = [row for row in effects if row["effect_type"] == effect_type]
    if not rows:
        fail(f"No effect rows for {effect_type}")
    if effect_type == "quantization":
        series = [("640", "640 px"), ("320", "320 px")]
    else:
        series = [("fp32", "FP32"), ("int8", "INT8")]
    colors = {"640 px": "#4C72B0", "320 px": "#DD8452", "FP32": "#55A868", "INT8": "#C44E52"}
    metric_labels = [METRIC_SPECS[metric][0] for metric in EFFECT_METRICS]
    positions = np.arange(len(metric_labels))
    height = 0.36
    fig, axis = plt.subplots(figsize=(9.5, 5.4))
    all_values = [float(row["percent_change"]) for row in rows]
    for offset, (key, label) in zip((-height / 2, height / 2), series):
        values = []
        for metric in EFFECT_METRICS:
            match = next(
                row
                for row in rows
                if row["metric"] == metric and row["series"] == key
            )
            values.append(float(match["percent_change"]))
        bars = axis.barh(
            positions + offset,
            values,
            height=height,
            label=label,
            color=colors[label],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            horizontal = 4 if value >= 0 else -4
            axis.annotate(
                f"{value:+.2f}%",
                xy=(value, bar.get_y() + bar.get_height() / 2),
                xytext=(horizontal, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )
    axis.axvline(0.0, color="black", linewidth=0.9)
    axis.set_yticks(positions, metric_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Relative change (%)")
    span = max(all_values) - min(all_values)
    margin = max(span * 0.18, 5.0)
    axis.set_xlim(min(all_values) - margin, max(all_values) + margin)
    axis.set_title(title)
    axis.legend(loc="lower right")
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=8.5, color="#333333")
    fig.tight_layout()
    save_figure(fig, path)


def plot_small_object(small: dict, path: Path) -> None:
    """Plot stratified recall by ground-truth object size (640 px only)."""
    groups = [group for group in small["groups"] if group["resolution_px"] == 640]
    if not groups:
        fail("No 640 px stratified recall data available to plot")
    groups = sorted(groups, key=lambda group: SIZE_GROUPS.index(group["size_group"]))
    labels = [
        f"{group['size_group']}\n(n={group['gt_count']})" for group in groups
    ]
    x = np.arange(len(groups))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.6, 5.0))
    fp32_means = [group["fp32_recall"]["mean"] for group in groups]
    int8_means = [group["int8_recall"]["mean"] for group in groups]
    fp32_stds = [group["fp32_recall"]["std"] for group in groups]
    int8_stds = [group["int8_recall"]["std"] for group in groups]
    axis.bar(
        x - width / 2,
        fp32_means,
        width,
        yerr=fp32_stds,
        label="FP32",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.0},
    )
    axis.bar(
        x + width / 2,
        int8_means,
        width,
        yerr=int8_stds,
        label="INT8",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.0},
    )
    for group_index, (group, xpos) in enumerate(zip(groups, x)):
        for seed_index, _seed in enumerate(SEEDS):
            fp32_value = group["fp32_recall"]["values"][seed_index]
            int8_value = group["int8_recall"]["values"][seed_index]
            jitter = (seed_index - 1) * 0.045
            axis.scatter(
                xpos - width / 2 + jitter,
                fp32_value,
                color="black",
                marker="o",
                s=14,
                zorder=3,
            )
            axis.scatter(
                xpos + width / 2 + jitter,
                int8_value,
                color="black",
                marker="D",
                s=14,
                zorder=3,
            )
        axis.text(
            xpos - width / 2,
            fp32_means[group_index] + fp32_stds[group_index] + 0.008,
            f"{fp32_means[group_index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        axis.text(
            xpos + width / 2,
            int8_means[group_index] + int8_stds[group_index] + 0.008,
            f"{int8_means[group_index]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.6, 1.02)
    axis.set_ylabel("Recall (confidence ≥ 0.25, IoU ≥ 0.50)")
    axis.set_title(
        "Stratified recall by ground-truth object size at 640 px (mean ± sample SD)"
    )
    axis.legend(
        handles=[
            Patch(facecolor="#4C72B0", edgecolor="black", label="FP32"),
            Patch(facecolor="#DD8452", edgecolor="black", label="INT8"),
        ],
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="#888888",
    )
    fig.text(
        0.5,
        -0.02,
        "640 px only. No 320 px stratified small-object analysis exists in "
        "results; no new inference was run.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout()
    save_figure(fig, path)


def plot_system_health(health: dict, path: Path) -> None:
    """Plot logged CPU frequency and temperature across the 24 ABBA runs."""
    rows = sorted(health["per_run"], key=lambda row: row["run_index"])
    indices = np.array([row["run_index"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), sharex=True)
    for axis in axes:
        axis.axvspan(0.5, 12.5, color="#4C72B0", alpha=0.07)
        axis.axvspan(12.5, 24.5, color="#DD8452", alpha=0.09)
        axis.axvline(12.5, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].plot(
        indices,
        [row["pre_freq_mhz"] for row in rows],
        color="#4C72B0",
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        label="Pre-run frequency",
    )
    axes[0].plot(
        indices,
        [row["post_freq_mhz"] for row in rows],
        color="#C44E52",
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="Post-run frequency",
    )
    axes[0].axhline(
        NOMINAL_FREQ_MHZ,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="Nominal 2400 MHz",
    )
    axes[0].set_ylabel("CPU frequency (MHz)")
    axes[0].set_title("Logged CPU frequency and temperature across ABBA runs")
    frequencies = [
        value
        for row in rows
        for value in (row["pre_freq_mhz"], row["post_freq_mhz"])
    ]
    freq_low, freq_high = min(frequencies), max(frequencies)
    freq_margin = max((freq_high - freq_low) * 0.15, 0.005)
    axes[0].set_ylim(freq_low - freq_margin, freq_high + freq_margin)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].text(
        0.01,
        0.06,
        f"Observed range {freq_low:.3f}–{freq_high:.3f} MHz "
        f"({(freq_high - freq_low) / freq_high * 100.0:.4f}%)",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#333333",
    )
    axes[0].legend(loc="lower right", ncol=3)
    axes[1].plot(
        indices,
        [row["pre_temp_c"] for row in rows],
        color="#4C72B0",
        marker="o",
        markersize=3.5,
        linewidth=1.2,
        label="Pre-run temperature",
    )
    axes[1].plot(
        indices,
        [row["post_temp_c"] for row in rows],
        color="#C44E52",
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="Post-run temperature",
    )
    axes[1].axhline(
        TEMP_WARN_C,
        color="#B22222",
        linestyle="--",
        linewidth=1.0,
        label="80 °C warning",
    )
    max_temp = max(row["max_temp_c"] for row in rows)
    max_row = next(row for row in rows if row["max_temp_c"] == max_temp)
    max_y = (
        max_row["pre_temp_c"]
        if max_row["pre_temp_c"] >= max_row["post_temp_c"]
        else max_row["post_temp_c"]
    )
    axes[1].annotate(
        f"max {max_temp:.1f} °C",
        xy=(max_row["run_index"], max_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_ylabel("CPU temperature (°C)")
    axes[1].set_xlabel("ABBA run sequence (1–12: 640 px, 13–24: 320 px)")
    axes[1].set_ylim(35.0, 85.0)
    axes[1].legend(loc="lower right", ncol=3)
    axes[0].text(
        0.28,
        0.93,
        "640 px (23 Nov)",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
        color="#4C72B0",
    )
    axes[0].text(
        0.72,
        0.93,
        "320 px (24 Nov)",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
        color="#C07030",
    )
    axes[1].text(
        0.01,
        0.03,
        "Throttle flags observed: "
        + ", ".join(f"0x{flag:x}" for flag in health["checks"]["throttle_flags_observed"]),
        transform=axes[1].transAxes,
        fontsize=8.5,
        color="#333333",
    )
    axes[1].set_xticks(range(1, len(rows) + 1))
    axes[1].set_xticklabels(
        [
            f"{row['seed']}{row['format'][0].upper()}{row['abba_repeat']}"
            for row in rows
        ],
        fontsize=7,
    )
    fig.tight_layout()
    save_figure(fig, path)


def format_mean_std(metric: dict, digits: int = 4) -> str:
    """Format a metric mean +/- sample SD for the console report."""
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def print_report(
    seed_rows: list[dict],
    summary: list[dict],
    effects: list[dict],
    small: dict,
    health: dict,
    output_dir: Path,
) -> None:
    """Print a concise verification and result report."""
    lookup = summary_lookup(summary)
    print(
        f"Verification: {len(seed_rows)} seed-level cells from "
        f"{EXPECTED_RUNS}/{EXPECTED_RUNS} ABBA runs"
    )
    print(
        "Aggregation: 2 ABBA repeats combined per seed/model; mean ± sample SD "
        "across 3 training seeds (ABBA repeats are not independent training runs)"
    )
    print("\nFinal benchmark summary")
    print(
        f"{'Configuration':<14}{'mAP50':>18}{'mAP50-95':>18}"
        f"{'Latency (ms)':>20}{'FPS':>16}{'Size (MiB)':>16}"
    )
    for resolution in RESOLUTIONS:
        for fmt in FORMATS:
            metrics = lookup[(resolution, fmt)]["metrics"]
            print(
                f"{f'{resolution} {fmt.upper()}':<14}"
                f"{format_mean_std(metrics['mAP50']):>18}"
                f"{format_mean_std(metrics['mAP50_95']):>18}"
                f"{format_mean_std(metrics['inference_mean_ms'], 2):>20}"
                f"{format_mean_std(metrics['inference_fps'], 2):>16}"
                f"{format_mean_std(metrics['model_mib'], 3):>16}"
            )
    print("\nPercentage effects (comparison vs baseline)")
    for effect_type in ("quantization", "resolution"):
        print(f"  {effect_type}:")
        for row in effects:
            if row["effect_type"] != effect_type:
                continue
            if effect_type == "quantization":
                scope = (
                    f"{row['resolution_px']} px "
                    f"{row['baseline']}->{row['comparison']}"
                )
            else:
                scope = f"640->320 px [{row['series'].upper()}]"
            print(
                f"    {scope} {row['metric_label']}: "
                f"{row['percent_change']:+.2f}% ({row['outcome']})"
            )
    print("\nStratified object-size recall")
    for group in small["groups"]:
        print(
            f"    {group['resolution_px']} px {group['size_group']:<6} "
            f"n={group['gt_count']:<3} FP32={group['fp32_recall']['mean']:.4f} "
            f"INT8={group['int8_recall']['mean']:.4f} "
            f"delta={group['delta_int8_minus_fp32']['mean']:+.4f}"
        )
    if small["missing_resolutions_px"]:
        print(
            "    Missing: "
            + ", ".join(f"{value} px" for value in small["missing_resolutions_px"])
            + " stratified analysis (not imputed)"
        )
    print("\nCPU health")
    for row in health["summary"]:
        print(
            f"    {row['resolution_px']} px: freq "
            f"{row['mean_freq_mhz']:.2f} MHz "
            f"[{row['min_freq_mhz']:.2f}, {row['max_freq_mhz']:.2f}], "
            f"max temp {row['max_temp_c']:.1f} °C, "
            f"throttled runs {row['throttled_runs']}/{row['n_runs']}, "
            f"status {row['status']}"
        )
    print(f"\nOutputs written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize 23 Nov 640 px and 24 Nov 320 px ABBA benchmark results"
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing raw results and ABBA logs",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help="Directory for summary CSV/JSON and figures",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    final_dir = args.final_dir.resolve()
    if not results_dir.is_dir():
        fail(f"Results directory does not exist: {results_dir}")

    logs = discover_logs(results_dir)
    runs, log_records = load_runs(results_dir, logs)
    seed_rows = combine_abba(runs)
    summary = build_summary(seed_rows)
    effects = build_effects(summary)
    small = load_small_object_analysis(results_dir)
    health = build_system_health(runs)

    accuracy_diff = max(
        row["mAP50_abba_max_abs_diff"] for row in seed_rows
    )
    accuracy_diff_95 = max(
        row["mAP50_95_abba_max_abs_diff"] for row in seed_rows
    )
    verification = {
        "expected_total_runs": EXPECTED_RUNS,
        "completed_total_runs": len(runs),
        "expected_runs_per_log": EXPECTED_RUNS_PER_LOG,
        "logs": log_records,
        "cells_verified": len(seed_rows),
        "abba_repeats_per_cell": ABBA_REPEATS_PER_CELL,
        "accuracy_repeat_max_abs_diff_mAP50": accuracy_diff,
        "accuracy_repeat_max_abs_diff_mAP50_95": accuracy_diff_95,
        "accuracy_repeat_consistency": (
            accuracy_diff <= 1e-6 and accuracy_diff_95 <= 1e-6
        ),
        "json_csv_log_model_size_checks": "passed",
        "date_filter": {
            str(resolution): EXPECTED_DATE_BY_RESOLUTION[resolution]
            for resolution in RESOLUTIONS
        },
        "status": "passed",
    }
    if not verification["accuracy_repeat_consistency"]:
        fail(
            "ABBA repeats disagree on deterministic mAP by more than 1e-6: "
            f"{accuracy_diff}, {accuracy_diff_95}"
        )
    if not health["checks"]["throttling_absent"]:
        fail("Throttling flags were observed in the logs; analysis stopped")

    final_dir.mkdir(parents=True, exist_ok=True)
    write_csv(final_dir / "abba_seed_results.csv", seed_rows, SEED_COLUMNS)
    write_csv(final_dir / "summary_metrics.csv", summary_rows_for_csv(summary), SUMMARY_COLUMNS)
    write_csv(final_dir / "effects.csv", effects, EFFECT_COLUMNS)
    write_csv(
        final_dir / "small_object_analysis.csv",
        small_object_rows_for_csv(small),
        SMALL_OBJECT_COLUMNS,
    )
    write_csv(final_dir / "system_health.csv", health["per_run"], HEALTH_RUN_COLUMNS)
    write_csv(
        final_dir / "system_health_summary.csv",
        health["summary"],
        HEALTH_SUMMARY_COLUMNS,
    )

    setup_matplotlib()
    plot_accuracy(summary, final_dir / "fig_accuracy_comparison")
    plot_latency_fps(summary, final_dir / "fig_latency_fps")
    plot_effect(
        effects,
        "quantization",
        "Effect of INT8 quantization (relative to FP32)",
        "Change = (INT8 − FP32) / FP32 × 100. Negative latency/size is better; "
        "positive FPS is better; negative mAP is worse.",
        final_dir / "fig_quantization_effect",
    )
    plot_effect(
        effects,
        "resolution",
        "Effect of reducing input resolution from 640 px to 320 px",
        "Change = (320 − 640) / 640 × 100. Negative latency/size is better; "
        "positive FPS is better; negative mAP is worse.",
        final_dir / "fig_resolution_effect",
    )
    plot_small_object(small, final_dir / "fig_small_object_recall")
    plot_system_health(health, final_dir / "fig_system_health")

    missing_data = [
        (
            f"{resolution} px stratified small-object recall: no existing full "
            "analysis file; not imputed and no new inference run"
        )
        for resolution in small["missing_resolutions_px"]
    ]
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "script": "src/summarize_benchmark.py",
            "scope": {
                "included": [
                    "2026-11-23 640 px ABBA campaign",
                    "2026-11-24 320 px ABBA campaign",
                ],
                "excluded": [
                    "older benchmark campaigns",
                    "smoke tests and subset evaluations",
                    "all raw files not referenced by the two ABBA logs",
                ],
            },
            "logs": [log.name for log in logs],
            "seeds": list(SEEDS),
            "formats": list(FORMATS),
            "resolutions_px": list(RESOLUTIONS),
            "abba_repeats_per_seed_model": ABBA_REPEATS_PER_CELL,
            "aggregation": (
                "Two ABBA repeats are averaged per seed/model. Mean and sample "
                "standard deviation (n-1) are computed across the three "
                "training seeds."
            ),
            "accuracy_policy": (
                "ABBA repeats are repeated measurements of the same trained "
                "checkpoint and are not treated as independent training runs. "
                "No n=6 accuracy standard deviation is reported."
            ),
            "percent_change_definition": (
                "(comparison - baseline) / baseline * 100"
            ),
            "sample_standard_deviation": "n-1",
            "raw_data_modified": False,
            "new_benchmark_or_training_started": False,
            "missing_data": missing_data,
            "model_size_units": "1 MiB = 1024^2 bytes",
            "figure_format": "300 dpi PNG and vector PDF",
        },
        "verification": verification,
        "run_inventory": [
            {
                "resolution_px": run["resolution_px"],
                "seed": run["seed"],
                "format": run["format"],
                "run_at": run["run_at"],
                "model": run["result"]["model"],
                "log_file": run["log_file"],
                "json_file": run["local_json"].name,
                "csv_file": run["local_csv"].name,
                "pre_health": run["pre_health"],
                "post_health": run["post_health"],
                "log_metrics": run["log_metrics"],
            }
            for run in sorted(
                runs, key=lambda run: (run["resolution_px"], run["run_at"])
            )
        ],
        "seed_results": seed_rows,
        "summary": summary,
        "effects": effects,
        "small_object_analysis": small,
        "system_health": health,
        "warnings": [
            "Only three training seeds are available; sample standard "
            "deviations are based on n=3 and are sensitive to individual seeds.",
            "The 640 px and 320 px results come from separately trained "
            "checkpoints, so the resolution effect combines training resolution "
            "with inference resolution; it is not a pure input-size ablation.",
            "The benchmark host is recorded as Raspberry Pi 5 in raw metadata; "
            "hardware identity was not independently re-verified here.",
            "All accuracy values refer to one fixed 227-image test split; no "
            "cross-dataset generalization claim is made.",
        ],
    }
    write_json(final_dir / "final_summary.json", payload)
    print_report(seed_rows, summary, effects, small, health, final_dir)


def summary_rows_for_csv(summary: list[dict]) -> list[dict]:
    """Flatten nested summary entries into a wide CSV row."""
    rows = []
    for entry in summary:
        row = {
            "resolution_px": entry["resolution_px"],
            "format": entry["format"],
            "n_seeds": entry["n_seeds"],
            "abba_repeats_per_seed": entry["abba_repeats_per_seed"],
        }
        for metric, stats in entry["metrics"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)
    return rows


def small_object_rows_for_csv(small: dict) -> list[dict]:
    """Flatten stratified analysis entries into CSV rows."""
    rows = []
    for group in small["groups"]:
        rows.append(
            {
                "resolution_px": group["resolution_px"],
                "size_group": group["size_group"],
                "gt_count": group["gt_count"],
                "n_seeds": group["n_seeds"],
                "fp32_recall_mean": group["fp32_recall"]["mean"],
                "fp32_recall_std": group["fp32_recall"]["std"],
                "int8_recall_mean": group["int8_recall"]["mean"],
                "int8_recall_std": group["int8_recall"]["std"],
                "delta_int8_minus_fp32_mean": group["delta_int8_minus_fp32"]["mean"],
                "delta_int8_minus_fp32_std": group["delta_int8_minus_fp32"]["std"],
                "fp32_recall_pct": group["fp32_recall"]["mean"] * 100.0,
                "int8_recall_pct": group["int8_recall"]["mean"] * 100.0,
                "source_json": group["source_json"],
            }
        )
    return rows


if __name__ == "__main__":
    main()
